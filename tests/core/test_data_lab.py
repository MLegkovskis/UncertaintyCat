from __future__ import annotations

import base64
import io

import pandas as pd
import pytest

from uncertaintycat_core.data_lab import (
    DatasetContent,
    DistributionFitRequest,
    fit_distributions,
    inspect_dataset,
)
from uncertaintycat_core.errors import InvalidModelError


def encoded(value: bytes | str) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return base64.b64encode(raw).decode()


CSV = """temperature,pressure,label
18.2,1.04,A
19.1,1.01,B
20.0,1.08,C
21.3,1.11,D
22.1,1.15,E
23.0,1.19,F
24.2,1.24,G
25.4,1.29,H
26.8,1.35,I
28.1,1.42,J
"""


def test_dataset_inspection_types_and_previews_csv_and_xlsx() -> None:
    csv = inspect_dataset(DatasetContent(content_base64=encoded(CSV), source_kind="csv"))
    assert csv["rowCount"] == 10
    assert [column["type"] for column in csv["columns"]] == ["numeric", "numeric", "text"]
    assert csv["preview"][0]["temperature"] == 18.2

    buffer = io.BytesIO()
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}).to_excel(buffer, index=False)
    xlsx = inspect_dataset(
        DatasetContent(content_base64=encoded(buffer.getvalue()), source_kind="xlsx")
    )
    assert xlsx["rowCount"] == 3
    assert all(column["type"] == "numeric" for column in xlsx["columns"])


@pytest.mark.scientific
def test_distribution_fit_ranks_candidates_and_requires_explicit_composition() -> None:
    request = DistributionFitRequest(
        content_base64=encoded(CSV),
        source_kind="csv",
        selected_columns=["temperature", "pressure"],
        candidates=["Normal", "Uniform", "KernelSmoothing"],
    )
    ranked = fit_distributions(request)
    assert ranked["generatedSource"] is None
    assert all(column["rankings"] for column in ranked["columns"])
    assert all(column["rankings"][0]["bic"] is not None for column in ranked["columns"])

    selected = {
        column["column"]: column["rankings"][0]["candidate"] for column in ranked["columns"]
    }
    composed = fit_distributions(
        request.model_copy(update={"selected_marginals": selected, "copula": "normal"})
    )
    assert composed["copula"]["className"] == "NormalCopula"
    assert "problem = ot.JointDistribution" in composed["generatedSource"]
    namespace: dict[str, object] = {}
    exec(composed["generatedSource"], namespace)
    assert namespace["problem"].getDimension() == 2  # type: ignore[attr-defined]


def test_distribution_fit_rejects_constant_and_non_finite_only_columns() -> None:
    constant = "x,y\n1,nan\n1,inf\n1,-inf\n1,nan\n1,inf\n"
    inspection = inspect_dataset(
        DatasetContent(content_base64=encoded(constant), source_kind="paste")
    )
    assert inspection["columns"][1]["nonFiniteCount"] == 3
    assert any("infinite value" in warning for warning in inspection["warnings"])
    with pytest.raises(InvalidModelError, match="constant"):
        fit_distributions(
            DistributionFitRequest(
                content_base64=encoded(constant),
                source_kind="paste",
                selected_columns=["x"],
                candidates=["Normal"],
            )
        )
    with pytest.raises(InvalidModelError, match="five finite"):
        fit_distributions(
            DistributionFitRequest(
                content_base64=encoded(constant),
                source_kind="paste",
                selected_columns=["y"],
                candidates=["Normal"],
            )
        )


def test_distribution_fit_is_repeatable_without_caller_rng_management() -> None:
    request = DistributionFitRequest(
        content_base64=encoded(CSV),
        source_kind="csv",
        selected_columns=["temperature", "pressure"],
        candidates=["Normal", "Uniform"],
    )
    first = fit_distributions(request)
    second = fit_distributions(request)
    assert second == first


def test_distribution_fit_candidate_evidence_survives_composition_and_order() -> None:
    request = DistributionFitRequest(
        content_base64=encoded(CSV),
        source_kind="csv",
        selected_columns=["temperature", "pressure"],
        candidates=["Normal", "Uniform"],
    )

    def candidate_evidence(result: dict) -> dict:
        return {
            column["column"]: {ranking["candidate"]: ranking for ranking in column["rankings"]}
            for column in result["columns"]
        }

    ranked = candidate_evidence(fit_distributions(request))
    composed = candidate_evidence(
        fit_distributions(
            request.model_copy(
                update={"selected_marginals": {"temperature": "Normal", "pressure": "Uniform"}}
            )
        )
    )
    reordered = candidate_evidence(
        fit_distributions(
            request.model_copy(
                update={
                    "selected_columns": ["pressure", "temperature"],
                    "candidates": ["Uniform", "Normal"],
                }
            )
        )
    )
    assert composed == ranked
    assert reordered == ranked


@pytest.mark.scientific
def test_distribution_fit_matches_direct_openturns_seeded_lilliefors() -> None:
    import openturns as ot

    request = DistributionFitRequest(
        content_base64=encoded(CSV),
        source_kind="csv",
        selected_columns=["temperature"],
        candidates=["Normal"],
    )
    actual = fit_distributions(request)
    # Independently pinned stream for the documented SHA-256 tuple
    # [42,"temperature","Normal"]; do not call the implementation seed logic.
    ot.RandomGenerator.SetSeed(1_940_970_326)
    sample = ot.Sample(
        [[value] for value in [18.2, 19.1, 20, 21.3, 22.1, 23, 24.2, 25.4, 26.8, 28.1]]
    )
    factory = ot.NormalFactory()
    distribution = factory.buildEstimator(sample).getDistribution()
    _, reference = ot.FittingTest.Lilliefors(sample, factory, 0.05)
    ranking = actual["columns"][0]["rankings"][0]
    assert ranking["parameters"] == list(distribution.getParameter())
    assert ranking["test"]["pValue"] == float(reference.getPValue())
    assert ranking["test"]["statistic"] == float(reference.getStatistic())
    assert actual["seed"] == 42
    assert actual["fittingVersion"] == "1.1.0"


def test_distribution_fit_calibrates_each_candidate_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import openturns as ot

    calls = 0
    original = ot.FittingTest.Lilliefors

    def counted(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(ot.FittingTest, "Lilliefors", counted)
    fit_distributions(
        DistributionFitRequest(
            content_base64=encoded(CSV),
            source_kind="csv",
            selected_columns=["temperature", "pressure"],
            candidates=["Normal", "Uniform"],
        )
    )
    # Independent loop-count oracle: two observed columns times two families.
    assert calls == 4


@pytest.mark.parametrize("seed", [-1, 2_147_483_648, 1.5])
def test_distribution_fit_rejects_invalid_seed_before_computation(seed: float) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="seed"):
        DistributionFitRequest(
            content_base64=encoded(CSV),
            source_kind="csv",
            selected_columns=["temperature"],
            candidates=["Normal"],
            seed=seed,  # type: ignore[arg-type]
        )
