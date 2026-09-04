"""Independent scientific and resource evidence for the bounded subset path."""

from __future__ import annotations

import json
import math

import openturns as ot
import pytest
from pydantic import ValidationError

from uncertaintycat_core import compile_model, run_analysis
from uncertaintycat_core.contracts import AnalysisRequest
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.plugins import reliability
from uncertaintycat_core.plugins.reliability import (
    ReliabilityConfig,
    plugin,
    subset_evaluation_limit,
)

RS_SOURCE = """import openturns as ot
model = ot.SymbolicFunction(["R", "S"], ["R-S"])
problem = ot.Normal([7.0, 2.0], [1.0, 1.0], ot.IdentityMatrix(2))
problem.setDescription(["R", "S"])
"""
NORMAL_SOURCE = """import openturns as ot
model = ot.SymbolicFunction(["x"], ["x"])
problem = ot.Normal()
"""


def _request(**config: object) -> AnalysisRequest:
    return AnalysisRequest(
        analysis_key="reliability",
        config={"method": "SUBSET_SAMPLING", "threshold": 0.0, "operator": "<", **config},
    )


@pytest.mark.scientific
def test_subset_official_resistance_stress_benchmark_repeatably() -> None:
    results = [run_analysis(compile_model(RS_SOURCE), _request(), seed=42) for _ in range(2)]
    first, second = results
    assert first.payload == second.payload
    # Independent analytical R-S ~ N(5, sqrt(2)), not a production estimator oracle.
    expected = 0.5 * math.erfc(2.5)
    assert first.payload.metrics["event_probability"] == pytest.approx(expected, abs=8e-5)
    assert first.payload.metrics["event_probability"] == pytest.approx(0.000222, abs=1e-12)
    assert first.runtime.model_evaluations == second.runtime.model_evaluations == 8_000
    assert first.plugin_version == "3.0.0"
    assert first.result_schema_version == "1.0.0"
    assert first.payload.facts["stopping_reason"] == "requested event threshold reached"
    assert first.payload.tables["subset_levels"].rows[-1][1] == 0.0
    assert first.payload.series == {}  # no resetting pseudo-convergence trace
    assert any("not an exact confidence guarantee" in warning for warning in first.warnings)
    assert len(first.payload.model_dump_json().encode()) < 16_384
    json.dumps(first.model_dump(mode="json"), allow_nan=False)

    # Stable OpenTURNS construction independently bypassing the production wrapper.
    f = ot.SymbolicFunction(["R", "S"], ["R-S"])
    event = ot.ThresholdEvent(
        ot.CompositeRandomVector(
            f, ot.RandomVector(ot.Normal([7.0, 2.0], [1.0, 1.0], ot.IdentityMatrix(2)))
        ),
        ot.Less(),
        0.0,
    )
    ot.RandomGenerator.SetSeed(42)
    algorithm = ot.SubsetSampling(event, 2.0, 0.1)
    algorithm.setMaximumOuterSampling(2_000)
    algorithm.setBlockSize(1)
    algorithm.run()
    official = algorithm.getResult()
    interval = official.getProbabilityDistribution().computeBilateralConfidenceInterval(0.95)
    assert first.payload.metrics["event_probability"] == official.getProbabilityEstimate()
    assert first.payload.metrics["coefficient_of_variation"] == official.getCoefficientOfVariation()
    assert first.payload.metrics["confidence_lower"] == max(0.0, interval.getLowerBound()[0])
    assert first.payload.metrics["confidence_upper"] == min(1.0, interval.getUpperBound()[0])
    assert f.getEvaluationCallsNumber() == first.runtime.model_evaluations


@pytest.mark.scientific
def test_subset_budget_is_guarded_before_each_original_model_invocation() -> None:
    runtime = compile_model(NORMAL_SOURCE)
    invocations = 0

    def counted(point: ot.Point) -> list[float]:
        nonlocal invocations
        invocations += 1
        return [float(point[0])]

    runtime.model = ot.PythonFunction(1, 1, counted)
    with pytest.raises(IncompatibleAnalysisError, match="exhausted") as error:
        run_analysis(
            runtime,
            _request(
                threshold=4.0, operator=">", maximum_evaluations=1_000, subset_sample_size=1_000
            ),
        )
    assert invocations == 1_000
    assert error.value.details["model_evaluations"] == invocations
    assert error.value.details["effective_evaluation_limit"] == 1_000


def _independent_population_oracle(population: int, requested: int) -> int:
    calls = 0
    for _ in range(10):
        if calls + population > requested:
            break
        for _ in range(population):
            calls += 1
    return calls


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("population", "budget"), [(100, 100), (100, 199), (1000, 1550), (2000, 20000), (5000, 50000)]
)
def test_subset_resource_limit_matches_independent_population_loop_oracle(
    population: int, budget: int
) -> None:
    assert subset_evaluation_limit(population, budget) == _independent_population_oracle(
        population, budget
    )
    # Pinned computeVarianceGamma nests lag, seed and paired-chain positions.
    # The fixed conditional probability and multiples-of-ten contract give length 10.
    seed_count = population // 10
    products_per_level = sum(
        1 for lag in range(9) for _ in range(seed_count) for _ in range(10 - lag - 1)
    )
    assert products_per_level == 45 * seed_count
    assert products_per_level * 9 <= 202_500  # max 9 conditional populations


@pytest.mark.scientific
def test_subset_ui_default_at_maximum_dimension_is_admissible() -> None:
    source = """import openturns as ot
names = [f"x{i}" for i in range(20)]
model = ot.SymbolicFunction(names, ["(" + "+".join(names) + ")/sqrt(20)"])
problem = ot.Normal(20)
"""
    runtime = compile_model(source)
    config = ReliabilityConfig(method="SUBSET_SAMPLING", threshold=4.0)
    assert config.subset_sample_size == 2_000
    assert config.maximum_evaluations == 20_000
    assert (
        subset_evaluation_limit(2_000, 20_000)
        == _independent_population_oracle(2_000, 20_000)
        == 20_000
    )
    assert plugin.applicability_warnings(runtime, config)
    guidance = next(
        item for item in runtime.assessment.recommendations if item.capability == "reliability"
    )
    assert guidance.safe_config["subset_sampling_available"] is True
    result = run_analysis(runtime, _request(threshold=4.0, operator=">"), seed=42)
    assert result.payload.metrics["event_probability"] == pytest.approx(
        0.5 * math.erfc(4 / math.sqrt(2)), abs=2e-5
    )
    assert result.runtime.model_evaluations == 10_000


@pytest.mark.scientific
def test_subset_first_over_budget_and_schema_maximum_rejected_before_sampling() -> None:
    runtime = compile_model(NORMAL_SOURCE)
    before = runtime.model.getEvaluationCallsNumber()
    plugin.applicability_warnings(
        runtime,
        ReliabilityConfig(
            method="SUBSET_SAMPLING",
            threshold=4,
            subset_sample_size=5000,
            maximum_evaluations=50000,
        ),
    )
    for budget in [50_001, 2_000_000]:
        with pytest.raises(IncompatibleAnalysisError, match="50,000"):
            run_analysis(runtime, _request(maximum_evaluations=budget))
    assert runtime.model.getEvaluationCallsNumber() == before


@pytest.mark.parametrize(
    "config",
    [
        {"subset_sample_size": 99},
        {"subset_sample_size": 5001},
        {"subset_sample_size": 101},
        {"subset_sample_size": 100.0},
        {"subset_sample_size": "100"},
        {"threshold": math.nan},
        {"threshold": math.inf},
        {"output_targets": [0, 0]},
        {"unknown": 1},
    ],
)
def test_subset_strict_contract_rejects_invalid_settings(config: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        run_analysis(compile_model(NORMAL_SOURCE), _request(**config))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"output_targets": [-1]}, "output target"),
        ({"output_targets": [1]}, "output target"),
        ({"maximum_evaluations": 1000}, "per level"),
        ({"block_size": 10}, "block_size=1"),
        ({"sample_size": 1000}, "ambiguous"),
    ],
)
def test_subset_semantic_rejections_before_evaluation(
    config: dict[str, object], message: str
) -> None:
    runtime = compile_model(NORMAL_SOURCE)
    before = runtime.model.getEvaluationCallsNumber()
    with pytest.raises(IncompatibleAnalysisError, match=message):
        run_analysis(runtime, _request(**config))
    assert runtime.model.getEvaluationCallsNumber() == before


@pytest.mark.parametrize(
    "source",
    [
        NORMAL_SOURCE.replace("ot.Normal()", "ot.Poisson(3.0)"),
        NORMAL_SOURCE.replace('["x"], ["x"]', '["x"], ["1.0"]'),
        'import openturns as ot\nnames=[f"x{i}" for i in range(21)]\n'
        'model=ot.SymbolicFunction(names,["+".join(names)])\nproblem=ot.Normal(21)',
    ],
)
def test_subset_rejects_inapplicable_model(source: str) -> None:
    runtime = compile_model(source)
    before = runtime.model.getEvaluationCallsNumber()
    with pytest.raises(IncompatibleAnalysisError):
        run_analysis(runtime, _request())
    assert runtime.model.getEvaluationCallsNumber() == before


def test_subset_nonfinite_model_failure_and_timeout_are_source_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = compile_model(NORMAL_SOURCE)
    runtime.model = ot.PythonFunction(1, 1, lambda _: [math.nan])
    with pytest.raises(IncompatibleAnalysisError, match="non-finite"):
        run_analysis(runtime, _request())

    def failing(_: ot.Point) -> list[float]:
        raise ValueError("PRIVATE_USER_EXCEPTION_MARKER")

    runtime.model = ot.PythonFunction(1, 1, failing)
    with pytest.raises(IncompatibleAnalysisError, match="model failed") as error:
        run_analysis(runtime, _request())
    assert "PRIVATE_USER_EXCEPTION_MARKER" not in str(error.value)
    monkeypatch.setattr(reliability, "SUBSET_MAXIMUM_SECONDS", 0.000000001)
    with pytest.raises(IncompatibleAnalysisError, match="time budget"):
        run_analysis(compile_model(NORMAL_SOURCE), _request())


def test_subset_degenerate_event_does_not_claim_certainty() -> None:
    with pytest.raises(IncompatibleAnalysisError, match="does not prove"):
        run_analysis(compile_model(NORMAL_SOURCE), _request(threshold=1000.0))


def test_subset_legacy_precision_is_unused_and_progress_is_bounded() -> None:
    progress: list[tuple[str, int, str, bool]] = []
    results = [
        run_analysis(
            compile_model(RS_SOURCE),
            _request(target_coefficient_of_variation=value),
            progress_callback=lambda *args: progress.append(args),
        )
        for value in [0.8, 0.001]
    ]
    assert results[0].payload == results[1].payload
    populations = [event for event in progress if event[0] == "subset_population"]
    assert len(populations) == 8
    assert all(event[3] and len(event[2]) < 150 for event in populations)
    assert all("R-S" not in event[2] for event in progress)


def test_subset_continuous_dependence_and_selected_output() -> None:
    source = """import openturns as ot
model = ot.SymbolicFunction(["x", "z"], ["1", "x+z"])
correlation = ot.CorrelationMatrix(2)
correlation[0,1] = 0.5
problem = ot.Normal([0.,0.], [1.,1.], correlation)
"""
    runtime = compile_model(source)
    result = run_analysis(runtime, _request(threshold=0.0, output_targets=[1]))
    assert result.payload.metrics["event_probability"] == pytest.approx(0.5, abs=0.04)
    with pytest.raises(IncompatibleAnalysisError, match="varies"):
        run_analysis(runtime, _request(output_targets=[0]))


def test_subset_conflicting_targets_and_historical_version_are_explicit() -> None:
    with pytest.raises(IncompatibleAnalysisError, match="Conflicting"):
        plugin.parse_config(
            {"method": "SUBSET_SAMPLING", "threshold": 0.0, "output_targets": [1]},
            seed=42,
            output_targets=[0],
        )
    with pytest.raises(IncompatibleAnalysisError, match="version"):
        run_analysis(
            compile_model(RS_SOURCE),
            AnalysisRequest(
                analysis_key="reliability",
                plugin_version="2.0.0",
                config={"method": "SUBSET_SAMPLING", "threshold": 0.0},
            ),
        )


def test_subset_rejects_partial_threshold_and_oversized_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ot.SubsetSampling.getThresholdPerStep

    def incomplete(algorithm: ot.SubsetSampling) -> ot.Point:
        thresholds = original(algorithm)
        thresholds[-1] = 1.0
        return thresholds

    monkeypatch.setattr(ot.SubsetSampling, "getThresholdPerStep", incomplete)
    with pytest.raises(IncompatibleAnalysisError, match="no intermediate-domain"):
        run_analysis(compile_model(RS_SOURCE), _request())
    monkeypatch.setattr(ot.SubsetSampling, "getThresholdPerStep", original)
    monkeypatch.setattr(reliability, "SUBSET_MAXIMUM_PAYLOAD_BYTES", 1)
    with pytest.raises(IncompatibleAnalysisError, match="report size"):
        run_analysis(compile_model(RS_SOURCE), _request())


@pytest.mark.scientific
def test_subset_maximum_budget_cannot_escape_guard() -> None:
    runtime = compile_model(NORMAL_SOURCE)
    before = runtime.model.getEvaluationCallsNumber()
    with pytest.raises(IncompatibleAnalysisError, match="exhausted") as error:
        run_analysis(
            runtime,
            _request(
                threshold=20.0,
                operator=">",
                subset_sample_size=5_000,
                maximum_evaluations=50_000,
            ),
        )
    assert runtime.model.getEvaluationCallsNumber() - before == 50_000
    assert error.value.details["model_evaluations"] == 50_000
