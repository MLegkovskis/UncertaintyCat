"""OpenTURNS-authoritative inspection and distribution fitting for empirical data."""

from __future__ import annotations

import base64
import io
import math
from typing import Any, Literal

import openturns as ot
import pandas as pd
from pydantic import Field, field_validator

from uncertaintycat_core.contracts import StrictModel
from uncertaintycat_core.errors import InvalidModelError

DatasetKind = Literal["csv", "xlsx", "paste"]
CopulaKind = Literal["independent", "normal", "bernstein"]


class DatasetContent(StrictModel):
    content_base64: str = Field(min_length=1, max_length=20_000_000)
    source_kind: DatasetKind


class DistributionFitRequest(DatasetContent):
    selected_columns: list[str] = Field(min_length=1, max_length=40)
    candidates: list[str] = Field(
        default_factory=lambda: [
            "Normal",
            "Uniform",
            "LogNormal",
            "Exponential",
            "Gamma",
            "Beta",
            "Triangular",
            "KernelSmoothing",
        ],
        min_length=1,
        max_length=8,
    )
    selected_marginals: dict[str, str] = Field(default_factory=dict)
    copula: CopulaKind = "independent"
    significance_level: float = Field(default=0.05, gt=0.0, lt=1.0)

    @field_validator("selected_columns")
    @classmethod
    def unique_columns(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Selected columns must be unique.")
        return value


FACTORIES: dict[str, type[ot.DistributionFactory]] = {
    "Normal": ot.NormalFactory,
    "Uniform": ot.UniformFactory,
    "LogNormal": ot.LogNormalFactory,
    "Exponential": ot.ExponentialFactory,
    "Gamma": ot.GammaFactory,
    "Beta": ot.BetaFactory,
    "Triangular": ot.TriangularFactory,
}


def _dataframe(request: DatasetContent) -> pd.DataFrame:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
    except ValueError as exc:
        raise InvalidModelError("The uploaded dataset is not valid base64 data.") from exc
    if not content:
        raise InvalidModelError("The uploaded dataset is empty.")
    try:
        if request.source_kind == "xlsx":
            frame = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        else:
            text = content.decode("utf-8-sig")
            frame = pd.read_csv(io.StringIO(text))
    except (UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise InvalidModelError(f"The dataset could not be parsed: {exc}") from exc
    if frame.empty or len(frame.columns) == 0:
        raise InvalidModelError("The dataset must contain a header and at least one data row.")
    if len(frame) > 1_000_000 or len(frame.columns) > 200:
        raise InvalidModelError("The dataset exceeds the one-million-row or 200-column limit.")
    frame.columns = [
        str(column).strip() or f"column_{index + 1}" for index, column in enumerate(frame)
    ]
    if len(set(frame.columns)) != len(frame.columns):
        raise InvalidModelError("Dataset column names must be unique after trimming.")
    return frame


def inspect_dataset(request: DatasetContent) -> dict[str, Any]:
    frame = _dataframe(request)
    columns: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name in frame.columns:
        raw = frame[name]
        numeric = pd.to_numeric(raw, errors="coerce")
        missing = int(raw.isna().sum())
        invalid_numeric = int((raw.notna() & numeric.isna()).sum())
        finite = numeric[
            numeric.map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
        ]
        non_finite = int(numeric.notna().sum()) - int(len(finite))
        inferred = "numeric" if invalid_numeric == 0 and int(numeric.notna().sum()) > 0 else "text"
        column = {
            "name": name,
            "type": inferred,
            "missingCount": missing,
            "invalidNumericCount": invalid_numeric,
            "nonFiniteCount": non_finite,
            "finiteCount": int(len(finite)),
            "uniqueCount": int(raw.nunique(dropna=True)),
        }
        if inferred == "numeric" and len(finite):
            sample = ot.Sample([[float(value)] for value in finite])
            column.update(
                minimum=float(sample.getMin()[0]),
                maximum=float(sample.getMax()[0]),
                mean=float(sample.computeMean()[0]),
            )
            if len(finite) < 20:
                warnings.append(f"{name}: fewer than 20 finite observations; fit evidence is weak.")
        if missing:
            warnings.append(
                f"{name}: {missing} missing value(s) will be excluded from marginal fits."
            )
        if invalid_numeric:
            warnings.append(
                f"{name}: {invalid_numeric} non-numeric value(s) prevent numeric fitting."
            )
        if non_finite:
            warnings.append(
                f"{name}: {non_finite} infinite value(s) will be excluded from fitting."
            )
        columns.append(column)
    preview = [
        {name: _json_cell(value) for name, value in row.items()}
        for row in frame.head(20).to_dict(orient="records")
    ]
    return {
        "rowCount": int(len(frame)),
        "columns": columns,
        "preview": preview,
        "warnings": warnings,
    }


def fit_distributions(request: DistributionFitRequest) -> dict[str, Any]:
    frame = _dataframe(request)
    missing_columns = [name for name in request.selected_columns if name not in frame.columns]
    if missing_columns:
        raise InvalidModelError(f"Unknown selected columns: {', '.join(missing_columns)}")
    unknown_candidates = [
        name for name in request.candidates if name not in {*FACTORIES, "KernelSmoothing"}
    ]
    if unknown_candidates:
        raise InvalidModelError(f"Unknown distribution candidates: {', '.join(unknown_candidates)}")

    column_results: list[dict[str, Any]] = []
    fitted: dict[str, ot.Distribution] = {}
    source_samples: dict[str, ot.Sample] = {}
    for name in request.selected_columns:
        numeric = pd.to_numeric(frame[name], errors="coerce")
        values = [
            float(value) for value in numeric if pd.notna(value) and math.isfinite(float(value))
        ]
        if len(values) < 5:
            raise InvalidModelError(f"{name} needs at least five finite numeric observations.")
        sample = ot.Sample([[value] for value in values])
        if float(sample.computeVariance()[0]) <= ot.SpecFunc.ScalarEpsilon:
            raise InvalidModelError(f"{name} is constant; distribution fitting is undefined.")
        source_samples[name] = sample
        rankings: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for candidate in request.candidates:
            try:
                distribution, test = _fit_candidate(sample, candidate, request.significance_level)
                rankings.append(
                    {
                        "candidate": candidate,
                        "distribution": _implementation_name(distribution),
                        "parameters": [float(value) for value in distribution.getParameter()],
                        "parameterDescription": [
                            str(value) for value in distribution.getParameterDescription()
                        ],
                        "bic": _criterion(sample, distribution, candidate, "bic"),
                        "aic": _criterion(sample, distribution, candidate, "aic"),
                        "aicc": _criterion(sample, distribution, candidate, "aicc"),
                        "test": test,
                    }
                )
                if request.selected_marginals.get(name) == candidate:
                    fitted[name] = distribution
            except Exception as exc:
                rejected.append({"candidate": candidate, "reason": str(exc)[:500]})
        rankings.sort(
            key=lambda row: (
                row["bic"] is None,
                float(row["bic"]) if row["bic"] is not None else math.inf,
            )
        )
        if not rankings:
            raise InvalidModelError(f"No requested candidate could be fitted to {name}.")
        selected_name = request.selected_marginals.get(name)
        if selected_name and selected_name not in {row["candidate"] for row in rankings}:
            raise InvalidModelError(
                f"The selected marginal {selected_name} could not be fitted to {name}."
            )
        chart_distribution = fitted.get(name) or _distribution_for_ranking(
            sample, rankings[0]["candidate"]
        )
        column_results.append(
            {
                "column": name,
                "sampleSize": sample.getSize(),
                "warnings": (["Fewer than 20 observations."] if sample.getSize() < 20 else []),
                "rankings": rankings,
                "rejectedCandidates": rejected,
                "plot": _plot_data(sample, chart_distribution),
                "selectedMarginal": selected_name,
            }
        )

    generated_source: str | None = None
    builder_spec: dict[str, Any] | None = None
    copula_result: dict[str, Any] | None = None
    if request.selected_marginals:
        missing = [name for name in request.selected_columns if name not in fitted]
        if missing:
            raise InvalidModelError(
                "Select one successfully fitted marginal for every column before "
                "generating a problem."
            )
        complete = frame[request.selected_columns].apply(pd.to_numeric, errors="coerce")
        complete = complete.dropna()
        complete = complete[
            complete.apply(lambda row: all(math.isfinite(float(value)) for value in row), axis=1)
        ]
        if len(complete) < 5:
            raise InvalidModelError("At least five complete finite rows are required for a copula.")
        multivariate_sample = ot.Sample(
            [[float(value) for value in row] for row in complete.itertuples(index=False, name=None)]
        )
        copula, copula_result = _fit_copula(multivariate_sample, request.copula)
        generated_source = _generated_source(
            request.selected_columns,
            fitted,
            source_samples,
            request.selected_marginals,
            copula,
            request.copula,
        )
        builder_spec = {
            "inputs": [
                {
                    "name": name,
                    "distribution": request.selected_marginals[name],
                    "parameters": [float(value) for value in fitted[name].getParameter()],
                    "source": "data_fit",
                }
                for name in request.selected_columns
            ],
            "dependence": request.copula,
            "copula": copula_result,
        }

    return {
        "openturnsVersion": ot.__version__,
        "columns": column_results,
        "copula": copula_result,
        "generatedSource": generated_source,
        "builderSpec": builder_spec,
        "assumptions": [
            "Marginal candidate parameters and information criteria are computed by OpenTURNS.",
            "Goodness-of-fit tests do not prove that a fitted family is the data-generating law.",
            "Copula fitting uses complete finite rows only and must be selected explicitly.",
        ],
    }


def _fit_candidate(
    sample: ot.Sample, candidate: str, level: float
) -> tuple[ot.Distribution, dict[str, Any]]:
    if candidate == "KernelSmoothing":
        distribution = ot.KernelSmoothing().build(sample)
        test_result = ot.FittingTest.Kolmogorov(sample, distribution, level)
        test_name = "Kolmogorov (parameters not re-estimated by this test)"
    else:
        factory = FACTORIES[candidate]()
        estimator = factory.buildEstimator(sample)
        distribution = estimator.getDistribution()
        _, test_result = ot.FittingTest.Lilliefors(sample, factory, level)
        test_name = "Lilliefors"
    return distribution, {
        "name": test_name,
        "statistic": float(test_result.getStatistic()),
        "pValue": float(test_result.getPValue()),
        "significanceLevel": level,
        "rejected": not bool(test_result.getBinaryQualityMeasure()),
    }


def _distribution_for_ranking(sample: ot.Sample, candidate: str) -> ot.Distribution:
    return _fit_candidate(sample, str(candidate), 0.05)[0]


def _implementation_name(distribution: ot.Distribution) -> str:
    implementation = getattr(distribution, "getImplementation", None)
    return str(implementation().getClassName()) if implementation else distribution.getClassName()


def _criterion(
    sample: ot.Sample, distribution: ot.Distribution, candidate: str, criterion: str
) -> float | None:
    if candidate == "KernelSmoothing":
        return None
    parameter_count = distribution.getParameterDimension()
    method = {
        "bic": ot.FittingTest.BIC,
        "aic": ot.FittingTest.AIC,
        "aicc": ot.FittingTest.AICC,
    }[criterion]
    return float(method(sample, distribution, parameter_count))


def _fit_copula(sample: ot.Sample, kind: CopulaKind) -> tuple[ot.Distribution, dict[str, Any]]:
    try:
        if sample.getDimension() == 1 or kind == "independent":
            copula: ot.Distribution = ot.IndependentCopula(sample.getDimension())
        elif kind == "normal":
            copula = ot.NormalCopulaFactory().build(sample)
        else:
            copula = ot.BernsteinCopulaFactory().build(sample)
    except Exception as exc:
        raise InvalidModelError(
            f"The selected {kind} copula could not be fitted to the complete rows: {exc}"
        ) from exc
    result: dict[str, Any] = {"kind": kind, "className": _implementation_name(copula)}
    if kind == "normal" and sample.getDimension() > 1:
        correlation = copula.getCorrelation()
        result["correlation"] = [
            [float(correlation[i, j]) for j in range(correlation.getDimension())]
            for i in range(correlation.getDimension())
        ]
    return copula, result


def _plot_data(sample: ot.Sample, distribution: ot.Distribution) -> dict[str, Any]:
    minimum = float(sample.getMin()[0])
    maximum = float(sample.getMax()[0])
    if minimum == maximum:
        maximum = minimum + 1.0
    grid = ot.RegularGrid(minimum, (maximum - minimum) / 100.0, 101).getVertices()
    x = [float(row[0]) for row in grid]
    pdf = [float(distribution.computePDF(row)) for row in grid]
    fitted_cdf = [float(distribution.computeCDF(row)) for row in grid]
    ordered = sample.sort()
    empirical_x = [float(row[0]) for row in ordered]
    empirical_cdf = [(index + 1) / sample.getSize() for index in range(sample.getSize())]
    probabilities = [(index + 0.5) / sample.getSize() for index in range(sample.getSize())]
    theoretical = [
        float(distribution.computeQuantile(probability)[0]) for probability in probabilities
    ]
    return {
        "sample": [float(row[0]) for row in sample[: min(sample.getSize(), 2_000)]],
        "pdf": {"x": x, "y": pdf},
        "cdf": {
            "empiricalX": empirical_x[:2_000],
            "empiricalY": empirical_cdf[:2_000],
            "fittedX": x,
            "fittedY": fitted_cdf,
        },
        "qq": {
            "theoretical": theoretical[:2_000],
            "observed": empirical_x[:2_000],
        },
    }


def _generated_source(
    columns: list[str],
    fitted: dict[str, ot.Distribution],
    samples: dict[str, ot.Sample],
    selected_marginals: dict[str, str],
    copula: ot.Distribution,
    kind: CopulaKind,
) -> str:
    lines = ["import openturns as ot", "", "marginals = ["]
    for name in columns:
        distribution = fitted[name]
        selected = selected_marginals[name]
        if selected == "KernelSmoothing":
            values = [float(row[0]) for row in samples[name]]
            lines.append(
                f"    ot.KernelSmoothing().build(ot.Sample({[[value] for value in values]!r})),"
            )
        else:
            parameters = ", ".join(repr(float(value)) for value in distribution.getParameter())
            lines.append(f"    ot.{selected}({parameters}),")
    lines.append("]")
    if kind == "normal" and len(columns) > 1:
        correlation = copula.getCorrelation()
        lines.extend([f"correlation = ot.CorrelationMatrix({len(columns)})"])
        for i in range(len(columns)):
            for j in range(i):
                lines.append(f"correlation[{i}, {j}] = {float(correlation[i, j])!r}")
        lines.append("copula = ot.NormalCopula(correlation)")
    elif kind == "bernstein" and len(columns) > 1:
        implementation = copula.getImplementation()
        copula_sample = [list(map(float, row)) for row in implementation.getCopulaSample()]
        bin_number = int(implementation.getBinNumber())
        lines.append(f"copula_sample = ot.Sample({copula_sample!r})")
        lines.append(f"copula = ot.EmpiricalBernsteinCopula(copula_sample, {bin_number}, True)")
    else:
        lines.append(f"copula = ot.IndependentCopula({len(columns)})")
    lines.extend(
        [
            "problem = ot.JointDistribution(marginals, copula)",
            f"problem.setDescription({columns!r})",
        ]
    )
    return "\n".join(lines) + "\n"


def _json_cell(value: Any) -> str | int | float | bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)
