"""Least-squares polynomial chaos surrogate and validation."""

from __future__ import annotations

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, SeriesData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.plugins.sobol import SobolPlugin


class PceConfig(StrictModel):
    degree: int = Field(default=3, ge=1, le=12)
    training_size: int = Field(default=1000, ge=30, le=200_000)
    validation_size: int = Field(default=500, ge=20, le=50_000)
    sparse: bool = True
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class PcePlugin(AnalysisPlugin[PceConfig]):
    key = "pce"
    version = "2.0.0"
    name = "Polynomial Chaos Surrogate"
    category = "Surrogate"
    description = "Fit and independently validate a polynomial chaos metamodel."
    assumptions = (
        "The selected polynomial degree and training design can resolve the response.",
        "Validation Q2 should be checked before using the surrogate downstream.",
    )
    supports_dependent_inputs = False
    supports_multi_output = False
    resource_class = "heavy"
    config_model = PceConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: PceConfig) -> list[str]:
        if not runtime.problem.isContinuous():
            raise IncompatibleAnalysisError(
                "Polynomial chaos currently requires continuous input marginals."
            )
        if not SobolPlugin._is_independent(runtime.problem):
            raise IncompatibleAnalysisError(
                "This polynomial-chaos basis requires independent input marginals."
            )
        return []

    def run(self, runtime: ModelRuntime, config: PceConfig) -> tuple[AnalysisPayload, int]:
        result, basis_size = fit_pce(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        ot.RandomGenerator.SetSeed(config.seed + 1)
        validation_x = runtime.problem.getSample(config.validation_size)
        observed_sample = runtime.model(validation_x).getMarginal(target)
        predicted_sample = result.getMetaModel()(validation_x)
        if float(observed_sample.computeVariance()[0]) <= ot.SpecFunc.ScalarEpsilon:
            raise IncompatibleAnalysisError(
                "PCE hold-out Q2 is undefined because the validation output is constant."
            )
        validation = ot.MetaModelValidation(observed_sample, predicted_sample)
        q2 = float(validation.computeR2Score()[0])
        rmse = float(validation.computeMeanSquaredError()[0]) ** 0.5
        output_standard_deviation = float(observed_sample.computeStandardDeviation()[0])
        normalized_rmse = rmse / output_standard_deviation
        coefficients = [float(row[0]) for row in result.getCoefficients()]
        selected_indices = list(result.getIndices())
        coefficient_rows = sorted(
            [[int(selected_indices[i]), coefficients[i]] for i in range(len(coefficients))],
            key=lambda row: abs(float(row[1])),
            reverse=True,
        )[:100]
        metrics: dict[str, str | int | float | bool | None] = {
            "degree": config.degree,
            "training_size": config.training_size,
            "validation_size": config.validation_size,
            "basis_size": int(basis_size),
            "retained_terms": len(coefficients),
            "validation_q2": q2,
            "validation_rmse": rmse,
            "validation_normalized_rmse": normalized_rmse,
        }
        if not config.sparse:
            try:
                analytical_validation = ot.FunctionalChaosValidation(result)
                metrics["analytical_cross_validation_q2"] = float(
                    analytical_validation.computeR2Score()[0]
                )
            except Exception:
                # OpenTURNS documents restrictions for analytical validation.
                # Independent hold-out validation above remains authoritative.
                metrics["analytical_cross_validation_q2"] = None
        sobol = ot.FunctionalChaosSobolIndices(result)
        sobol_rows = [
            [
                runtime.metadata.inputs[index].name,
                float(sobol.getSobolIndex(index)),
                float(sobol.getSobolTotalIndex(index)),
            ]
            for index in range(runtime.metadata.input_dimension)
        ]
        return AnalysisPayload(
            metrics=metrics,
            tables={
                "coefficients": TableData(
                    columns=["Basis Index", "Coefficient"],
                    rows=coefficient_rows,
                    row_count=len(coefficients),
                    truncated=len(coefficients) > len(coefficient_rows),
                ),
                "pce_sobol_indices": TableData(
                    columns=["Input", "First Order", "Total Order"],
                    rows=sobol_rows,
                    row_count=len(sobol_rows),
                ),
            },
            series={
                "validation": SeriesData(
                    name="PCE validation",
                    x=[float(row[0]) for row in observed_sample],
                    y=[float(row[0]) for row in predicted_sample],
                    x_label="Observed",
                    y_label="Predicted",
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "sparse_selection": config.sparse,
            },
        ), config.training_size + config.validation_size


def fit_pce(runtime: ModelRuntime, config: PceConfig) -> tuple[ot.FunctionalChaosResult, int]:
    """Fit the exact PCE persisted by Surrogate Studio."""
    plugin.applicability_warnings(runtime, config)
    target = config.output_targets[0] if config.output_targets else 0
    if target >= runtime.metadata.output_dimension:
        raise IncompatibleAnalysisError("The requested output target does not exist.")
    ot.RandomGenerator.SetSeed(config.seed)
    basis = build_pce_basis(runtime.problem)
    enumeration = basis.getEnumerateFunction()
    basis_size = enumeration.getBasisSizeFromTotalDegree(config.degree)
    strategy = ot.FixedStrategy(basis, basis_size)
    training_x = runtime.problem.getSample(config.training_size)
    training_y = runtime.model(training_x).getMarginal(target)
    projection = (
        ot.LeastSquaresStrategy(training_x, training_y, ot.LeastSquaresMetaModelSelectionFactory())
        if config.sparse
        else ot.LeastSquaresStrategy(training_x, training_y)
    )
    try:
        algorithm = ot.FunctionalChaosAlgorithm(
            training_x, training_y, runtime.problem, strategy, projection
        )
        algorithm.run()
        return algorithm.getResult(), int(basis_size)
    except Exception as exc:
        raise IncompatibleAnalysisError(
            f"PCE construction failed for this distribution and configuration: {exc}"
        ) from exc


def build_pce_basis(distribution: ot.Distribution) -> ot.OrthogonalProductPolynomialFactory:
    """Build the marginal-product basis shared by PCE-backed analyses."""
    collection = ot.PolynomialFamilyCollection(distribution.getDimension())
    for index in range(distribution.getDimension()):
        collection[index] = ot.StandardDistributionPolynomialFactory(
            distribution.getMarginal(index)
        )
    return ot.OrthogonalProductPolynomialFactory(collection)


plugin = PcePlugin()
