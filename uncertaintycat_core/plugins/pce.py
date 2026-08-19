"""Least-squares polynomial chaos surrogate and validation."""

from __future__ import annotations

import numpy as np
import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, SeriesData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class PceConfig(StrictModel):
    degree: int = Field(default=3, ge=1, le=12)
    training_size: int = Field(default=1000, ge=30, le=200_000)
    validation_size: int = Field(default=500, ge=20, le=50_000)
    sparse: bool = True
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class PcePlugin(AnalysisPlugin[PceConfig]):
    key = "pce"
    version = "1.0.0"
    name = "Polynomial Chaos Surrogate"
    category = "Surrogate"
    description = "Fit and independently validate a polynomial chaos metamodel."
    assumptions = (
        "The selected polynomial degree and training design can resolve the response.",
        "Validation Q2 should be checked before using the surrogate downstream.",
    )
    supports_multi_output = False
    resource_class = "heavy"
    config_model = PceConfig

    def run(self, runtime: ModelRuntime, config: PceConfig) -> tuple[AnalysisPayload, int]:
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        ot.RandomGenerator.SetSeed(config.seed)
        basis = _build_basis(runtime.problem)
        enumeration = basis.getEnumerateFunction()
        basis_size = enumeration.getBasisSizeFromTotalDegree(config.degree)
        strategy = ot.FixedStrategy(basis, basis_size)
        training_x = runtime.problem.getSample(config.training_size)
        training_y = runtime.model(training_x).getMarginal(target)
        projection = (
            ot.LeastSquaresStrategy(
                training_x, training_y, ot.LeastSquaresMetaModelSelectionFactory()
            )
            if config.sparse
            else ot.LeastSquaresStrategy(training_x, training_y)
        )
        try:
            algorithm = ot.FunctionalChaosAlgorithm(
                training_x, training_y, runtime.problem, strategy, projection
            )
            algorithm.run()
            result = algorithm.getResult()
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"PCE construction failed for this distribution and configuration: {exc}"
            ) from exc
        ot.RandomGenerator.SetSeed(config.seed + 1)
        validation_x = runtime.problem.getSample(config.validation_size)
        observed = np.asarray(runtime.model(validation_x).getMarginal(target), dtype=float).reshape(
            -1
        )
        predicted = np.asarray(result.getMetaModel()(validation_x), dtype=float).reshape(-1)
        residual = observed - predicted
        denominator = float(np.sum((observed - observed.mean()) ** 2))
        q2 = (
            1.0 - float(np.sum(residual**2)) / denominator
            if denominator > np.finfo(float).eps
            else 0.0
        )
        rmse = float(np.sqrt(np.mean(residual**2)))
        coefficients = np.asarray(result.getCoefficients(), dtype=float).reshape(-1)
        selected_indices = list(result.getIndices())
        coefficient_rows = sorted(
            [[int(selected_indices[i]), float(coefficients[i])] for i in range(len(coefficients))],
            key=lambda row: abs(float(row[1])),
            reverse=True,
        )[:100]
        return AnalysisPayload(
            metrics={
                "degree": config.degree,
                "training_size": config.training_size,
                "validation_size": config.validation_size,
                "basis_size": int(basis_size),
                "retained_terms": len(coefficients),
                "validation_q2": q2,
                "validation_rmse": rmse,
            },
            tables={
                "coefficients": TableData(
                    columns=["Basis Index", "Coefficient"],
                    rows=coefficient_rows,
                    row_count=len(coefficients),
                    truncated=len(coefficients) > len(coefficient_rows),
                )
            },
            series={
                "validation": SeriesData(
                    name="PCE validation",
                    x=[float(value) for value in observed],
                    y=[float(value) for value in predicted],
                    x_label="Observed",
                    y_label="Predicted",
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "sparse_selection": config.sparse,
            },
        ), config.training_size + config.validation_size


def _build_basis(distribution: ot.Distribution) -> ot.OrthogonalProductPolynomialFactory:
    collection = ot.PolynomialFamilyCollection(distribution.getDimension())
    for index in range(distribution.getDimension()):
        collection[index] = ot.StandardDistributionPolynomialFactory(
            distribution.getMarginal(index)
        )
    return ot.OrthogonalProductPolynomialFactory(collection)


plugin = PcePlugin()
