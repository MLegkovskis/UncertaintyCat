"""Surrogate-validated ANCOVA sensitivity analysis for dependent inputs."""

from __future__ import annotations

import math

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, SeriesData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.plugins.pce import build_pce_basis

MINIMUM_VALIDATION_Q2 = 0.8
MAXIMUM_BASIS_SIZE = 500
SEED_MODULUS = 2_147_483_648


class AncovaConfig(StrictModel):
    degree: int = Field(default=3, ge=1, le=6)
    training_size: int = Field(default=1000, ge=64, le=10_000)
    validation_size: int = Field(default=500, ge=64, le=2_000)
    ancova_sample_size: int = Field(default=2000, ge=128, le=20_000)
    sparse: bool = True
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class AncovaPlugin(AnalysisPlugin[AncovaConfig]):
    key = "ancova"
    version = "1.0.0"
    name = "ANCOVA Dependent-Input Sensitivity"
    category = "Sensitivity"
    description = "Separate physical and correlation-driven first-order variance contributions."
    assumptions = (
        "ANCOVA uses a polynomial-chaos decomposition fitted on an independent product "
        "distribution with the model's declared marginals.",
        "The polynomial-chaos metamodel must reach Q2 of at least 0.8 on a hold-out from "
        "the actual dependent input distribution.",
        "Reported physical and correlation contributions are first-order variance "
        "contributions, not total-order indices or causal effects.",
        "Correlation contributions may be negative and are not clipped.",
    )
    supports_dependent_inputs = True
    requires_dependent_inputs = True
    supports_multi_output = False
    resource_class = "heavy"
    config_model = AncovaConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: AncovaConfig) -> list[str]:
        dimension = runtime.metadata.input_dimension
        if dimension < 2:
            raise IncompatibleAnalysisError("ANCOVA requires at least two input variables.")
        if dimension > 10:
            raise IncompatibleAnalysisError(
                "ANCOVA is capped at ten inputs because its polynomial basis grows "
                "combinatorially. Screen dimensions first."
            )
        if not runtime.problem.isContinuous():
            raise IncompatibleAnalysisError("ANCOVA currently requires continuous input marginals.")
        if runtime.problem.hasIndependentCopula():
            raise IncompatibleAnalysisError(
                "ANCOVA is reserved for dependent inputs; use Sobol or polynomial chaos "
                "for an independent input distribution."
            )
        return []

    def run(self, runtime: ModelRuntime, config: AncovaConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target < 0 or target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")

        independent = ot.JointDistribution(
            [runtime.problem.getMarginal(index) for index in range(runtime.problem.getDimension())]
        )
        independent.setDescription(runtime.problem.getDescription())
        basis = build_pce_basis(independent)
        enumeration = basis.getEnumerateFunction()
        basis_size = int(enumeration.getBasisSizeFromTotalDegree(config.degree))
        if basis_size > MAXIMUM_BASIS_SIZE:
            raise IncompatibleAnalysisError(
                f"The requested ANCOVA polynomial basis has {basis_size} terms; "
                f"the maximum is {MAXIMUM_BASIS_SIZE}. Reduce the degree or dimensions."
            )
        if config.training_size <= basis_size:
            raise IncompatibleAnalysisError(
                "ANCOVA training size must be greater than the polynomial basis size "
                f"({basis_size})."
            )

        try:
            ot.RandomGenerator.SetSeed(config.seed)
            training_x = independent.getSample(config.training_size)
            training_y = runtime.model(training_x).getMarginal(target)
            projection = (
                ot.LeastSquaresStrategy(
                    training_x,
                    training_y,
                    ot.LeastSquaresMetaModelSelectionFactory(),
                )
                if config.sparse
                else ot.LeastSquaresStrategy(training_x, training_y)
            )
            algorithm = ot.FunctionalChaosAlgorithm(
                training_x,
                training_y,
                independent,
                ot.FixedStrategy(basis, basis_size),
                projection,
            )
            algorithm.run()
            pce_result = algorithm.getResult()

            ot.RandomGenerator.SetSeed((config.seed + 1) % SEED_MODULUS)
            validation_x = runtime.problem.getSample(config.validation_size)
            observed = runtime.model(validation_x).getMarginal(target)
            observed_variance = float(observed.computeVariance()[0])
            if (
                not math.isfinite(observed_variance)
                or observed_variance <= ot.SpecFunc.ScalarEpsilon
            ):
                raise IncompatibleAnalysisError(
                    "ANCOVA is undefined because the dependent hold-out output is constant."
                )
            predicted = pce_result.getMetaModel()(validation_x)
            validation = ot.MetaModelValidation(observed, predicted)
            q2 = float(validation.computeR2Score()[0])
            rmse = float(validation.computeMeanSquaredError()[0]) ** 0.5
            normalized_rmse = rmse / observed_variance**0.5
            if not all(math.isfinite(value) for value in (q2, rmse, normalized_rmse)):
                raise IncompatibleAnalysisError(
                    "ANCOVA validation produced a non-finite metamodel diagnostic."
                )
            if q2 < MINIMUM_VALIDATION_Q2:
                raise IncompatibleAnalysisError(
                    "ANCOVA stopped because dependent-input validation Q2 "
                    f"({q2:.4f}) is below the required {MINIMUM_VALIDATION_Q2:.1f}. "
                    "Increase the training design or degree before interpreting indices."
                )

            ot.RandomGenerator.SetSeed((config.seed + 2) % SEED_MODULUS)
            dependent_sample = runtime.problem.getSample(config.ancova_sample_size)
            ancova = ot.ANCOVA(pce_result, dependent_sample)
            contributions = [float(value) for value in ancova.getIndices(0)]
            physical = [float(value) for value in ancova.getUncorrelatedIndices(0)]
        except IncompatibleAnalysisError:
            raise
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"ANCOVA construction failed for this model and distribution: {exc}"
            ) from exc

        correlation = [contributions[index] - physical[index] for index in range(len(physical))]
        if not all(
            math.isfinite(value)
            for values in (contributions, physical, correlation)
            for value in values
        ):
            raise IncompatibleAnalysisError("ANCOVA produced non-finite sensitivity indices.")

        names = [item.name for item in runtime.metadata.inputs]
        order = sorted(range(len(names)), key=lambda index: abs(contributions[index]), reverse=True)
        rank_by_index = {input_index: rank for rank, input_index in enumerate(order, start=1)}
        rows: list[list[str | int | float]] = [
            [
                names[index],
                contributions[index],
                physical[index],
                correlation[index],
                rank_by_index[index],
            ]
            for index in order
        ]
        coefficients = pce_result.getCoefficients()
        top_index = order[0]
        return (
            AnalysisPayload(
                metrics={
                    "degree": config.degree,
                    "training_size": config.training_size,
                    "validation_size": config.validation_size,
                    "ancova_sample_size": config.ancova_sample_size,
                    "basis_size": basis_size,
                    "retained_terms": coefficients.getSize(),
                    "validation_q2": q2,
                    "validation_rmse": rmse,
                    "validation_normalized_rmse": normalized_rmse,
                    "sum_ancova_contributions": sum(contributions),
                    "sum_physical_contributions": sum(physical),
                    "sum_correlation_contributions": sum(correlation),
                },
                tables={
                    "indices": TableData(
                        columns=[
                            "Input",
                            "ANCOVA Contribution",
                            "Physical Contribution",
                            "Correlation Contribution",
                            "Absolute Contribution Rank",
                        ],
                        rows=rows,
                        row_count=len(rows),
                    )
                },
                series={
                    "validation": SeriesData(
                        name="Dependent-input PCE validation",
                        x=[float(row[0]) for row in observed],
                        y=[float(row[0]) for row in predicted],
                        x_label="Observed",
                        y_label="Predicted",
                    )
                },
                facts={
                    "output": runtime.metadata.outputs[target].name,
                    "copula": runtime.metadata.copula,
                    "most_influential_input_by_absolute_contribution": names[top_index],
                    "largest_absolute_ancova_contribution": abs(contributions[top_index]),
                    "sparse_selection": config.sparse,
                    "reference_distribution": "Independent product of declared marginals",
                    "validation_distribution": "Declared dependent input distribution",
                },
            ),
            config.training_size + config.validation_size,
        )


plugin = AncovaPlugin()
