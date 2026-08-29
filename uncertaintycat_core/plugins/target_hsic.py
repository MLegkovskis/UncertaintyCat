"""Target-domain sensitivity using OpenTURNS' stable HSIC estimator."""

from __future__ import annotations

import math
from typing import Literal

import openturns as ot
from pydantic import Field, FiniteFloat

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.progress import report_progress

MAXIMUM_INPUT_DIMENSION = 20
MAXIMUM_HSIC_WORK_UNITS = 150_000_000
MINIMUM_TARGET_OBSERVATIONS = 5


def estimate_hsic_work_units(sample_size: int, input_dimension: int, permutations: int) -> int:
    """Conservatively bound the quadratic kernel operations used by OpenTURNS.

    OpenTURNS evaluates every input HSIC statistic for every output permutation.
    The four additional passes cover covariance construction, the observed and
    normalized indices, and asymptotic diagnostics.  This is a resource bound,
    not a wall-clock prediction.
    """

    return sample_size**2 * (input_dimension + 1) * (permutations + 4)


class TargetHsicConfig(StrictModel):
    sample_size: int = Field(default=250, ge=50, le=500)
    permutations: int = Field(default=100, ge=0, le=200)
    threshold: FiniteFloat = 0.0
    operator: Literal["<=", ">="] = ">="
    smoothing_scale_fraction: FiniteFloat = Field(default=0.1, ge=0.01, le=1.0)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class TargetHsicPlugin(AnalysisPlugin[TargetHsicConfig]):
    key = "target_hsic"
    version = "1.1.0"
    name = "Target-Domain HSIC Sensitivity"
    category = "Sensitivity"
    description = (
        "Rank inputs associated with a scalar output entering a user-defined critical region."
    )
    assumptions = (
        "Target R2-HSIC is a normalized empirical kernel association with a smoothed "
        "critical-domain score; it is not a variance contribution or causal effect.",
        "The critical-domain score is exp(-distance / s), where s is the configured "
        "fraction of the sampled output standard deviation.",
        "Gaussian kernels use empirical input and raw-output standard deviations, so "
        "the indices depend on the sampled distribution, bandwidths, and sample size.",
        "Permutation p-values are finite Monte Carlo independence diagnostics under the "
        "sampled joint distribution; asymptotic p-values are approximations.",
        "Finite-sample unbiased U-statistic estimates can be slightly negative; values are "
        "reported as returned by OpenTURNS rather than clipped.",
        "This analysis screens association with a target domain. It does not estimate the "
        "domain probability or establish predictive validity beyond the sampled distribution.",
    )
    supports_dependent_inputs = True
    supports_multi_output = False
    resource_class = "standard"
    config_model = TargetHsicConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: TargetHsicConfig) -> list[str]:
        dimension = runtime.metadata.input_dimension
        if dimension > MAXIMUM_INPUT_DIMENSION:
            raise IncompatibleAnalysisError(
                f"Target-domain HSIC is capped at {MAXIMUM_INPUT_DIMENSION} inputs because "
                "kernel and permutation work grows quadratically with sample size."
            )
        if not runtime.problem.isContinuous():
            raise IncompatibleAnalysisError(
                "Target-domain HSIC currently requires continuous input marginals because "
                "it uses Gaussian input kernels."
            )
        target = config.output_targets[0] if config.output_targets else 0
        if target < 0 or target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        work_units = estimate_hsic_work_units(config.sample_size, dimension, config.permutations)
        if work_units > MAXIMUM_HSIC_WORK_UNITS:
            raise IncompatibleAnalysisError(
                "The requested target-domain HSIC workload exceeds the bounded quadratic "
                f"kernel-operation budget ({work_units:,} > {MAXIMUM_HSIC_WORK_UNITS:,}). "
                "Reduce the sample size or permutations."
            )
        warnings = [
            "Target-domain HSIC is an association screen, not a failure-probability "
            "estimate, variance allocation, or causal analysis."
        ]
        if runtime.metadata.dependent_inputs:
            warnings.append(
                "Inputs are dependent, so target-HSIC associations can reflect both the "
                "model response and the declared input dependence structure."
            )
        return warnings

    def run(self, runtime: ModelRuntime, config: TargetHsicConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        calls_before = runtime.model.getEvaluationCallsNumber()
        report_progress("sampling", 22, f"Evaluating {config.sample_size} model samples.")
        inputs, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        model_evaluations = int(runtime.model.getEvaluationCallsNumber() - calls_before)
        if model_evaluations < 0:
            raise IncompatibleAnalysisError(
                "The OpenTURNS model-evaluation counter moved backwards unexpectedly."
            )
        input_sample = ot.Sample(inputs.tolist())
        output_values = [float(value) for value in outputs[:, target]]
        output_sample = ot.Sample([[value] for value in output_values])

        output_standard_deviation = float(output_sample.computeStandardDeviation()[0])
        if (
            not math.isfinite(output_standard_deviation)
            or output_standard_deviation <= ot.SpecFunc.ScalarEpsilon
        ):
            raise IncompatibleAnalysisError(
                "Target-domain HSIC is undefined because the selected output is constant."
            )

        in_target = [
            value <= config.threshold if config.operator == "<=" else value >= config.threshold
            for value in output_values
        ]
        target_count = sum(in_target)
        outside_count = config.sample_size - target_count
        if target_count < MINIMUM_TARGET_OBSERVATIONS:
            raise IncompatibleAnalysisError(
                "Target-domain HSIC requires at least "
                f"{MINIMUM_TARGET_OBSERVATIONS} sampled observations in the critical "
                f"domain; this run found {target_count}. Adjust the threshold or sample size."
            )
        if outside_count < MINIMUM_TARGET_OBSERVATIONS:
            raise IncompatibleAnalysisError(
                "Target-domain HSIC requires at least "
                f"{MINIMUM_TARGET_OBSERVATIONS} sampled observations outside the critical "
                f"domain; this run found {outside_count}. Adjust the threshold or sample size."
            )

        report_progress(
            "target_domain",
            34,
            "Checking sampled coverage on both sides of the critical-domain threshold.",
        )
        kernels = ot.CovarianceModelCollection()
        for index in range(runtime.metadata.input_dimension):
            standard_deviation = float(
                input_sample.getMarginal(index).computeStandardDeviation()[0]
            )
            if (
                not math.isfinite(standard_deviation)
                or standard_deviation <= ot.SpecFunc.ScalarEpsilon
            ):
                raise IncompatibleAnalysisError(
                    "Target-domain HSIC is undefined because sampled input "
                    f"'{runtime.metadata.inputs[index].name}' is constant."
                )
            kernel = ot.SquaredExponential(1)
            kernel.setScale([standard_deviation])
            kernels.add(kernel)

        output_kernel = ot.SquaredExponential(1)
        output_kernel.setScale([output_standard_deviation])
        kernels.add(output_kernel)

        smoothing_scale = config.smoothing_scale_fraction * output_standard_deviation
        if config.operator == ">=":
            critical_domain = ot.Interval([float(config.threshold)], [float(ot.SpecFunc.Infinity)])
            target_definition = (
                f"{runtime.metadata.outputs[target].name} >= {float(config.threshold):.12g}"
            )
        else:
            critical_domain = ot.Interval([float(-ot.SpecFunc.Infinity)], [float(config.threshold)])
            target_definition = (
                f"{runtime.metadata.outputs[target].name} <= {float(config.threshold):.12g}"
            )
        distance = ot.DistanceToDomainFunction(critical_domain)
        exponential_filter = ot.SymbolicFunction(["distance", "scale"], ["exp(-distance/scale)"])
        filter_function = ot.ComposedFunction(
            ot.ParametricFunction(exponential_filter, [1], [smoothing_scale]), distance
        )

        try:
            report_progress(
                "kernel_construction",
                46,
                "Constructing Gaussian kernels and the smoothed critical-domain filter.",
            )
            estimator = ot.HSICEstimatorTargetSensitivity(
                kernels,
                input_sample,
                output_sample,
                ot.HSICUStat(),
                filter_function,
            )
            estimator.setPermutationSize(config.permutations)
            report_progress(
                "observed_indices",
                56,
                "OpenTURNS is computing observed target-domain HSIC indices.",
                indeterminate=True,
            )
            hsic_indices = [float(value) for value in estimator.getHSICIndices()]
            normalized_indices = [float(value) for value in estimator.getR2HSICIndices()]
            asymptotic_p_values = [float(value) for value in estimator.getPValuesAsymptotic()]
            if config.permutations:
                report_progress(
                    "permutation_inference",
                    66,
                    f"OpenTURNS is evaluating {config.permutations} permutation replicates.",
                    indeterminate=True,
                )
                permutation_p_values: list[float | None] = [
                    float(value) for value in estimator.getPValuesPermutation()
                ]
            else:
                permutation_p_values = [None] * runtime.metadata.input_dimension
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"OpenTURNS could not compute target-domain HSIC for this sample: {exc}"
            ) from exc

        finite_values = (
            hsic_indices
            + normalized_indices
            + asymptotic_p_values
            + [value for value in permutation_p_values if value is not None]
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise IncompatibleAnalysisError(
                "OpenTURNS produced a non-finite target-domain HSIC diagnostic."
            )

        report_progress("ranking", 90, "Ranking critical-domain association evidence.")
        names = [item.name for item in runtime.metadata.inputs]
        order = sorted(range(len(names)), key=lambda index: normalized_indices[index], reverse=True)
        rank_by_index = {input_index: rank for rank, input_index in enumerate(order, start=1)}
        rows: list[list[str | int | float | None]] = [
            [
                names[index],
                normalized_indices[index],
                hsic_indices[index],
                permutation_p_values[index],
                asymptotic_p_values[index],
                rank_by_index[index],
            ]
            for index in order
        ]
        top = order[0]
        work_units = estimate_hsic_work_units(
            config.sample_size,
            runtime.metadata.input_dimension,
            config.permutations,
        )
        return (
            AnalysisPayload(
                metrics={
                    "sample_size": config.sample_size,
                    "permutations": config.permutations,
                    "target_observations": target_count,
                    "outside_target_observations": outside_count,
                    "target_fraction": target_count / config.sample_size,
                    "smoothing_scale": smoothing_scale,
                    "smoothing_scale_fraction": float(config.smoothing_scale_fraction),
                    "estimated_quadratic_work_units": work_units,
                    "model_evaluations": model_evaluations,
                },
                tables={
                    "target_indices": TableData(
                        columns=[
                            "Input",
                            "Target R2-HSIC",
                            "Target HSIC",
                            "Permutation p-value",
                            "Asymptotic p-value",
                            "Target-association rank",
                        ],
                        rows=rows,
                        row_count=len(rows),
                    )
                },
                facts={
                    "output": runtime.metadata.outputs[target].name,
                    "critical_domain": target_definition,
                    "filter": "exp(-distance_to_critical_domain / smoothing_scale)",
                    "estimator": "OpenTURNS unbiased HSIC U-statistic",
                    "kernel_bandwidths": (
                        "Empirical standard deviations of sampled inputs and raw output"
                    ),
                    "quadratic_work_unit_definition": (
                        "sample_size^2 * (input_dimension + 1) * (permutations + 4); "
                        "a conservative kernel-operation resource bound, not elapsed time"
                    ),
                    "strongest_target_association_input": names[top],
                    "largest_target_r2_hsic": normalized_indices[top],
                    "interpretation_boundary": (
                        "Association with the smoothed critical-domain score; not event "
                        "probability, variance contribution, causal influence, or "
                        "out-of-domain predictive evidence"
                    ),
                    "report_payload_limit": (
                        f"At most {MAXIMUM_INPUT_DIMENSION} input rows; no raw sample is stored"
                    ),
                },
            ),
            model_evaluations,
        )


plugin = TargetHsicPlugin()
