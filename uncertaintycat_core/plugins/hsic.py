"""Kernel dependence sensitivity using normalized empirical HSIC."""

from __future__ import annotations

import math

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.progress import report_progress

MAXIMUM_HSIC_WORK_UNITS = 150_000_000


def estimate_hsic_work_units(sample_size: int, input_dimension: int, permutations: int) -> int:
    """Conservatively account for observed, normalized and permuted quadratic kernels."""

    return sample_size**2 * (input_dimension + 1) * (permutations + 4)


def maximum_hsic_sample_size(input_dimension: int, permutations: int) -> int:
    """Return the largest schema-valid sample admitted by the compute resource envelope."""

    divisor = (input_dimension + 1) * (permutations + 4)
    return min(2_000, math.isqrt(MAXIMUM_HSIC_WORK_UNITS // divisor))


class HsicConfig(StrictModel):
    sample_size: int = Field(default=250, ge=30, le=2_000)
    permutations: int = Field(default=100, ge=0, le=2_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class HsicPlugin(AnalysisPlugin[HsicConfig]):
    key = "hsic"
    version = "2.1.0"
    name = "HSIC Dependence Analysis"
    category = "Sensitivity"
    description = (
        "Detect nonlinear and non-monotonic input-output dependence with kernel statistics."
    )
    assumptions = ("The empirical kernel statistic depends on bandwidth and sample size.",)
    supports_multi_output = False
    resource_class = "standard"
    config_model = HsicConfig

    def model_incompatibility_reasons(self, runtime: ModelRuntime) -> list[tuple[str, str]]:
        issues = super().model_incompatibility_reasons(runtime)
        if not runtime.problem.isContinuous():
            issues.append(
                (
                    "CONTINUOUS_INPUTS_REQUIRED",
                    "HSIC requires continuous input marginals because this implementation "
                    "uses Gaussian kernels.",
                )
            )
        if not any(output.variable for output in runtime.assessment.profile.pilot_outputs):
            issues.append(
                (
                    "VARIABLE_OUTPUT_REQUIRED",
                    "HSIC requires at least one output that varies in the deterministic "
                    "validation sample.",
                )
            )
        if (
            maximum_hsic_sample_size(runtime.metadata.input_dimension, HsicConfig().permutations)
            < 30
        ):
            issues.append(
                (
                    "HSIC_RESOURCE_LIMIT",
                    "HSIC has no schema-valid sample size within the bounded compute "
                    "envelope for this input dimension.",
                )
            )
        return issues

    def safe_model_config(
        self, runtime: ModelRuntime
    ) -> dict[str, str | int | float | bool | None]:
        permutations = HsicConfig().permutations
        return {
            "maximum_sample_size": maximum_hsic_sample_size(
                runtime.metadata.input_dimension, permutations
            ),
            "permutations": permutations,
        }

    def applicability_warnings(self, runtime: ModelRuntime, config: HsicConfig) -> list[str]:
        if not runtime.problem.isContinuous():
            raise IncompatibleAnalysisError(
                "HSIC requires continuous input marginals because it uses Gaussian kernels."
            )
        target = config.output_targets[0] if config.output_targets else 0
        if target < 0 or target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        work_units = estimate_hsic_work_units(
            config.sample_size, runtime.metadata.input_dimension, config.permutations
        )
        if work_units > MAXIMUM_HSIC_WORK_UNITS:
            maximum = maximum_hsic_sample_size(
                runtime.metadata.input_dimension, config.permutations
            )
            raise IncompatibleAnalysisError(
                "The requested HSIC workload exceeds the bounded quadratic kernel-operation "
                f"budget ({work_units:,} > {MAXIMUM_HSIC_WORK_UNITS:,}). For this "
                f"{runtime.metadata.input_dimension}-input model and {config.permutations} "
                f"permutations, use at most {maximum} samples."
            )
        return (
            [
                "Inputs are dependent, so global HSIC associations can reflect both direct "
                "input-output dependence and dependence transmitted through other inputs."
            ]
            if runtime.metadata.dependent_inputs
            else []
        )

    def run(self, runtime: ModelRuntime, config: HsicConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        report_progress("sampling", 22, f"Evaluating {config.sample_size} model samples.")
        inputs, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        input_sample = ot.Sample(inputs.tolist())
        output_sample = ot.Sample([[float(value)] for value in outputs[:, target]])
        output_standard_deviation = float(output_sample.computeStandardDeviation()[0])
        if (
            not math.isfinite(output_standard_deviation)
            or output_standard_deviation <= ot.SpecFunc.ScalarEpsilon
        ):
            raise IncompatibleAnalysisError(
                "HSIC is undefined because the selected output is constant in the analysis sample."
            )
        report_progress(
            "kernel_construction", 36, "Constructing Gaussian input and output kernels."
        )
        kernels: list[ot.CovarianceModel] = []
        for index in range(runtime.metadata.input_dimension):
            marginal_sample = input_sample.getMarginal(index)
            standard_deviation = float(marginal_sample.computeStandardDeviation()[0])
            if (
                not math.isfinite(standard_deviation)
                or standard_deviation <= ot.SpecFunc.ScalarEpsilon
            ):
                raise IncompatibleAnalysisError(
                    "HSIC is undefined because sampled input "
                    f"'{runtime.metadata.inputs[index].name}' is constant."
                )
            kernel = ot.SquaredExponential(1)
            kernel.setScale([standard_deviation])
            kernels.append(kernel)
        output_kernel = ot.SquaredExponential(1)
        output_kernel.setScale([output_standard_deviation])
        kernels.append(output_kernel)
        estimator = ot.HSICEstimatorGlobalSensitivity(
            kernels, input_sample, output_sample, ot.HSICUStat()
        )
        estimator.setPermutationSize(config.permutations)
        report_progress("observed_indices", 48, "Computing normalized observed HSIC indices.")
        try:
            scores = [float(value) for value in estimator.getR2HSICIndices()]
            if config.permutations:
                report_progress(
                    "permutation_inference",
                    58,
                    f"OpenTURNS is evaluating {config.permutations} permutation replicates.",
                    indeterminate=True,
                )
                p_values: list[float | None] = [
                    float(value) for value in estimator.getPValuesPermutation()
                ]
            else:
                p_values = [None] * runtime.metadata.input_dimension
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"OpenTURNS could not compute HSIC for this sample: {exc}"
            ) from exc
        if not all(math.isfinite(value) for value in scores) or not all(
            value is None or math.isfinite(value) for value in p_values
        ):
            raise IncompatibleAnalysisError("OpenTURNS produced a non-finite HSIC diagnostic.")
        report_progress("ranking", 90, "Ranking the retained input-dependence evidence.")
        names = [item.name for item in runtime.metadata.inputs]
        rows: list[list[str | float | None]] = [
            [name, scores[index], p_values[index]] for index, name in enumerate(names)
        ]
        top = max(range(len(scores)), key=scores.__getitem__)
        return AnalysisPayload(
            metrics={
                "sample_size": config.sample_size,
                "permutations": config.permutations,
                "estimated_quadratic_work_units": estimate_hsic_work_units(
                    config.sample_size,
                    runtime.metadata.input_dimension,
                    config.permutations,
                ),
            },
            tables={
                "indices": TableData(
                    columns=["Variable", "Normalized HSIC", "Permutation p-value"],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "strongest_dependence_input": names[top],
                "largest_normalized_hsic": scores[top],
            },
        ), config.sample_size


plugin = HsicPlugin()
