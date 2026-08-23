"""Kernel dependence sensitivity using normalized empirical HSIC."""

from __future__ import annotations

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class HsicConfig(StrictModel):
    sample_size: int = Field(default=250, ge=30, le=2_000)
    permutations: int = Field(default=100, ge=0, le=2_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class HsicPlugin(AnalysisPlugin[HsicConfig]):
    key = "hsic"
    version = "2.0.0"
    name = "HSIC Dependence Analysis"
    category = "Sensitivity"
    description = (
        "Detect nonlinear and non-monotonic input-output dependence with kernel statistics."
    )
    assumptions = ("The empirical kernel statistic depends on bandwidth and sample size.",)
    supports_multi_output = False
    resource_class = "standard"
    config_model = HsicConfig

    def run(self, runtime: ModelRuntime, config: HsicConfig) -> tuple[AnalysisPayload, int]:
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        inputs, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        input_sample = ot.Sample(inputs.tolist())
        output_sample = ot.Sample([[float(value)] for value in outputs[:, target]])
        kernels: list[ot.CovarianceModel] = []
        for index in range(runtime.metadata.input_dimension):
            marginal_sample = input_sample.getMarginal(index)
            kernel = ot.SquaredExponential(1)
            kernel.setScale(marginal_sample.computeStandardDeviation())
            kernels.append(kernel)
        output_kernel = ot.SquaredExponential(1)
        output_kernel.setScale(output_sample.computeStandardDeviation())
        kernels.append(output_kernel)
        estimator = ot.HSICEstimatorGlobalSensitivity(
            kernels, input_sample, output_sample, ot.HSICUStat()
        )
        estimator.setPermutationSize(config.permutations)
        scores = [float(value) for value in estimator.getR2HSICIndices()]
        p_values = (
            [float(value) for value in estimator.getPValuesPermutation()]
            if config.permutations
            else [None] * runtime.metadata.input_dimension
        )
        names = [item.name for item in runtime.metadata.inputs]
        rows: list[list[str | float | None]] = [
            [name, scores[index], p_values[index]] for index, name in enumerate(names)
        ]
        top = max(range(len(scores)), key=scores.__getitem__)
        return AnalysisPayload(
            metrics={"sample_size": config.sample_size, "permutations": config.permutations},
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
