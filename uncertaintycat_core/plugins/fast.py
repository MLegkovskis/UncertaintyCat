"""Fourier amplitude sensitivity test."""

from __future__ import annotations

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.plugins.sobol import SobolPlugin


class FastConfig(StrictModel):
    sample_size: int = Field(default=400, ge=65, le=131_072)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class FastPlugin(AnalysisPlugin[FastConfig]):
    key = "fast"
    version = "2.0.0"
    name = "FAST Sensitivity Analysis"
    category = "Sensitivity"
    description = "Estimate first- and total-order variance effects using Fourier amplitudes."
    assumptions = (
        "Inputs must be independent.",
        "The selected output must have non-zero variance.",
    )
    supports_dependent_inputs = False
    supports_multi_output = False
    resource_class = "heavy"
    config_model = FastConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: FastConfig) -> list[str]:
        if not SobolPlugin._is_independent(runtime.problem):
            raise IncompatibleAnalysisError("FAST requires an independent input distribution.")
        return []

    def run(self, runtime: ModelRuntime, config: FastConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        ot.RandomGenerator.SetSeed(config.seed)
        selected_model = runtime.model.getMarginal(target)
        algorithm = ot.FAST(selected_model, runtime.problem, config.sample_size)
        first = algorithm.getFirstOrderIndices()
        total = algorithm.getTotalOrderIndices()
        names = [item.name for item in runtime.metadata.inputs]
        rows = [
            [name, float(first[i]), float(total[i]), float(total[i] - first[i])]
            for i, name in enumerate(names)
        ]
        top = max(range(len(names)), key=lambda index: float(total[index]))
        evaluations = config.sample_size * runtime.metadata.input_dimension
        return AnalysisPayload(
            metrics={"sample_size": config.sample_size, "model_evaluations_estimate": evaluations},
            tables={
                "indices": TableData(
                    columns=["Variable", "First Order", "Total Order", "Interaction"],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "most_influential_input": names[top],
                "largest_total_order_index": float(total[top]),
            },
        ), evaluations


plugin = FastPlugin()
