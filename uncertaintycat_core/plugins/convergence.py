"""Running expectation and uncertainty convergence diagnostics."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, SeriesData, StrictModel
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class ConvergenceConfig(StrictModel):
    sample_size: int = Field(default=5000, ge=30, le=500_000)
    max_points: int = Field(default=300, ge=30, le=2_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class ConvergencePlugin(AnalysisPlugin[ConvergenceConfig]):
    key = "convergence"
    version = "1.0.0"
    name = "Expectation Convergence"
    category = "Diagnostics"
    description = (
        "Track the running mean and its Monte Carlo confidence band as evaluations accumulate."
    )
    assumptions = ("Samples are representative draws from the declared input distribution.",)
    supports_multi_output = False
    config_model = ConvergenceConfig

    def run(self, runtime: ModelRuntime, config: ConvergenceConfig) -> tuple[AnalysisPayload, int]:
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        _, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        values = outputs[:, target]
        counts = np.arange(1, config.sample_size + 1)
        running_mean = np.cumsum(values) / counts
        square_sum = np.cumsum(values**2)
        variance = np.maximum(
            (square_sum - counts * running_mean**2) / np.maximum(counts - 1, 1), 0
        )
        half_width = 1.96 * np.sqrt(variance / counts)
        chosen = (
            np.unique(
                np.linspace(
                    1, config.sample_size, min(config.max_points, config.sample_size), dtype=int
                )
            )
            - 1
        )
        x = [int(counts[index]) for index in chosen]
        mean_y = [float(running_mean[index]) for index in chosen]
        return AnalysisPayload(
            metrics={
                "sample_size": config.sample_size,
                "final_mean": float(running_mean[-1]),
                "final_standard_error": float(half_width[-1] / 1.96),
            },
            series={
                "running_mean": SeriesData(
                    name="Running mean",
                    x=x,
                    y=mean_y,
                    x_label="Model evaluations",
                    y_label=runtime.metadata.outputs[target].name,
                ),
                "confidence_lower": SeriesData(
                    name="95% confidence lower",
                    x=x,
                    y=[float(running_mean[i] - half_width[i]) for i in chosen],
                    x_label="Model evaluations",
                    y_label=runtime.metadata.outputs[target].name,
                ),
                "confidence_upper": SeriesData(
                    name="95% confidence upper",
                    x=x,
                    y=[float(running_mean[i] + half_width[i]) for i in chosen],
                    x_label="Model evaluations",
                    y_label=runtime.metadata.outputs[target].name,
                ),
            },
            facts={"output": runtime.metadata.outputs[target].name},
        ), config.sample_size


plugin = ConvergencePlugin()
