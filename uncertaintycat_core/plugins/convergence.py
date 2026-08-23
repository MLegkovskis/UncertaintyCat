"""OpenTURNS-authoritative expectation convergence diagnostics."""

from __future__ import annotations

import math

import openturns as ot
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
    version = "2.0.0"
    name = "Expectation Convergence"
    category = "Diagnostics"
    description = "Track OpenTURNS' expectation estimate and Monte Carlo uncertainty history."
    assumptions = ("Samples are representative draws from the declared input distribution.",)
    supports_multi_output = False
    config_model = ConvergenceConfig

    def run(self, runtime: ModelRuntime, config: ConvergenceConfig) -> tuple[AnalysisPayload, int]:
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        selected_model = runtime.model.getMarginal(target)
        output_vector = ot.CompositeRandomVector(selected_model, ot.RandomVector(runtime.problem))
        algorithm = ot.ExpectationSimulationAlgorithm(output_vector)
        algorithm.setBlockSize(1)
        algorithm.setMaximumOuterSampling(config.sample_size)
        algorithm.setMaximumCoefficientOfVariation(-1.0)
        algorithm.setMaximumStandardDeviation(-1.0)
        algorithm.setConvergenceStrategy(ot.Compact(max(16, config.max_points // 2)))
        ot.RandomGenerator.SetSeed(config.seed)
        algorithm.run()
        result = algorithm.getResult()
        history = algorithm.getConvergenceStrategy().getSample()
        counts = [int(round(row[2])) for row in history]
        means = [float(row[0]) for row in history]
        variances = [max(float(row[1]), 0.0) for row in history]
        half_widths = [1.96 * math.sqrt(value) for value in variances]
        evaluations = int(result.getOuterSampling() * result.getBlockSize())
        return AnalysisPayload(
            metrics={
                "sample_size": evaluations,
                "final_mean": float(result.getExpectationEstimate()[0]),
                "final_standard_error": float(result.getStandardDeviation()[0]),
            },
            series={
                "running_mean": SeriesData(
                    name="Running expectation estimate",
                    x=counts,
                    y=means,
                    x_label="Model evaluations",
                    y_label=runtime.metadata.outputs[target].name,
                ),
                "confidence_lower": SeriesData(
                    name="95% confidence lower",
                    x=counts,
                    y=[mean - width for mean, width in zip(means, half_widths, strict=True)],
                    x_label="Model evaluations",
                    y_label=runtime.metadata.outputs[target].name,
                ),
                "confidence_upper": SeriesData(
                    name="95% confidence upper",
                    x=counts,
                    y=[mean + width for mean, width in zip(means, half_widths, strict=True)],
                    x_label="Model evaluations",
                    y_label=runtime.metadata.outputs[target].name,
                ),
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "stopping_reason": "maximum evaluations reached",
            },
        ), evaluations


plugin = ConvergencePlugin()
