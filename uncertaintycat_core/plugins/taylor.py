"""Local derivative-based Taylor sensitivity analysis."""

from __future__ import annotations

import numpy as np
import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class TaylorConfig(StrictModel):
    relative_step: float = Field(default=1e-5, gt=0, le=0.1)
    validation_size: int = Field(default=500, ge=20, le=20_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class TaylorPlugin(AnalysisPlugin[TaylorConfig]):
    key = "taylor"
    version = "1.0.0"
    name = "Taylor Sensitivity Analysis"
    category = "Sensitivity"
    description = (
        "Rank local variance contributions and validate the first-order surrogate globally."
    )
    assumptions = (
        "The response is differentiable near the input mean.",
        "Local importance may not represent a strongly nonlinear global response.",
    )
    supports_multi_output = False
    config_model = TaylorConfig

    def run(self, runtime: ModelRuntime, config: TaylorConfig) -> tuple[AnalysisPayload, int]:
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        mean = np.asarray(runtime.problem.getMean(), dtype=float)
        standard_deviation = np.asarray(runtime.problem.getStandardDeviation(), dtype=float)
        nominal = float(runtime.model(ot.Point(mean))[target])
        gradients = np.zeros(runtime.metadata.input_dimension)
        for index, scale in enumerate(standard_deviation):
            step = config.relative_step * max(abs(float(scale)), 1.0)
            lower, upper = mean.copy(), mean.copy()
            lower[index] -= step
            upper[index] += step
            gradients[index] = (
                float(runtime.model(ot.Point(upper))[target])
                - float(runtime.model(ot.Point(lower))[target])
            ) / (2 * step)
        contributions = gradients**2 * standard_deviation**2
        total = float(contributions.sum())
        indices = (
            contributions / total if total > np.finfo(float).eps else np.zeros_like(contributions)
        )
        x_validation, y_validation = runtime.sample_and_evaluate(
            config.validation_size, config.seed
        )
        predicted = nominal + (x_validation - mean) @ gradients
        observed = y_validation[:, target]
        residual = observed - predicted
        variance = float(np.sum((observed - observed.mean()) ** 2))
        q2 = 1.0 - float(np.sum(residual**2)) / variance if variance > np.finfo(float).eps else 0.0
        rmse = float(np.sqrt(np.mean(residual**2)))
        names = [item.name for item in runtime.metadata.inputs]
        rows = [
            [
                name,
                float(mean[i]),
                float(gradients[i]),
                float(standard_deviation[i] ** 2),
                float(indices[i]),
            ]
            for i, name in enumerate(names)
        ]
        top = int(np.argmax(indices))
        evaluations = 1 + 2 * runtime.metadata.input_dimension + config.validation_size
        return AnalysisPayload(
            metrics={
                "nominal_output": nominal,
                "linear_surrogate_q2": q2,
                "linear_surrogate_rmse": rmse,
                "validation_size": config.validation_size,
            },
            tables={
                "indices": TableData(
                    columns=[
                        "Variable",
                        "Nominal Input",
                        "Gradient",
                        "Input Variance",
                        "Taylor Index",
                    ],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "most_influential_input": names[top],
                "largest_taylor_index": float(indices[top]),
            },
        ), evaluations


plugin = TaylorPlugin()
