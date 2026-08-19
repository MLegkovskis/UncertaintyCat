"""Distribution-aware Morris elementary-effects screening."""

from __future__ import annotations

import numpy as np
import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.plugins.sobol import SobolPlugin


class MorrisConfig(StrictModel):
    trajectories: int = Field(default=10, ge=4, le=1_000)
    levels: int = Field(default=6, ge=4, le=50)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class MorrisPlugin(AnalysisPlugin[MorrisConfig]):
    key = "morris"
    version = "1.0.0"
    name = "Morris Screening"
    category = "Sensitivity"
    description = "Screen many inputs using signed and absolute elementary effects."
    assumptions = (
        "Inputs must be independent.",
        "Effects are computed along randomized trajectories in probability space.",
    )
    supports_dependent_inputs = False
    supports_multi_output = False
    config_model = MorrisConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: MorrisConfig) -> list[str]:
        if not SobolPlugin._is_independent(runtime.problem):
            raise IncompatibleAnalysisError(
                "Morris probability-space trajectories require independent inputs."
            )
        return []

    def run(self, runtime: ModelRuntime, config: MorrisConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        dimension = runtime.metadata.input_dimension
        rng = np.random.default_rng(config.seed)
        delta = config.levels / (2.0 * (config.levels - 1))
        base_grid = np.arange(config.levels // 2, dtype=float) / (config.levels - 1)
        effects: list[list[float]] = [[] for _ in range(dimension)]
        evaluations = 0
        for _ in range(config.trajectories):
            unit = rng.choice(base_grid, size=dimension).astype(float)
            current = _from_unit(runtime, unit)
            current_output = float(runtime.model(ot.Point(current))[target])
            evaluations += 1
            for index in rng.permutation(dimension):
                next_unit = unit.copy()
                next_unit[index] += delta
                next_point = _from_unit(runtime, next_unit)
                next_output = float(runtime.model(ot.Point(next_point))[target])
                effects[index].append((next_output - current_output) / delta)
                unit, current_output = next_unit, next_output
                evaluations += 1
        names = [item.name for item in runtime.metadata.inputs]
        rows: list[list[str | float]] = []
        absolute_means: list[float] = []
        for name, values in zip(names, effects, strict=True):
            array = np.asarray(values)
            mean_absolute = float(np.mean(np.abs(array)))
            absolute_means.append(mean_absolute)
            rows.append([name, float(np.mean(array)), mean_absolute, float(np.std(array, ddof=1))])
        top = int(np.argmax(absolute_means))
        return AnalysisPayload(
            metrics={
                "trajectories": config.trajectories,
                "levels": config.levels,
                "model_evaluations": evaluations,
            },
            tables={
                "effects": TableData(
                    columns=["Variable", "Mean Effect", "Mean Absolute Effect", "Effect Std"],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "most_influential_input": names[top],
                "largest_mean_absolute_effect": absolute_means[top],
            },
        ), evaluations


def _from_unit(runtime: ModelRuntime, point: np.ndarray) -> np.ndarray:
    clipped = np.clip(point, 1e-10, 1 - 1e-10)
    return np.asarray(
        [
            runtime.problem.getMarginal(index).computeQuantile(float(value))[0]
            for index, value in enumerate(clipped)
        ],
        dtype=float,
    )


plugin = MorrisPlugin()
