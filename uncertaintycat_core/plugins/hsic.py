"""Kernel dependence sensitivity using normalized empirical HSIC."""

from __future__ import annotations

import numpy as np
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
    version = "1.0.0"
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
        y_kernel = _centered_kernel(outputs[:, target])
        rng = np.random.default_rng(config.seed)
        names = [item.name for item in runtime.metadata.inputs]
        rows: list[list[str | float | None]] = []
        scores: list[float] = []
        for index, name in enumerate(names):
            x_kernel = _centered_kernel(inputs[:, index])
            score = _normalized_hsic(x_kernel, y_kernel)
            p_value: float | None = None
            if config.permutations:
                exceedances = 0
                for _ in range(config.permutations):
                    order = rng.permutation(config.sample_size)
                    if _normalized_hsic(x_kernel, y_kernel[order][:, order]) >= score:
                        exceedances += 1
                p_value = (exceedances + 1) / (config.permutations + 1)
            scores.append(score)
            rows.append([name, score, p_value])
        top = int(np.argmax(scores))
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


def _centered_kernel(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    squared_distance = (vector[:, None] - vector[None, :]) ** 2
    positive = squared_distance[squared_distance > 0]
    bandwidth_squared = float(np.median(positive)) if positive.size else 1.0
    kernel = np.exp(-squared_distance / max(2.0 * bandwidth_squared, np.finfo(float).eps))
    return kernel - kernel.mean(axis=0)[None, :] - kernel.mean(axis=1)[:, None] + kernel.mean()


def _normalized_hsic(x_kernel: np.ndarray, y_kernel: np.ndarray) -> float:
    numerator = float(np.sum(x_kernel * y_kernel))
    denominator = float(np.sqrt(np.sum(x_kernel * x_kernel) * np.sum(y_kernel * y_kernel)))
    return max(0.0, numerator / denominator) if denominator > np.finfo(float).eps else 0.0


plugin = HsicPlugin()
