"""Vectorized multi-output Monte Carlo analysis."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, SeriesData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class MonteCarloConfig(StrictModel):
    sample_size: int = Field(default=1000, ge=10, le=1_000_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list)
    inline_sample_limit: int = Field(default=2000, ge=0, le=20_000)


class MonteCarloPlugin(AnalysisPlugin[MonteCarloConfig]):
    key = "monte_carlo"
    version = "1.0.0"
    name = "Monte Carlo"
    category = "Propagation"
    description = "Propagate the input distribution through the model and summarize outputs."
    config_model = MonteCarloConfig

    def run(self, runtime: ModelRuntime, config: MonteCarloConfig) -> tuple[AnalysisPayload, int]:
        targets = config.output_targets or list(range(runtime.metadata.output_dimension))
        if any(target >= runtime.metadata.output_dimension for target in targets):
            raise IncompatibleAnalysisError("A requested output target does not exist.")
        inputs, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        names = [runtime.metadata.outputs[target].name for target in targets]
        rows: list[list[float | str]] = []
        metrics: dict[str, float | int | str | bool | None] = {"sample_size": config.sample_size}
        series: dict[str, SeriesData] = {}
        facts: dict[str, float | int | str | bool | None] = {}
        for name, target in zip(names, targets, strict=True):
            values = outputs[:, target]
            mean = float(np.mean(values))
            std = float(np.std(values))
            q025, q25, q50, q75, q975 = [
                float(v) for v in np.quantile(values, [0.025, 0.25, 0.5, 0.75, 0.975])
            ]
            rows.append(
                [
                    name,
                    mean,
                    std,
                    float(np.min(values)),
                    q025,
                    q25,
                    q50,
                    q75,
                    q975,
                    float(np.max(values)),
                ]
            )
            metrics[f"{name}.mean"] = mean
            metrics[f"{name}.std"] = std
            facts[f"{name}.mean"] = mean
            facts[f"{name}.95_interval"] = f"[{q025:.6g}, {q975:.6g}]"
            visible = values[: config.inline_sample_limit]
            series[f"output.{target}"] = SeriesData(
                name=name,
                x=list(range(len(visible))),
                y=[float(value) for value in visible],
                x_label="Sample",
                y_label=name,
            )

        inline_rows = min(config.sample_size, config.inline_sample_limit)
        sample_columns = [item.name for item in runtime.metadata.inputs] + names
        sample_rows = np.column_stack([inputs[:inline_rows], outputs[:inline_rows, targets]])
        tables = {
            "summary": TableData(
                columns=[
                    "Output",
                    "Mean",
                    "Std",
                    "Min",
                    "2.5%",
                    "25%",
                    "Median",
                    "75%",
                    "97.5%",
                    "Max",
                ],
                rows=rows,
                row_count=len(rows),
            ),
            "samples": TableData(
                columns=sample_columns,
                rows=[[float(value) for value in row] for row in sample_rows],
                row_count=config.sample_size,
                truncated=inline_rows < config.sample_size,
            ),
        }
        return AnalysisPayload(
            metrics=metrics, tables=tables, series=series, facts=facts
        ), config.sample_size


plugin = MonteCarloPlugin()
