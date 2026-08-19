"""Correlation and standardized-regression sensitivity measures."""

from __future__ import annotations

import numpy as np
from pydantic import Field
from scipy import stats

from uncertaintycat_core.contracts import AnalysisPayload, MatrixData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class CorrelationConfig(StrictModel):
    sample_size: int = Field(default=1000, ge=30, le=200_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list)


class CorrelationPlugin(AnalysisPlugin[CorrelationConfig]):
    key = "correlation"
    version = "1.0.0"
    name = "Correlation Analysis"
    category = "Sensitivity"
    description = "Compare Pearson, Spearman, partial, and standardized regression effects."
    assumptions = (
        "Pearson and standardized regression coefficients describe linear effects.",
        "Rank coefficients describe monotonic effects and do not prove causality.",
    )
    config_model = CorrelationConfig

    def run(self, runtime: ModelRuntime, config: CorrelationConfig) -> tuple[AnalysisPayload, int]:
        targets = config.output_targets or list(range(runtime.metadata.output_dimension))
        if any(target >= runtime.metadata.output_dimension for target in targets):
            raise IncompatibleAnalysisError("A requested output target does not exist.")
        inputs, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        names = [item.name for item in runtime.metadata.inputs]
        output_names = [runtime.metadata.outputs[target].name for target in targets]
        matrices: dict[str, MatrixData] = {}
        measures: dict[str, list[list[float | None]]] = {
            "pearson": [],
            "spearman": [],
            "partial": [],
            "src": [],
            "srrc": [],
        }
        rows: list[list[str | float | None]] = []
        standardized_x = stats.zscore(inputs, axis=0)
        ranked_x = np.column_stack([stats.rankdata(inputs[:, i]) for i in range(inputs.shape[1])])
        ranked_x = stats.zscore(ranked_x, axis=0)
        for target, output_name in zip(targets, output_names, strict=True):
            y = outputs[:, target]
            standardized_y = stats.zscore(y)
            ranked_y = stats.zscore(stats.rankdata(y))
            src = np.linalg.lstsq(standardized_x, standardized_y, rcond=None)[0]
            srrc = np.linalg.lstsq(ranked_x, ranked_y, rcond=None)[0]
            combined = np.column_stack([inputs, y])
            correlation = np.corrcoef(combined, rowvar=False)
            try:
                precision = np.linalg.pinv(correlation)
                partial = -precision[:-1, -1] / np.sqrt(
                    np.maximum(
                        precision[:-1, :-1].diagonal() * precision[-1, -1], np.finfo(float).eps
                    )
                )
            except np.linalg.LinAlgError:
                partial = np.full(inputs.shape[1], np.nan)
            for index, input_name in enumerate(names):
                pearson = _safe_stat(stats.pearsonr(inputs[:, index], y).statistic)
                spearman = _safe_stat(stats.spearmanr(inputs[:, index], y).statistic)
                coefficient_values = [
                    pearson,
                    spearman,
                    _safe_stat(partial[index]),
                    _safe_stat(src[index]),
                    _safe_stat(srrc[index]),
                ]
                rows.append([output_name, input_name, *coefficient_values])
                for key, value in zip(measures, coefficient_values, strict=True):
                    if len(measures[key]) <= output_names.index(output_name):
                        measures[key].append([])
                    measures[key][-1].append(value)
        for key, matrix_values in measures.items():
            matrices[key] = MatrixData(
                row_labels=output_names, column_labels=names, values=matrix_values
            )
        return AnalysisPayload(
            metrics={"sample_size": config.sample_size, "output_count": len(targets)},
            tables={
                "coefficients": TableData(
                    columns=["Output", "Input", "Pearson", "Spearman", "Partial", "SRC", "SRRC"],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            matrices=matrices,
        ), config.sample_size


def _safe_stat(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


plugin = CorrelationPlugin()
