"""Correlation and standardized-regression sensitivity measures."""

from __future__ import annotations

import math

import openturns as ot
from pydantic import Field

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
    version = "2.0.0"
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
        input_sample = ot.Sample(inputs.tolist())
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
        for output_index, (target, output_name) in enumerate(
            zip(targets, output_names, strict=True)
        ):
            output_sample = ot.Sample([[float(value)] for value in outputs[:, target]])
            analysis = ot.CorrelationAnalysis(input_sample, output_sample)
            coefficients = {
                "pearson": analysis.computeLinearCorrelation(),
                "spearman": analysis.computeSpearmanCorrelation(),
                "partial": analysis.computePCC(),
                "src": analysis.computeSRC(),
                "srrc": analysis.computeSRRC(),
            }
            for index, input_name in enumerate(names):
                coefficient_values = [_safe_stat(coefficients[key][index]) for key in measures]
                rows.append([output_name, input_name, *coefficient_values])
                for key, value in zip(measures, coefficient_values, strict=True):
                    if len(measures[key]) <= output_index:
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
    return float(value) if math.isfinite(value) else None


plugin = CorrelationPlugin()
