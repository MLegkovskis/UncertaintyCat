"""Serializable exploratory data analysis built on shared model samples."""

from __future__ import annotations

import math

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, MatrixData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class EdaConfig(StrictModel):
    sample_size: int = Field(default=1000, ge=20, le=200_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list)


class EdaPlugin(AnalysisPlugin[EdaConfig]):
    key = "eda"
    version = "2.0.0"
    name = "Exploratory Data Analysis"
    category = "Exploration"
    description = "Describe outputs and quantify linear and rank relationships with inputs."
    config_model = EdaConfig

    def run(self, runtime: ModelRuntime, config: EdaConfig) -> tuple[AnalysisPayload, int]:
        targets = config.output_targets or list(range(runtime.metadata.output_dimension))
        if any(target >= runtime.metadata.output_dimension for target in targets):
            raise IncompatibleAnalysisError("A requested output target does not exist.")
        inputs, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        input_sample = ot.Sample(inputs.tolist())
        input_names = [item.name for item in runtime.metadata.inputs]
        summary_rows: list[list[float | str]] = []
        pearson_values: list[list[float | None]] = []
        spearman_values: list[list[float | None]] = []
        facts: dict[str, float | str | int | bool | None] = {}
        for target in targets:
            name = runtime.metadata.outputs[target].name
            output_sample = ot.Sample([[float(value)] for value in outputs[:, target]])
            mean = output_sample.computeMean()[0]
            standard_deviation = output_sample.computeStandardDeviation()[0]
            summary_rows.append(
                [
                    name,
                    float(mean),
                    float(standard_deviation),
                    float(output_sample.computeSkewness()[0]),
                    float(output_sample.computeKurtosis()[0] - 3.0),
                    float(output_sample.computeQuantilePerComponent(0.25)[0]),
                    float(output_sample.computeQuantilePerComponent(0.5)[0]),
                    float(output_sample.computeQuantilePerComponent(0.75)[0]),
                ]
            )
            analysis = ot.CorrelationAnalysis(input_sample, output_sample)
            pearson_row = [_safe(value) for value in analysis.computeLinearCorrelation()]
            spearman_row = [_safe(value) for value in analysis.computeSpearmanCorrelation()]
            pearson_values.append(pearson_row)
            spearman_values.append(spearman_row)
            ranked = sorted(
                zip(input_names, pearson_row, strict=True),
                key=lambda pair: abs(pair[1] or 0.0),
                reverse=True,
            )
            if ranked:
                facts[f"{name}.strongest_linear_input"] = ranked[0][0]
                facts[f"{name}.strongest_linear_correlation"] = ranked[0][1]

        output_names = [runtime.metadata.outputs[target].name for target in targets]
        return (
            AnalysisPayload(
                metrics={"sample_size": config.sample_size, "output_count": len(targets)},
                tables={
                    "summary": TableData(
                        columns=[
                            "Output",
                            "Mean",
                            "Sample Std",
                            "Skewness",
                            "Excess Kurtosis",
                            "25%",
                            "Median",
                            "75%",
                        ],
                        rows=summary_rows,
                        row_count=len(summary_rows),
                    )
                },
                matrices={
                    "pearson": MatrixData(
                        row_labels=output_names, column_labels=input_names, values=pearson_values
                    ),
                    "spearman": MatrixData(
                        row_labels=output_names, column_labels=input_names, values=spearman_values
                    ),
                },
                facts=facts,
            ),
            config.sample_size,
        )


plugin = EdaPlugin()


def _safe(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
