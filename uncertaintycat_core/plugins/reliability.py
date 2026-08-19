"""Threshold-event reliability analysis with FORM and direct simulation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import openturns as ot
from pydantic import Field
from scipy import stats

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin

OPERATORS = {">": ot.Greater(), ">=": ot.GreaterOrEqual(), "<": ot.Less(), "<=": ot.LessOrEqual()}


class ReliabilityConfig(StrictModel):
    method: Literal["FORM", "MONTE_CARLO"] = "FORM"
    threshold: float
    operator: Literal[">", ">=", "<", "<="] = ">"
    sample_size: int = Field(default=20_000, ge=100, le=2_000_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class ReliabilityPlugin(AnalysisPlugin[ReliabilityConfig]):
    key = "reliability"
    version = "1.0.0"
    name = "Reliability Analysis"
    category = "Reliability"
    description = "Estimate a threshold-event probability with FORM or reproducible Monte Carlo."
    assumptions = (
        "FORM is a local approximation around the most probable failure point.",
        "Threshold direction defines the failure event.",
    )
    supports_multi_output = False
    resource_class = "heavy"
    config_model = ReliabilityConfig

    def run(self, runtime: ModelRuntime, config: ReliabilityConfig) -> tuple[AnalysisPayload, int]:
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        if config.method == "MONTE_CARLO":
            return self._monte_carlo(runtime, config, target)
        selected_model = runtime.model.getMarginal(target)
        event = ot.ThresholdEvent(
            ot.CompositeRandomVector(selected_model, ot.RandomVector(runtime.problem)),
            OPERATORS[config.operator],
            config.threshold,
        )
        optimizer = ot.Cobyla()
        optimizer.setMaximumCallsNumber(10_000)
        optimizer.setStartingPoint(runtime.problem.getMean())
        try:
            algorithm = ot.FORM(optimizer, event)
            algorithm.run()
            result = algorithm.getResult()
        except Exception as exc:
            raise IncompatibleAnalysisError(f"FORM could not locate a design point: {exc}") from exc
        probability = float(result.getEventProbability())
        beta = float(result.getHasoferReliabilityIndex())
        design = list(result.getPhysicalSpaceDesignPoint())
        importance = list(result.getImportanceFactors())
        names = [item.name for item in runtime.metadata.inputs]
        rows = [[name, float(design[i]), float(importance[i])] for i, name in enumerate(names)]
        return AnalysisPayload(
            metrics={
                "event_probability": probability,
                "reliability_index": beta,
                "threshold": config.threshold,
            },
            tables={
                "design_point": TableData(
                    columns=["Variable", "Physical Design Point", "Importance Factor"],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "method": "FORM",
                "operator": config.operator,
                "output": runtime.metadata.outputs[target].name,
            },
        ), int(result.getOptimizationResult().getCallsNumber())

    def _monte_carlo(
        self, runtime: ModelRuntime, config: ReliabilityConfig, target: int
    ) -> tuple[AnalysisPayload, int]:
        _, outputs = runtime.sample_and_evaluate(config.sample_size, config.seed)
        values = outputs[:, target]
        failures = {
            ">": values > config.threshold,
            ">=": values >= config.threshold,
            "<": values < config.threshold,
            "<=": values <= config.threshold,
        }[config.operator]
        count = int(np.sum(failures))
        probability = count / config.sample_size
        interval = stats.binomtest(count, config.sample_size).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        beta = float(stats.norm.isf(probability)) if 0 < probability < 1 else None
        return AnalysisPayload(
            metrics={
                "event_probability": probability,
                "reliability_index": beta,
                "threshold": config.threshold,
                "failures": count,
                "sample_size": config.sample_size,
                "confidence_lower": float(interval.low),
                "confidence_upper": float(interval.high),
            },
            facts={
                "method": "MONTE_CARLO",
                "operator": config.operator,
                "output": runtime.metadata.outputs[target].name,
            },
        ), config.sample_size


plugin = ReliabilityPlugin()
