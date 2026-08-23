"""Guided threshold-event reliability with stable OpenTURNS algorithms."""

from __future__ import annotations

import math
from typing import Literal

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import (
    AnalysisPayload,
    SeriesData,
    StrictModel,
    TableData,
)
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin

OPERATORS = {
    ">": ot.Greater(),
    ">=": ot.GreaterOrEqual(),
    "<": ot.Less(),
    "<=": ot.LessOrEqual(),
}


class ReliabilityConfig(StrictModel):
    method: Literal["FORM", "SORM", "MONTE_CARLO", "DIRECTIONAL_SAMPLING", "SUBSET_SAMPLING"] = (
        "FORM"
    )
    threshold: float
    operator: Literal[">", ">=", "<", "<="] = ">"
    maximum_evaluations: int = Field(default=20_000, ge=100, le=2_000_000)
    sample_size: int | None = Field(default=None, ge=100, le=2_000_000)
    target_coefficient_of_variation: float = Field(default=0.05, gt=0, le=1)
    block_size: int = Field(default=1, ge=1, le=10_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class ReliabilityPlugin(AnalysisPlugin[ReliabilityConfig]):
    key = "reliability"
    version = "2.0.0"
    name = "Reliability Analysis"
    category = "Reliability"
    description = "Evaluate an explicit failure event with stable OpenTURNS reliability methods."
    assumptions = (
        "FORM and SORM are local design-point approximations.",
        "Simulation methods are sampling estimates with convergence uncertainty.",
        "Threshold direction defines the failure event.",
    )
    supports_multi_output = False
    resource_class = "heavy"
    config_model = ReliabilityConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: ReliabilityConfig) -> list[str]:
        if config.method != "MONTE_CARLO" and not runtime.problem.isContinuous():
            raise IncompatibleAnalysisError(
                f"{config.method.replace('_', ' ').title()} requires continuous inputs "
                "for the OpenTURNS standard-space transformation."
            )
        return []

    def run(self, runtime: ModelRuntime, config: ReliabilityConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        selected_model = runtime.model.getMarginal(target)
        event = ot.ThresholdEvent(
            ot.CompositeRandomVector(selected_model, ot.RandomVector(runtime.problem)),
            OPERATORS[config.operator],
            config.threshold,
        )
        if config.method in {"FORM", "SORM"}:
            return self._analytical(runtime, config, target, event)
        return self._simulation(runtime, config, target, event)

    def _analytical(
        self,
        runtime: ModelRuntime,
        config: ReliabilityConfig,
        target: int,
        event: ot.RandomVector,
    ) -> tuple[AnalysisPayload, int]:
        optimizer = ot.Cobyla()
        optimizer.setMaximumCallsNumber(config.maximum_evaluations)
        optimizer.setStartingPoint(runtime.problem.getMean())
        try:
            algorithm = (
                ot.FORM(optimizer, event) if config.method == "FORM" else ot.SORM(optimizer, event)
            )
            algorithm.run()
            result = algorithm.getResult()
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"{config.method} could not locate a design point: {exc}"
            ) from exc
        if config.method == "FORM":
            probability = float(result.getEventProbability())
            probability_metrics: dict[str, float] = {"event_probability": probability}
        else:
            probability = float(result.getEventProbabilityBreitung())
            probability_metrics = {
                "event_probability": probability,
                "event_probability_breitung": probability,
                "event_probability_hohenbichler": float(result.getEventProbabilityHohenbichler()),
                "event_probability_tvedt": float(result.getEventProbabilityTvedt()),
            }
        beta = float(result.getHasoferReliabilityIndex())
        physical = result.getPhysicalSpaceDesignPoint()
        standard = result.getStandardSpaceDesignPoint()
        importance = result.getImportanceFactors()
        names = [item.name for item in runtime.metadata.inputs]
        rows = [
            [name, float(physical[index]), float(standard[index]), float(importance[index])]
            for index, name in enumerate(names)
        ]
        calls = int(result.getOptimizationResult().getCallsNumber())
        return AnalysisPayload(
            metrics={
                **probability_metrics,
                "reliability_index": beta,
                "threshold": config.threshold,
                "model_evaluations": calls,
            },
            tables={
                "design_point": TableData(
                    columns=[
                        "Variable",
                        "Physical Design Point",
                        "Standard Design Point",
                        "Importance Factor",
                    ],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "method": config.method,
                "operator": config.operator,
                "output": runtime.metadata.outputs[target].name,
                "stopping_reason": "design point optimization completed",
                "local_approximation": True,
            },
        ), calls

    def _simulation(
        self,
        runtime: ModelRuntime,
        config: ReliabilityConfig,
        target: int,
        event: ot.RandomVector,
    ) -> tuple[AnalysisPayload, int]:
        budget = config.sample_size or config.maximum_evaluations
        block_size = min(config.block_size, budget)
        if config.method == "MONTE_CARLO":
            algorithm: ot.SimulationAlgorithm = ot.ProbabilitySimulationAlgorithm(
                event, ot.MonteCarloExperiment()
            )
        elif config.method == "DIRECTIONAL_SAMPLING":
            algorithm = ot.DirectionalSampling(event)
        else:
            algorithm = ot.SubsetSampling(event)
        algorithm.setBlockSize(block_size)
        algorithm.setMaximumOuterSampling(math.ceil(budget / block_size))
        algorithm.setMaximumCoefficientOfVariation(config.target_coefficient_of_variation)
        algorithm.setMaximumStandardDeviation(-1.0)
        algorithm.setConvergenceStrategy(ot.Compact(250))
        calls_before = runtime.model.getEvaluationCallsNumber()
        ot.RandomGenerator.SetSeed(config.seed)
        try:
            algorithm.run()
            result = algorithm.getResult()
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"{config.method.replace('_', ' ').title()} could not estimate the event: {exc}"
            ) from exc
        calls = max(0, runtime.model.getEvaluationCallsNumber() - calls_before)
        probability = float(result.getProbabilityEstimate())
        standard_error = float(result.getStandardDeviation())
        coefficient = float(result.getCoefficientOfVariation())
        lower = max(0.0, probability - 1.96 * standard_error)
        upper = min(1.0, probability + 1.96 * standard_error)
        history = algorithm.getConvergenceStrategy().getSample()
        x = [int(round(row[2] * block_size)) for row in history]
        estimates = [float(row[0]) for row in history]
        history_variances = [max(0.0, float(row[1])) for row in history]
        stopping_reason = (
            "target coefficient of variation reached"
            if math.isfinite(coefficient) and coefficient <= config.target_coefficient_of_variation
            else "maximum evaluations reached"
        )
        return AnalysisPayload(
            metrics={
                "event_probability": probability,
                "reliability_index": (
                    float(ot.Normal().computeQuantile(1.0 - probability)[0])
                    if 0.0 < probability < 1.0
                    else None
                ),
                "threshold": config.threshold,
                "standard_error": standard_error,
                "coefficient_of_variation": coefficient,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "model_evaluations": calls,
            },
            series={
                "probability_history": SeriesData(
                    name="Failure probability estimate",
                    x=x,
                    y=estimates,
                    x_label="Simulation blocks",
                    y_label="Probability of failure",
                ),
                "confidence_lower_history": SeriesData(
                    name="95% confidence lower",
                    x=x,
                    y=[
                        max(0.0, estimate - 1.96 * math.sqrt(variance))
                        for estimate, variance in zip(estimates, history_variances, strict=True)
                    ],
                ),
                "confidence_upper_history": SeriesData(
                    name="95% confidence upper",
                    x=x,
                    y=[
                        min(1.0, estimate + 1.96 * math.sqrt(variance))
                        for estimate, variance in zip(estimates, history_variances, strict=True)
                    ],
                ),
            },
            facts={
                "method": config.method,
                "operator": config.operator,
                "output": runtime.metadata.outputs[target].name,
                "stopping_reason": stopping_reason,
                "sampling_estimate": True,
            },
        ), calls


plugin = ReliabilityPlugin()
