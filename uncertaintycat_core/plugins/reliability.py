"""Guided threshold-event reliability with stable OpenTURNS algorithms."""

from __future__ import annotations

import json
import math
import time
from typing import Any, Literal

import openturns as ot
from pydantic import Field, FiniteFloat, StrictInt

from uncertaintycat_core.contracts import (
    AnalysisPayload,
    SeriesData,
    StrictModel,
    TableData,
)
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.progress import report_progress

SUBSET_MAXIMUM_DIMENSION = 20
SUBSET_MAXIMUM_EVALUATIONS = 50_000
SUBSET_MAXIMUM_LEVELS = 10
SUBSET_MAXIMUM_SECONDS = 60.0
SUBSET_MAXIMUM_PAYLOAD_BYTES = 16_384
SUBSET_CONSTANT_REASON = (
    "Subset sampling requires a selected output that varies in the validation pilot."
)


def subset_evaluation_limit(sample_size: int, budget: int) -> int:
    """Reserve complete populations; the point guard also enforces this bound at runtime."""
    return sample_size * min(SUBSET_MAXIMUM_LEVELS, budget // sample_size)


def _subset_model_reason(runtime: ModelRuntime) -> str | None:
    if runtime.metadata.input_dimension > SUBSET_MAXIMUM_DIMENSION:
        return "Subset sampling is limited to 20 inputs in the bounded reliability workspace."
    if not runtime.problem.isContinuous():
        return "Subset sampling requires continuous inputs for its standard-space transformation."
    return None


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
    threshold: FiniteFloat
    operator: Literal[">", ">=", "<", "<="] = ">"
    maximum_evaluations: int = Field(default=20_000, ge=100, le=2_000_000)
    sample_size: int | None = Field(default=None, ge=100, le=2_000_000)
    target_coefficient_of_variation: float = Field(default=0.05, gt=0, le=1)
    block_size: int = Field(default=1, ge=1, le=10_000)
    subset_sample_size: StrictInt = Field(default=2_000, ge=100, le=5_000, multiple_of=10)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class ReliabilityPlugin(AnalysisPlugin[ReliabilityConfig]):
    key = "reliability"
    version = "3.0.0"
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

    def parse_config(
        self, raw: dict[str, Any], *, seed: int, output_targets: list[int]
    ) -> ReliabilityConfig:
        if (
            raw.get("method") == "SUBSET_SAMPLING"
            and output_targets
            and "output_targets" in raw
            and raw["output_targets"] != output_targets
        ):
            raise IncompatibleAnalysisError("Conflicting subset output targets are not allowed.")
        return super().parse_config(raw, seed=seed, output_targets=output_targets)

    def safe_model_config(
        self, runtime: ModelRuntime
    ) -> dict[str, str | int | float | bool | None]:
        reason = _subset_model_reason(runtime)
        return {
            "subset_sampling_available": reason is None,
            "subset_sampling_incompatibility": reason,
            "subset_sample_size": 2_000,
            "subset_maximum_sample_size": 5_000,
            "subset_maximum_evaluations": SUBSET_MAXIMUM_EVALUATIONS,
            "subset_maximum_levels": SUBSET_MAXIMUM_LEVELS,
        }

    def applicability_warnings(self, runtime: ModelRuntime, config: ReliabilityConfig) -> list[str]:
        if config.method == "SUBSET_SAMPLING":
            target = config.output_targets[0] if config.output_targets else 0
            if not 0 <= target < runtime.metadata.output_dimension:
                raise IncompatibleAnalysisError("The requested output target does not exist.")
            reason = _subset_model_reason(runtime)
            if reason:
                raise IncompatibleAnalysisError(reason)
            if not runtime.assessment.profile.pilot_outputs[target].variable:
                raise IncompatibleAnalysisError(SUBSET_CONSTANT_REASON)
            if config.sample_size is not None:
                raise IncompatibleAnalysisError(
                    "For subset sampling use subset_sample_size per level and "
                    "maximum_evaluations for the total budget; sample_size is ambiguous."
                )
            if config.block_size != 1:
                raise IncompatibleAnalysisError(
                    "Bounded subset sampling requires block_size=1 to preserve chain ordering."
                )
            if config.maximum_evaluations > SUBSET_MAXIMUM_EVALUATIONS:
                raise IncompatibleAnalysisError(
                    "Subset sampling permits at most 50,000 total model evaluations."
                )
            if config.subset_sample_size > config.maximum_evaluations:
                raise IncompatibleAnalysisError(
                    "Subset samples per level must not exceed the total model-evaluation budget."
                )
            return [
                "Subset sampling stops when its adaptive thresholds reach the requested event, "
                "not when a target coefficient of variation is reached. The legacy precision "
                "setting is unused for this method.",
                "The nominal 95% interval uses OpenTURNS' Normal approximation with estimated "
                "within-chain correlation. Between-level dependence and finite-sample bias can "
                "make it optimistic; it is not an exact confidence guarantee.",
                "Reaching the event threshold does not establish chain mixing, causal validity, "
                "or predictive validity outside the declared input distribution.",
            ]
        if config.method != "MONTE_CARLO" and not runtime.problem.isContinuous():
            raise IncompatibleAnalysisError(
                f"{config.method.replace('_', ' ').title()} requires continuous inputs "
                "for the OpenTURNS standard-space transformation."
            )
        return []

    def run(self, runtime: ModelRuntime, config: ReliabilityConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target < 0 or target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        if config.method == "SUBSET_SAMPLING":
            return self._subset(runtime, config, target)
        selected_model = runtime.model.getMarginal(target)
        event = ot.ThresholdEvent(
            ot.CompositeRandomVector(selected_model, ot.RandomVector(runtime.problem)),
            OPERATORS[config.operator],
            config.threshold,
        )
        if config.method in {"FORM", "SORM"}:
            return self._analytical(runtime, config, target, event)
        return self._simulation(runtime, config, target, event)

    def _subset(
        self, runtime: ModelRuntime, config: ReliabilityConfig, target: int
    ) -> tuple[AnalysisPayload, int]:
        population = config.subset_sample_size
        limit = subset_evaluation_limit(population, config.maximum_evaluations)
        maximum_levels = limit // population
        calls = 0
        failure: str | None = None
        started = time.perf_counter()

        def evaluate(point: ot.Point) -> list[float]:
            nonlocal calls, failure
            if calls >= limit:
                failure = "Subset sampling exhausted its model-evaluation budget before completion."
            elif time.perf_counter() - started >= SUBSET_MAXIMUM_SECONDS:
                failure = "Subset sampling exceeded its time budget before completion."
            if failure:
                raise RuntimeError(failure)
            if calls % population == 0:
                report_progress(
                    "subset_population",
                    18,
                    f"Evaluating subset population {calls // population + 1} "
                    f"of at most {maximum_levels}.",
                    indeterminate=True,
                )
            calls += 1
            try:
                # One invocation per point, no sample fallback or inferred OT counter.
                value = float(runtime.model(point)[target])
            except Exception:
                failure = "The model failed during subset sampling."
                raise RuntimeError(failure) from None
            if not math.isfinite(value):
                failure = "The model returned a non-finite value during subset sampling."
                raise RuntimeError(failure)
            return [value]

        guarded_model = ot.PythonFunction(runtime.metadata.input_dimension, 1, evaluate)
        event = ot.ThresholdEvent(
            ot.CompositeRandomVector(guarded_model, ot.RandomVector(runtime.problem)),
            OPERATORS[config.operator],
            config.threshold,
        )
        try:
            algorithm = ot.SubsetSampling(event, 2.0, 0.1)
            algorithm.setBlockSize(1)
            # This is population size, NOT an all-level evaluation limit.
            algorithm.setMaximumOuterSampling(population)
            algorithm.setMaximumTimeDuration(SUBSET_MAXIMUM_SECONDS)
            algorithm.setKeepSample(False)
            algorithm.setConvergenceStrategy(ot.Compact(10))
            ot.RandomGenerator.SetSeed(config.seed)
            algorithm.run()
            result = algorithm.getResult()
        except Exception:
            raise IncompatibleAnalysisError(
                failure or "OpenTURNS could not complete bounded subset sampling.",
                details={"model_evaluations": calls, "effective_evaluation_limit": limit},
            ) from None
        thresholds = [float(value) for value in algorithm.getThresholdPerStep()]
        probabilities = [float(value) for value in algorithm.getProbabilityEstimatePerStep()]
        levels = int(algorithm.getStepsNumber())
        if (
            failure
            or time.perf_counter() - started >= SUBSET_MAXIMUM_SECONDS
            or not 1 <= levels <= maximum_levels
            or len(thresholds) != levels
            or len(probabilities) != levels
            or thresholds[-1] != config.threshold
            or calls != population * levels
        ):
            raise IncompatibleAnalysisError(
                "Subset sampling did not complete the requested event within its bounds; "
                "no intermediate-domain probability is reported.",
                details={"model_evaluations": calls, "effective_evaluation_limit": limit},
            )
        probability = float(result.getProbabilityEstimate())
        variance = float(result.getVarianceEstimate())
        coefficient = float(result.getCoefficientOfVariation())
        if (
            not all(
                math.isfinite(value)
                for value in [probability, variance, coefficient, *thresholds, *probabilities]
            )
            or not 0.0 < probability < 1.0
            or variance <= 0.0
            or coefficient <= 0.0
        ):
            raise IncompatibleAnalysisError(
                "Subset sampling produced degenerate probability/uncertainty evidence; "
                "this does not prove the event impossible or certain.",
                details={"model_evaluations": calls},
            )
        interval = result.getProbabilityDistribution().computeBilateralConfidenceInterval(0.95)
        payload = AnalysisPayload(
            metrics={
                "event_probability": probability,
                "threshold": config.threshold,
                "standard_error": float(result.getStandardDeviation()),
                "coefficient_of_variation": coefficient,
                "confidence_lower": max(0.0, float(interval.getLowerBound()[0])),
                "confidence_upper": min(1.0, float(interval.getUpperBound()[0])),
                "model_evaluations": calls,
                "requested_evaluation_budget": config.maximum_evaluations,
                "effective_evaluation_limit": limit,
                "samples_per_level": population,
                "completed_levels": levels,
                "maximum_levels": maximum_levels,
            },
            tables={
                "subset_levels": TableData(
                    columns=["Level", "Output Threshold", "Cumulative Probability Estimate"],
                    rows=[
                        [index + 1, threshold, probabilities[index]]
                        for index, threshold in enumerate(thresholds)
                    ],
                    row_count=levels,
                )
            },
            facts={
                "method": config.method,
                "operator": config.operator,
                "output": runtime.metadata.outputs[target].name[:200],
                "stopping_reason": "requested event threshold reached",
                "sampling_estimate": True,
                "conditional_probability": 0.1,
                "proposal_range": 2.0,
                "coefficient_of_variation_is_stopping_target": False,
                "uncertainty_approximation": (
                    "Nominal 95% OpenTURNS Normal interval, clipped to [0,1]; estimated "
                    "within-chain correlation, neglecting between-level dependence. "
                    "Not an exact confidence guarantee."
                ),
                "evaluation_accounting": (
                    "Exact top-level model point invocations during this analysis, including "
                    "rejected MCMC proposals. Excludes source construction/validation and "
                    "does not count internal subcomputations of the user model."
                ),
                "history_interpretation": (
                    "Each row estimates its own intermediate threshold event. Only the final "
                    "row estimates the requested event; these are not repeated estimates "
                    "or a convergence/confidence trace for one fixed event."
                ),
            },
        )
        serialized = json.dumps(payload.model_dump(mode="json"), allow_nan=False)
        if len(serialized.encode()) > SUBSET_MAXIMUM_PAYLOAD_BYTES:
            raise IncompatibleAnalysisError("Subset sampling exceeded its bounded report size.")
        return payload, calls

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
        else:
            algorithm = ot.DirectionalSampling(event)
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
