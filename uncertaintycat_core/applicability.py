"""Catalog-wide, deterministic model-to-analysis compatibility assessment."""

from __future__ import annotations

from uncertaintycat_core.catalog import analysis_catalog, get_plugin
from uncertaintycat_core.contracts import AnalysisRecommendation
from uncertaintycat_core.model import ModelRuntime


def _declared_model_constraints(plugin_key: str, runtime: ModelRuntime) -> list[tuple[str, str]]:
    """Lift existing config-independent plugin constraints into validation-time UX."""

    issues: list[tuple[str, str]] = []
    continuous = runtime.problem.isContinuous()
    dimension = runtime.metadata.input_dimension
    if plugin_key == "ancova":
        if not 2 <= dimension <= 10:
            issues.append(
                (
                    "ANCOVA_DIMENSION_LIMIT",
                    "ANCOVA requires between two and ten input variables for its "
                    "validated dependent-input PCE decomposition.",
                )
            )
        if not continuous:
            issues.append(
                (
                    "CONTINUOUS_INPUTS_REQUIRED",
                    "ANCOVA currently requires every input marginal to be continuous.",
                )
            )
    if plugin_key in {"gpr", "pce"} and not continuous:
        issues.append(
            (
                "CONTINUOUS_INPUTS_REQUIRED",
                "Gaussian process surrogates currently require every input marginal to "
                "be continuous."
                if plugin_key == "gpr"
                else "Polynomial chaos currently requires every input marginal to be continuous.",
            )
        )
    if plugin_key == "target_hsic":
        if dimension > 20:
            issues.append(
                (
                    "TARGET_HSIC_DIMENSION_LIMIT",
                    "Target-domain HSIC is limited to 20 inputs because its kernel and "
                    "permutation work grows quadratically.",
                )
            )
        if not continuous:
            issues.append(
                (
                    "CONTINUOUS_INPUTS_REQUIRED",
                    "Target-domain HSIC requires continuous input marginals because it "
                    "uses Gaussian input kernels.",
                )
            )
    if plugin_key == "calibration_nlls" and dimension > 32:
        issues.append(
            (
                "CALIBRATION_DIMENSION_LIMIT",
                "Nonlinear least-squares calibration is limited to models with at most "
                "32 inputs in the bounded calibration workspace.",
            )
        )
    if plugin_key == "calibration_nlls":
        input_names = [item.name for item in runtime.metadata.inputs]
        if len(input_names) != len(set(input_names)):
            issues.append(
                (
                    "UNIQUE_INPUT_NAMES_REQUIRED",
                    "Nonlinear least-squares calibration requires unique input names for "
                    "the named observation contract.",
                )
            )
        if not any(item.kind == "continuous" for item in runtime.metadata.inputs):
            issues.append(
                (
                    "CONTINUOUS_PARAMETER_REQUIRED",
                    "Nonlinear least-squares calibration requires at least one continuous "
                    "input that can be selected as a calibration parameter.",
                )
            )
    return issues


def _available_rationale(
    plugin_key: str, runtime: ModelRuntime, expensive: bool
) -> tuple[str, int]:
    dimension = runtime.metadata.input_dimension
    if plugin_key == "monte_carlo":
        return "BASELINE_PROPAGATION", 1
    if plugin_key == "eda":
        return "BASELINE_OUTPUT_CHARACTERISATION", 1
    if plugin_key == "convergence":
        return "BASELINE_CONVERGENCE_EVIDENCE", 1
    if plugin_key == "ancova":
        return "DEPENDENT_INPUT_VARIANCE_DECOMPOSITION", 2
    if plugin_key == "morris":
        return (
            "HIGH_DIMENSION_SCREENING" if dimension >= 8 else "LOW_DIMENSION_SCREENING_OPTIONAL"
        ), 2 if dimension >= 15 else 3
    if plugin_key in {"gpr", "pce"}:
        return (
            "DIRECT_MODEL_RUNTIME_EXCEEDS_FIVE_SECONDS"
            if expensive
            else "DIRECT_MODEL_RUNTIME_WITHIN_FIVE_SECONDS",
            3,
        )
    if plugin_key == "reliability":
        return "USER_DEFINED_FAILURE_EVENT_REQUIRED", 4
    if plugin_key == "target_hsic":
        return "USER_DEFINED_CRITICAL_DOMAIN_REQUIRED", 4
    return "PLUGIN_MODEL_CONTRACT_SATISFIED", 3


def build_analysis_recommendations(
    runtime: ModelRuntime,
    *,
    projected_1000_evaluation_runtime_ms: float,
) -> list[AnalysisRecommendation]:
    """Evaluate every registered plugin against one compiled immutable model."""

    expensive = projected_1000_evaluation_runtime_ms > 5_000
    recommendations: list[AnalysisRecommendation] = []
    baseline = {"monte_carlo", "eda", "convergence"}
    for entry in analysis_catalog():
        plugin = get_plugin(entry.key)
        issues = [
            *plugin.model_incompatibility_reasons(runtime),
            *_declared_model_constraints(entry.key, runtime),
        ]
        rationale, priority = _available_rationale(entry.key, runtime, expensive)
        status = (
            "incompatible"
            if issues
            else "recommended"
            if entry.key in baseline
            or entry.key == "ancova"
            or (entry.key == "morris" and runtime.metadata.input_dimension >= 8)
            or (entry.key in {"gpr", "pce"} and expensive)
            else "available"
        )
        guidance: list[str] = []
        if not issues and entry.key == "reliability":
            guidance.append(
                "Define a scalar failure event and select a compatible method before execution."
            )
        if not issues and entry.key == "target_hsic":
            guidance.append("Define a scalar critical output domain before target-HSIC execution.")
        recommendations.append(
            AnalysisRecommendation(
                capability=entry.key,
                status=status,
                priority=priority,
                rationale_codes=[code for code, _ in issues] if issues else [rationale],
                projected_evaluations=(
                    1_000
                    if entry.key in baseline
                    else 250
                    if entry.key in {"hsic", "target_hsic"}
                    else None
                ),
                projected_runtime_ms=(
                    projected_1000_evaluation_runtime_ms if entry.key in baseline else None
                ),
                compatibility_warnings=[message for _, message in issues] or guidance,
                safe_config=plugin.safe_model_config(runtime),
            )
        )
    recommendations.append(
        AnalysisRecommendation(
            capability="distribution_fitting",
            status="incompatible",
            priority=4,
            rationale_codes=["NO_EMPIRICAL_DATA_ATTACHED"],
            compatibility_warnings=[
                "Attach empirical data in Data Lab before fitting distributions."
            ],
        )
    )
    return recommendations
