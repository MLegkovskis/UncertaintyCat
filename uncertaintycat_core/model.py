"""Compilation, preflight checks, and validated access to OpenTURNS models."""

from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import openturns as ot

from uncertaintycat_core.contracts import (
    AnalysisRecommendation,
    ModelAssessment,
    ModelMetadata,
    ModelProfile,
    OutputMetadata,
    PilotOutputSummary,
    VariableMetadata,
    WorkflowRecommendation,
)
from uncertaintycat_core.errors import InvalidModelError, UnsafeModelError

MAX_SOURCE_BYTES = 256 * 1024
ALLOWED_IMPORT_ROOTS = {
    "math",
    "numpy",
    "openturns",
    "pandas",
    "scipy",
}
BLOCKED_CALLS = {"__import__", "compile", "eval", "exec", "input", "open"}


def recommend_workflow(
    *,
    input_dimension: int,
    projected_1000_evaluation_runtime_ms: float,
    surrogate_eligible: bool,
) -> WorkflowRecommendation:
    """Choose the next scientific workspace from measured model properties."""
    if input_dimension >= 15:
        return WorkflowRecommendation(
            path="dimensionality_reduction",
            rationale_codes=["HIGH_DIMENSION_SCREENING"],
        )
    if projected_1000_evaluation_runtime_ms > 5_000 and surrogate_eligible:
        return WorkflowRecommendation(
            path="surrogate",
            rationale_codes=["DIRECT_MODEL_RUNTIME_EXCEEDS_FIVE_SECONDS"],
        )
    return WorkflowRecommendation(
        path="direct",
        rationale_codes=["DIRECT_EVALUATION_PRACTICAL"],
    )


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def preflight_source(source: str) -> None:
    """Provide fast feedback; isolation, not this AST check, is the security boundary."""
    if not source.strip():
        raise InvalidModelError("Model source is empty.")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise UnsafeModelError(f"Model source exceeds the {MAX_SOURCE_BYTES} byte limit.")
    try:
        tree = ast.parse(source, filename="model.py")
    except SyntaxError as exc:
        raise InvalidModelError(
            f"Model source has invalid Python syntax at line {exc.lineno}: {exc.msg}.",
            details={"line": exc.lineno, "offset": exc.offset},
        ) from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".", 1)[0] for alias in node.names]
            rejected = sorted(set(roots) - ALLOWED_IMPORT_ROOTS)
            if rejected:
                raise UnsafeModelError(
                    f"Import '{rejected[0]}' is not in the curated scientific runtime.",
                    details={"allowed_imports": sorted(ALLOWED_IMPORT_ROOTS)},
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise UnsafeModelError(
                    f"Import '{root or '<relative>'}' is not in the curated scientific runtime.",
                    details={"allowed_imports": sorted(ALLOWED_IMPORT_ROOTS)},
                )
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                raise UnsafeModelError(f"Call to '{node.func.id}' is not allowed in model source.")


def _description(values: Any, size: int, prefix: str) -> list[str]:
    try:
        raw = list(values)
    except Exception:
        raw = []
    return [
        str(raw[i]) if i < len(raw) and str(raw[i]).strip() else f"{prefix}{i + 1}"
        for i in range(size)
    ]


def _as_2d_output(raw: Any, expected_rows: int) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(expected_rows, -1) if expected_rows > 1 else values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise InvalidModelError(
            "Model returned an inconsistent output shape.",
            details={"expected_rows": expected_rows, "actual_shape": list(values.shape)},
        )
    if not np.isfinite(values).all():
        raise InvalidModelError("Model validation produced NaN or infinite output values.")
    return values


@dataclass
class ModelRuntime:
    source: str
    model: ot.Function
    problem: ot.Distribution
    metadata: ModelMetadata
    assessment: ModelAssessment
    _sample_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    def sample_and_evaluate(self, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        key = (sample_size, seed)
        if key not in self._sample_cache:
            ot.RandomGenerator.SetSeed(seed)
            input_sample = self.problem.getSample(sample_size)
            try:
                raw_output = self.model(input_sample)
                output = _as_2d_output(raw_output, sample_size)
            except Exception:
                rows = [_as_2d_output(self.model(point), 1)[0] for point in input_sample]
                output = np.asarray(rows, dtype=float)
            self._sample_cache[key] = (np.asarray(input_sample, dtype=float), output)
        return self._sample_cache[key]


def compile_model(source: str, *, validation_sample_size: int = 8, seed: int = 42) -> ModelRuntime:
    preflight_source(source)
    started = time.perf_counter()
    namespace: dict[str, Any] = {"__name__": "__uncertaintycat_model__"}
    try:
        exec(compile(source, "model.py", "exec"), namespace)
    except Exception as exc:
        raise InvalidModelError(f"Model source could not be evaluated: {exc}") from exc

    model = namespace.get("model")
    problem = namespace.get("problem")
    if not isinstance(model, ot.Function):
        raise InvalidModelError("Model source must define 'model' as an OpenTURNS Function.")
    # OpenTURNS' SWIG hierarchy exposes concrete distributions as
    # DistributionImplementation instances rather than instances of the
    # ot.Distribution handle class.
    if not isinstance(problem, ot.DistributionImplementation):
        raise InvalidModelError("Model source must define 'problem' as an OpenTURNS Distribution.")
    input_dimension = model.getInputDimension()
    if input_dimension != problem.getDimension():
        raise InvalidModelError(
            "Model input dimension does not match the problem distribution.",
            details={"model": input_dimension, "problem": problem.getDimension()},
        )

    ot.RandomGenerator.SetSeed(seed)
    sample = problem.getSample(validation_sample_size)
    evaluation_started = time.perf_counter()
    batch_supported = True
    try:
        output = _as_2d_output(model(sample), validation_sample_size)
    except Exception:
        batch_supported = False
        try:
            output = np.asarray([_as_2d_output(model(point), 1)[0] for point in sample])
        except Exception as exc:
            raise InvalidModelError(f"Model failed its validation evaluations: {exc}") from exc
    evaluation_runtime_ms = (time.perf_counter() - evaluation_started) * 1000

    output_dimension = model.getOutputDimension()
    if output.shape[1] != output_dimension:
        raise InvalidModelError(
            "Declared and observed model output dimensions differ.",
            details={"declared": output_dimension, "observed": output.shape[1]},
        )

    input_names = _description(problem.getDescription(), input_dimension, "X")
    output_names = _description(model.getOutputDescription(), output_dimension, "Y")
    inputs: list[VariableMetadata] = []
    for index, name in enumerate(input_names):
        marginal = problem.getMarginal(index)
        try:
            parameters = [float(value) for value in marginal.getParameter()]
        except Exception:
            parameters = []
        inputs.append(
            VariableMetadata(
                index=index,
                name=name,
                distribution=marginal.getImplementation()
                .getClassName()
                .replace("Implementation", ""),
                parameters=parameters,
                mean=float(marginal.getMean()[0]),
                standard_deviation=float(marginal.getStandardDeviation()[0]),
                kind=(
                    "continuous"
                    if marginal.isContinuous()
                    else "discrete"
                    if marginal.isDiscrete()
                    else "mixed"
                ),
            )
        )
    function_type = type(model).__name__
    if function_type == "Function":
        function_type = "PythonFunction"
    symbolic = function_type == "SymbolicFunction"
    copula = problem.getCopula().getImplementation().getClassName()
    dependent_inputs = not problem.hasIndependentCopula()
    output_sample = ot.Sample(output.tolist())
    standard_deviations = output_sample.computeStandardDeviation()
    metadata = ModelMetadata(
        source_hash=_source_hash(source),
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        inputs=inputs,
        outputs=[OutputMetadata(index=i, name=name) for i, name in enumerate(output_names)],
        openturns_version=ot.__version__,
        batch_evaluation_supported=batch_supported,
        validation_sample_size=validation_sample_size,
        validation_runtime_ms=(time.perf_counter() - started) * 1000,
        warnings=["Model output is constant in the validation sample."]
        if all(value <= np.finfo(float).eps for value in standard_deviations)
        else [],
        function_type=function_type,
        exact_gradient_available=symbolic,
        exact_hessian_available=symbolic,
        copula=copula,
        dependent_inputs=dependent_inputs,
    )
    means = output_sample.computeMean()
    minima = output_sample.getMin()
    maxima = output_sample.getMax()
    quantile_05 = output_sample.computeQuantilePerComponent(0.05)
    quantile_95 = output_sample.computeQuantilePerComponent(0.95)
    projected_runtime_ms = evaluation_runtime_ms * 1000 / validation_sample_size
    continuous = sum(item.kind == "continuous" for item in inputs)
    discrete = sum(item.kind == "discrete" for item in inputs)
    smooth_candidate = symbolic and continuous == input_dimension and not dependent_inputs
    ancova_compatible = (
        dependent_inputs and continuous == input_dimension and 2 <= input_dimension <= 10
    )
    expensive = projected_runtime_ms > 5_000
    workflow = recommend_workflow(
        input_dimension=input_dimension,
        projected_1000_evaluation_runtime_ms=projected_runtime_ms,
        surrogate_eligible=(
            (continuous == input_dimension and input_dimension <= 10) or smooth_candidate
        ),
    )
    recommendations = [
        AnalysisRecommendation(
            capability="monte_carlo",
            status="recommended",
            priority=1,
            rationale_codes=["BASELINE_PROPAGATION"],
            projected_evaluations=1000,
            projected_runtime_ms=projected_runtime_ms,
        ),
        AnalysisRecommendation(
            capability="eda",
            status="recommended",
            priority=1,
            rationale_codes=["BASELINE_OUTPUT_CHARACTERISATION"],
            projected_evaluations=1000,
            projected_runtime_ms=projected_runtime_ms,
        ),
        AnalysisRecommendation(
            capability="convergence",
            status="recommended",
            priority=1,
            rationale_codes=["BASELINE_CONVERGENCE_EVIDENCE"],
            projected_evaluations=1000,
            projected_runtime_ms=projected_runtime_ms,
        ),
        AnalysisRecommendation(
            capability="ancova",
            status="recommended" if ancova_compatible else "incompatible",
            priority=2,
            rationale_codes=(
                ["DEPENDENT_INPUT_VARIANCE_DECOMPOSITION"]
                if ancova_compatible
                else ["INDEPENDENT_INPUTS_USE_SOBOL"]
                if not dependent_inputs
                else ["ANCOVA_DIMENSION_LIMIT"]
                if not 2 <= input_dimension <= 10
                else ["ANCOVA_REQUIRES_CONTINUOUS_INPUTS"]
            ),
            projected_evaluations=1500,
            projected_runtime_ms=evaluation_runtime_ms * 1500 / validation_sample_size,
            compatibility_warnings=(
                []
                if ancova_compatible
                else ["ANCOVA requires two to ten continuous inputs with a dependent copula."]
            ),
        ),
        AnalysisRecommendation(
            capability="morris",
            status="recommended" if input_dimension >= 8 else "available",
            priority=2 if input_dimension >= 15 else 3,
            rationale_codes=(
                ["HIGH_DIMENSION_SCREENING"]
                if input_dimension >= 15
                else ["DIMENSION_SCREENING_THRESHOLD"]
                if input_dimension >= 8
                else ["LOW_DIMENSION_SCREENING_OPTIONAL"]
            ),
            projected_evaluations=10 * (input_dimension + 1),
            projected_runtime_ms=evaluation_runtime_ms
            * 10
            * (input_dimension + 1)
            / validation_sample_size,
        ),
        AnalysisRecommendation(
            capability="gpr",
            status="recommended"
            if expensive and continuous == input_dimension and input_dimension <= 10
            else "available"
            if continuous == input_dimension and input_dimension <= 10
            else "incompatible",
            priority=3,
            rationale_codes=["DIRECT_MODEL_RUNTIME_EXCEEDS_FIVE_SECONDS"]
            if expensive
            else ["DIRECT_MODEL_RUNTIME_WITHIN_FIVE_SECONDS"],
            compatibility_warnings=[]
            if continuous == input_dimension and input_dimension <= 10
            else ["GPR baseline eligibility requires at most ten continuous inputs."],
        ),
        AnalysisRecommendation(
            capability="pce",
            status="recommended"
            if expensive and smooth_candidate
            else "available"
            if smooth_candidate
            else "incompatible",
            priority=3,
            rationale_codes=["SYMBOLIC_SMOOTH_CONTINUOUS_MODEL"]
            if smooth_candidate
            else ["PCE_SMOOTH_CONTINUOUS_ELIGIBILITY_NOT_ESTABLISHED"],
            compatibility_warnings=[]
            if smooth_candidate
            else ["PCE requires independent validation for suitable continuous, smooth models."],
        ),
        AnalysisRecommendation(
            capability="reliability",
            status="available",
            priority=4,
            rationale_codes=["USER_DEFINED_FAILURE_EVENT_REQUIRED"],
            compatibility_warnings=[
                "Reliability is never selected without an explicit failure event."
            ],
        ),
        AnalysisRecommendation(
            capability="distribution_fitting",
            status="incompatible",
            priority=4,
            rationale_codes=["NO_EMPIRICAL_DATA_ATTACHED"],
            compatibility_warnings=[
                "Attach empirical data in Data Lab before fitting distributions."
            ],
        ),
    ]
    assessment = ModelAssessment(
        workflow=workflow,
        profile=ModelProfile(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            continuous_marginals=continuous,
            discrete_marginals=discrete,
            copula=copula,
            dependent_inputs=dependent_inputs,
            function_type=function_type,
            batch_support=batch_supported,
            validation_evaluation_runtime_ms=evaluation_runtime_ms,
            projected_1000_evaluation_runtime_ms=projected_runtime_ms,
            pilot_sample_size=validation_sample_size,
            pilot_outputs=[
                PilotOutputSummary(
                    output_index=index,
                    output_name=output_names[index],
                    minimum=float(minima[index]),
                    maximum=float(maxima[index]),
                    mean=float(means[index]),
                    standard_deviation=float(standard_deviations[index]),
                    quantile_05=float(quantile_05[index]),
                    quantile_95=float(quantile_95[index]),
                    variable=float(standard_deviations[index]) > np.finfo(float).eps,
                )
                for index in range(output_dimension)
            ],
        ),
        recommendations=recommendations,
    )
    return ModelRuntime(
        source=source,
        model=model,
        problem=problem,
        metadata=metadata,
        assessment=assessment,
    )


def validate_model_source(
    source: str, *, validation_sample_size: int = 8, seed: int = 42
) -> ModelMetadata:
    return compile_model(source, validation_sample_size=validation_sample_size, seed=seed).metadata
