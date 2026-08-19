"""Compilation, preflight checks, and validated access to OpenTURNS models."""

from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import openturns as ot

from uncertaintycat_core.contracts import ModelMetadata, OutputMetadata, VariableMetadata
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
    batch_supported = True
    try:
        output = _as_2d_output(model(sample), validation_sample_size)
    except Exception:
        batch_supported = False
        try:
            output = np.asarray([_as_2d_output(model(point), 1)[0] for point in sample])
        except Exception as exc:
            raise InvalidModelError(f"Model failed its validation evaluations: {exc}") from exc

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
            )
        )
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
        if np.all(np.std(output, axis=0) <= np.finfo(float).eps)
        else [],
    )
    return ModelRuntime(source=source, model=model, problem=problem, metadata=metadata)


def validate_model_source(
    source: str, *, validation_sample_size: int = 8, seed: int = 42
) -> ModelMetadata:
    return compile_model(source, validation_sample_size=validation_sample_size, seed=seed).metadata
