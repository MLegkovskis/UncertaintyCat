"""Analysis execution and provenance envelope construction."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import UUID, uuid4

import openturns as ot

import uncertaintycat_core
from uncertaintycat_core.catalog import get_plugin
from uncertaintycat_core.contracts import (
    AnalysisRequest,
    AnalysisResultEnvelope,
    RunRequest,
    RuntimeMetrics,
)
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime, compile_model
from uncertaintycat_core.progress import ProgressCallback, progress_scope, report_progress

SuiteProgressCallback = Callable[[str, int, int], None]


def run_analysis(
    runtime: ModelRuntime,
    request: AnalysisRequest,
    *,
    seed: int = 42,
    run_id: UUID | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AnalysisResultEnvelope:
    plugin = get_plugin(request.analysis_key)
    if request.plugin_version and request.plugin_version != plugin.version:
        raise IncompatibleAnalysisError(
            f"Requested {plugin.key} version {request.plugin_version}, "
            f"but {plugin.version} is installed."
        )
    config = plugin.parse_config(request.config, seed=seed, output_targets=request.output_targets)
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    with progress_scope(progress_callback):
        report_progress("applicability", 12, "Checking the model against the analysis contract.")
        warnings = plugin.applicability_warnings(runtime, config)
        report_progress("openturns", 18, f"OpenTURNS is running {plugin.name}.", indeterminate=True)
        payload, evaluations = plugin.run(runtime, config)
        report_progress("serializing", 95, "Serializing deterministic numerical evidence.")
    duration_ms = (time.perf_counter() - started) * 1000
    return AnalysisResultEnvelope(
        run_id=run_id or uuid4(),
        analysis_key=plugin.key,
        plugin_version=plugin.version,
        result_schema_version=plugin.result_schema_version,
        model_hash=runtime.metadata.source_hash,
        seed=seed,
        uq_core_version=uncertaintycat_core.__version__,
        openturns_version=ot.__version__,
        started_at=started_at,
        runtime=RuntimeMetrics(duration_ms=duration_ms, model_evaluations=evaluations),
        warnings=runtime.metadata.warnings + warnings,
        assumptions=list(plugin.assumptions),
        payload=payload,
    )


def run_suite(
    request: RunRequest, progress_callback: SuiteProgressCallback | None = None
) -> list[AnalysisResultEnvelope]:
    if request.model_source is None:
        raise ValueError("The local runner requires model_source.")
    runtime = compile_model(request.model_source, seed=request.seed)
    run_id = uuid4()
    results: list[AnalysisResultEnvelope] = []
    for index, analysis in enumerate(request.analyses, start=1):
        if progress_callback:
            progress_callback(analysis.analysis_key, index, len(request.analyses))
        results.append(
            run_analysis(runtime, analysis, seed=request.seed + index - 1, run_id=run_id)
        )
    return results
