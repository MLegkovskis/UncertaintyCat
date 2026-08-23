"""Build and serialize promoted OpenTURNS surrogate results."""

from __future__ import annotations

import base64
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import (
    AnalysisRequest,
    ModelAssessment,
    ModelMetadata,
    StrictModel,
)
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime, compile_model
from uncertaintycat_core.plugins.gpr import GprConfig, fit_gpr
from uncertaintycat_core.plugins.pce import PceConfig, fit_pce


class SurrogateSerializationRequest(StrictModel):
    source: str
    method: Literal["pce", "gpr"]
    config: dict[str, Any]
    output_targets: list[int]
    seed: int = 42


class PromotedSurrogateExecutionRequest(StrictModel):
    xml_base64: str
    method: Literal["pce", "gpr"]
    analysis: AnalysisRequest
    metadata: ModelMetadata
    assessment: ModelAssessment
    surrogate_id: str
    surrogate_output_target: int = Field(default=0, ge=0)
    seed: int = 42
    run_id: UUID | None = None


def serialize_surrogate(request: SurrogateSerializationRequest) -> dict[str, Any]:
    """Rebuild an exact validated configuration and serialize it through ot.Study XML."""
    runtime = compile_model(request.source, seed=request.seed)
    analysis = AnalysisRequest(
        analysis_key=request.method,
        config=request.config,
        output_targets=request.output_targets,
    )
    if request.method == "pce":
        pce_config = PceConfig.model_validate(
            {**analysis.config, "seed": request.seed, "output_targets": analysis.output_targets}
        )
        result, _ = fit_pce(runtime, pce_config)
        plugin_version = "2.0.0"
        result_type = "FunctionalChaosResult"
    elif request.method == "gpr":
        gpr_config = GprConfig.model_validate(
            {**analysis.config, "seed": request.seed, "output_targets": analysis.output_targets}
        )
        result, _, _ = fit_gpr(runtime, gpr_config)
        plugin_version = "2.0.0"
        result_type = "GaussianProcessRegressionResult"
    else:  # pragma: no cover - protected by the request contract
        raise IncompatibleAnalysisError("Only PCE and GPR results can be promoted.")

    with tempfile.TemporaryDirectory(prefix="uncertaintycat-surrogate-") as directory:
        path = Path(directory) / "surrogate.xml"
        study = ot.Study()
        study.setStorageManager(ot.XMLStorageManager(str(path)))
        study.add("surrogate_result", result)
        study.add("input_distribution", runtime.problem)
        study.save()
        xml = path.read_bytes()
    return {
        "xmlBase64": base64.b64encode(xml).decode("ascii"),
        "sha256": hashlib.sha256(xml).hexdigest(),
        "sizeBytes": len(xml),
        "resultType": result_type,
        "pluginVersion": plugin_version,
        "openturnsVersion": ot.__version__,
        "sourceModelHash": runtime.metadata.source_hash,
    }


def execute_promoted_surrogate(request: PromotedSurrogateExecutionRequest) -> dict[str, Any]:
    """Load an owner-selected promoted result and execute an analysis against its metamodel."""
    from uncertaintycat_core.runner import run_analysis

    try:
        xml = base64.b64decode(request.xml_base64, validate=True)
    except ValueError as exc:
        raise IncompatibleAnalysisError("The promoted surrogate artifact is invalid.") from exc
    with tempfile.TemporaryDirectory(prefix="uncertaintycat-surrogate-run-") as directory:
        path = Path(directory) / "surrogate.xml"
        path.write_bytes(xml)
        study = ot.Study()
        study.setStorageManager(ot.XMLStorageManager(str(path)))
        study.load()
        distribution = ot.Distribution()
        study.fillObject("input_distribution", distribution)
        if request.method == "pce":
            persisted_result: ot.PersistentObject = ot.FunctionalChaosResult()
        else:
            persisted_result = ot.GaussianProcessRegressionResult()
        study.fillObject("surrogate_result", persisted_result)
    metamodel = persisted_result.getMetaModel()
    if request.surrogate_output_target >= request.metadata.output_dimension:
        raise IncompatibleAnalysisError(
            "The promoted surrogate output target is outside the source model metadata."
        )
    source_output = request.metadata.outputs[request.surrogate_output_target]
    metadata = request.metadata.model_copy(
        update={
            "output_dimension": 1,
            "outputs": [source_output.model_copy(update={"index": 0})],
        }
    )
    pilot = next(
        (
            item.model_copy(update={"output_index": 0})
            for item in request.assessment.profile.pilot_outputs
            if item.output_index == request.surrogate_output_target
        ),
        None,
    )
    profile = request.assessment.profile.model_copy(
        update={
            "output_dimension": 1,
            "pilot_outputs": [pilot] if pilot is not None else [],
        }
    )
    assessment = request.assessment.model_copy(update={"profile": profile})
    runtime = ModelRuntime(
        source=f"promoted-surrogate:{request.surrogate_id}",
        model=metamodel,
        problem=distribution,
        metadata=metadata,
        assessment=assessment,
    )
    result = run_analysis(
        runtime,
        request.analysis,
        seed=request.seed,
        run_id=request.run_id,
    )
    result.payload.facts["evidence_source"] = "promoted_surrogate"
    result.payload.facts["surrogate_id"] = request.surrogate_id
    result.payload.facts["surrogate_method"] = request.method
    result.payload.facts["source_output_index"] = request.surrogate_output_target
    result.payload.facts["source_output_name"] = source_output.name
    return {"result": result.model_dump(mode="json")}
