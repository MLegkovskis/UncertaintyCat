"""HTTP boundary for isolated OpenTURNS model validation and analysis execution."""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import Field

from uncertaintycat_core import analysis_catalog, compile_model, run_analysis
from uncertaintycat_core.contracts import AnalysisRequest, StrictModel
from uncertaintycat_core.data_lab import (
    DatasetContent,
    DistributionFitRequest,
    fit_distributions,
    inspect_dataset,
)
from uncertaintycat_core.errors import UncertaintyCatError
from uncertaintycat_core.surrogate import (
    PromotedSurrogateExecutionRequest,
    SurrogateSerializationRequest,
    execute_promoted_surrogate,
    serialize_surrogate,
)


class ValidationRequest(StrictModel):
    source: str
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    validation_sample_size: int = Field(default=8, ge=1, le=100)


class ExecuteRequest(StrictModel):
    source: str
    analysis: AnalysisRequest
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    run_id: UUID | None = None


def require_internal_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = os.getenv("UNCERTAINTYCAT_INTERNAL_TOKEN")
    if not expected:
        return
    supplied = authorization.removeprefix("Bearer ") if authorization else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid internal service token")


app = FastAPI(
    title="UncertaintyCat Compute Service",
    version="0.2.0",
    docs_url=None if os.getenv("UNCERTAINTYCAT_DISABLE_DOCS") == "1" else "/docs",
)


@app.exception_handler(UncertaintyCatError)
async def uncertaintycat_error_handler(_request: Any, exc: UncertaintyCatError) -> JSONResponse:
    status = 422 if exc.code in {"invalid_model", "unsafe_model", "incompatible_analysis"} else 400
    return JSONResponse(
        status_code=status,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "uncertaintycat-compute"}


@app.get("/v1/catalog", dependencies=[Depends(require_internal_token)])
def catalog() -> list[dict[str, Any]]:
    return [entry.model_dump(mode="json") for entry in analysis_catalog()]


@app.post("/v1/validate", dependencies=[Depends(require_internal_token)])
def validate_model(request: ValidationRequest) -> dict[str, Any]:
    runtime = compile_model(
        request.source,
        validation_sample_size=request.validation_sample_size,
        seed=request.seed,
    )
    return {
        "metadata": runtime.metadata.model_dump(mode="json"),
        "assessment": runtime.assessment.model_dump(mode="json"),
    }


@app.post("/v1/execute", dependencies=[Depends(require_internal_token)])
def execute(request: ExecuteRequest) -> dict[str, Any]:
    runtime = compile_model(request.source, seed=request.seed)
    result = run_analysis(
        runtime,
        request.analysis,
        seed=request.seed,
        run_id=request.run_id,
    )
    return {"result": result.model_dump(mode="json")}


@app.post("/v1/data/inspect", dependencies=[Depends(require_internal_token)])
def inspect_data(request: DatasetContent) -> dict[str, Any]:
    return {"dataset": inspect_dataset(request)}


@app.post("/v1/data/fit", dependencies=[Depends(require_internal_token)])
def fit_data(request: DistributionFitRequest) -> dict[str, Any]:
    return {"fit": fit_distributions(request)}


@app.post("/v1/surrogates/serialize", dependencies=[Depends(require_internal_token)])
def serialize_surrogate_result(request: SurrogateSerializationRequest) -> dict[str, Any]:
    return {"surrogate": serialize_surrogate(request)}


@app.post("/v1/surrogates/execute", dependencies=[Depends(require_internal_token)])
def execute_surrogate_result(request: PromotedSurrogateExecutionRequest) -> dict[str, Any]:
    return execute_promoted_surrogate(request)
