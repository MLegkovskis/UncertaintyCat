"""One-shot JSON protocol used inside a disposable Cloudflare Sandbox."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from services.compute.main import ExecuteRequest, ValidationRequest
from uncertaintycat_core import analysis_catalog, compile_model, run_analysis
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


def _response(status: int, body: dict[str, Any] | list[dict[str, Any]]) -> None:
    print(json.dumps({"status": status, "body": body}, allow_nan=False, separators=(",", ":")))


def _payload(path: str | None) -> Any:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    operation = sys.argv[1] if len(sys.argv) > 1 else ""
    path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        if operation == "catalog":
            _response(200, [entry.model_dump(mode="json") for entry in analysis_catalog()])
            return 0
        if operation == "validate":
            validation_request = ValidationRequest.model_validate(_payload(path))
            runtime = compile_model(
                validation_request.source,
                validation_sample_size=validation_request.validation_sample_size,
                seed=validation_request.seed,
            )
            _response(
                200,
                {
                    "metadata": runtime.metadata.model_dump(mode="json"),
                    "assessment": runtime.assessment.model_dump(mode="json"),
                },
            )
            return 0
        if operation == "execute":
            execute_request = ExecuteRequest.model_validate(_payload(path))
            runtime = compile_model(execute_request.source, seed=execute_request.seed)
            result = run_analysis(
                runtime,
                execute_request.analysis,
                seed=execute_request.seed,
                run_id=execute_request.run_id,
            )
            _response(200, {"result": result.model_dump(mode="json")})
            return 0
        if operation == "inspect-data":
            request = DatasetContent.model_validate(_payload(path))
            _response(200, {"dataset": inspect_dataset(request)})
            return 0
        if operation == "fit-data":
            request = DistributionFitRequest.model_validate(_payload(path))
            _response(200, {"fit": fit_distributions(request)})
            return 0
        if operation == "serialize-surrogate":
            surrogate_request = SurrogateSerializationRequest.model_validate(_payload(path))
            _response(200, {"surrogate": serialize_surrogate(surrogate_request)})
            return 0
        if operation == "execute-surrogate":
            execution_request = PromotedSurrogateExecutionRequest.model_validate(_payload(path))
            _response(200, execute_promoted_surrogate(execution_request))
            return 0
        _response(
            400, {"error": {"code": "invalid_operation", "message": "Unknown compute operation."}}
        )
    except (ValidationError, json.JSONDecodeError) as exc:
        _response(422, {"error": {"code": "invalid_request", "message": str(exc)}})
    except UncertaintyCatError as exc:
        status = (
            422 if exc.code in {"invalid_model", "unsafe_model", "incompatible_analysis"} else 400
        )
        _response(
            status,
            {"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )
    except Exception:
        _response(
            500,
            {
                "error": {
                    "code": "compute_internal_error",
                    "message": "The isolated compute process failed unexpectedly.",
                }
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
