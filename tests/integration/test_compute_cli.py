from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.compute.main import app
from uncertaintycat_core.model import compile_model
from uncertaintycat_core.surrogate import SurrogateSerializationRequest, serialize_surrogate


@pytest.mark.parametrize("method", ["pce", "gpr"])
def test_sandbox_cli_executes_promoted_surrogates_with_http_parity(
    tmp_path: Path, method: str
) -> None:
    source = """import openturns as ot
model = ot.SymbolicFunction(["private_input"], ["private_input", "private_input^2"])
model.setOutputDescription(["linear", "squared"])
problem = ot.Uniform(-1.0, 1.0)
problem.setDescription(["private_input"])
"""
    runtime = compile_model(source)
    serialized = serialize_surrogate(
        SurrogateSerializationRequest(
            source=source,
            method=method,  # type: ignore[arg-type]
            config={"training_size": 32, "validation_size": 20},
            output_targets=[1],
            seed=42,
        )
    )
    request = {
        "xml_base64": serialized["xmlBase64"],
        "method": method,
        "metadata": runtime.metadata.model_dump(mode="json"),
        "assessment": runtime.assessment.model_dump(mode="json"),
        "surrogate_id": "retained-surrogate-under-test",
        "surrogate_output_target": 1,
        "analysis": {
            "analysis_key": "monte_carlo",
            "config": {"sample_size": 40},
            "output_targets": [0],
        },
        "seed": 42,
    }
    request_path = tmp_path / "surrogate-request.json"
    request_path.write_text(json.dumps(request))
    completed = subprocess.run(
        [sys.executable, "-m", "services.compute.cli", "execute-surrogate", str(request_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(completed.stdout)
    assert envelope["status"] == 200
    result = envelope["body"]["result"]
    assert result["runtime"]["model_evaluations"] == 40
    assert result["model_hash"] == runtime.metadata.source_hash
    assert result["payload"]["facts"]["evidence_source"] == "promoted_surrogate"
    assert result["payload"]["facts"]["source_output_index"] == 1
    assert result["payload"]["facts"]["source_output_name"] == "squared"
    assert "surrogate_loading" in completed.stderr
    assert "private_input" not in completed.stderr
    assert "xml_base64" not in completed.stderr
    response = TestClient(app).post("/v1/surrogates/execute", json=request)
    assert response.status_code == 200
    assert response.json()["result"]["payload"] == result["payload"]


@pytest.mark.parametrize("operation", ["execute", "execute-surrogate", "validate"])
def test_compute_schema_errors_never_include_private_request_values(
    tmp_path: Path, operation: str
) -> None:
    marker = "PRIVATE_MODEL_SOURCE_OR_SERIALIZED_ARTIFACT"
    request = {"source": {"definition": marker}, "xml_base64": marker, marker: marker}
    request_path = tmp_path / "malformed-request.json"
    request_path.write_text(json.dumps(request))
    completed = subprocess.run(
        [sys.executable, "-m", "services.compute.cli", operation, str(request_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(completed.stdout)
    assert envelope["status"] == 422
    assert envelope["body"]["error"]["code"] == "invalid_request"
    assert marker not in completed.stdout + completed.stderr
    assert "input_value" not in completed.stdout
    path = "/v1/surrogates/execute" if operation == "execute-surrogate" else f"/v1/{operation}"
    response = TestClient(app).post(path, json=request)
    assert response.status_code == 422
    assert response.json() == envelope["body"]
    assert marker not in response.text


def test_sandbox_cli_streams_hsic_phases_without_corrupting_result_envelope(
    tmp_path: Path,
) -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["private_x"], ["private_x * private_x"])
problem = ot.Uniform(-1.0, 1.0)
"""
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "source": source,
                "seed": 42,
                "analysis": {
                    "analysis_key": "hsic",
                    "config": {"sample_size": 50, "permutations": 5},
                    "output_targets": [0],
                },
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.compute.cli",
            "execute",
            str(request_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(completed.stdout)
    events = [
        json.loads(line.removeprefix("UNCERTAINTYCAT_PROGRESS "))
        for line in completed.stderr.splitlines()
        if line.startswith("UNCERTAINTYCAT_PROGRESS ")
    ]

    assert envelope["status"] == 200
    assert envelope["body"]["result"]["analysis_key"] == "hsic"
    assert [event["phase"] for event in events] == [
        "model_loading",
        "applicability",
        "openturns",
        "sampling",
        "kernel_construction",
        "observed_indices",
        "permutation_inference",
        "ranking",
        "serializing",
    ]
    assert [event["percent"] for event in events] == sorted(event["percent"] for event in events)
    assert (
        next(event for event in events if event["phase"] == "permutation_inference")[
            "indeterminate"
        ]
        is True
    )
    assert "private_x" not in completed.stderr
    assert "SymbolicFunction" not in completed.stderr
