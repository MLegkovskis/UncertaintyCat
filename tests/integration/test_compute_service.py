from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from services.compute.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_validate_and_execute() -> None:
    source = Path("examples/Ishigami.py").read_text()
    validation = client.post("/v1/validate", json={"source": source})
    assert validation.status_code == 200
    assert validation.json()["metadata"]["input_dimension"] == 3

    execution = client.post(
        "/v1/execute",
        json={
            "source": source,
            "analysis": {
                "analysis_key": "monte_carlo",
                "config": {"sample_size": 30, "inline_sample_limit": 5},
            },
            "seed": 42,
        },
    )
    assert execution.status_code == 200
    assert execution.json()["result"]["status"] == "succeeded"

    gpr_execution = client.post(
        "/v1/execute",
        json={
            "source": source,
            "analysis": {
                "analysis_key": "gpr",
                "config": {"training_size": 32, "validation_size": 20},
                "output_targets": [0],
            },
            "seed": 42,
        },
    )
    assert gpr_execution.status_code == 200
    assert gpr_execution.json()["result"]["analysis_key"] == "gpr"
    assert gpr_execution.json()["result"]["runtime"]["model_evaluations"] == 52


def test_public_error_is_structured() -> None:
    response = client.post(
        "/v1/validate",
        json={"source": "import os\nmodel = None\nproblem = None"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsafe_model"


def test_data_lab_and_promoted_surrogate_http_contracts() -> None:
    csv = "x,y\n1,10\n2,13\n3,11\n4,20\n5,16\n6,31\n7,25\n8,38\n"
    content = base64.b64encode(csv.encode()).decode()
    inspected = client.post(
        "/v1/data/inspect",
        json={"content_base64": content, "source_kind": "paste"},
    )
    assert inspected.status_code == 200
    assert inspected.json()["dataset"]["rowCount"] == 8
    fitted = client.post(
        "/v1/data/fit",
        json={
            "content_base64": content,
            "source_kind": "paste",
            "selected_columns": ["x", "y"],
            "candidates": ["Normal", "Uniform"],
            "selected_marginals": {"x": "Normal", "y": "Normal"},
            "copula": "normal",
        },
    )
    assert fitted.status_code == 200
    assert fitted.json()["fit"]["copula"]["className"] == "NormalCopula"

    source = Path("examples/Ishigami.py").read_text()
    validation = client.post("/v1/validate", json={"source": source}).json()
    serialized = client.post(
        "/v1/surrogates/serialize",
        json={
            "source": source,
            "method": "pce",
            "config": {"degree": 2, "training_size": 48, "validation_size": 20},
            "output_targets": [0],
            "seed": 17,
        },
    )
    assert serialized.status_code == 200
    artifact = serialized.json()["surrogate"]
    executed = client.post(
        "/v1/surrogates/execute",
        json={
            "xml_base64": artifact["xmlBase64"],
            "method": "pce",
            "analysis": {
                "analysis_key": "monte_carlo",
                "config": {"sample_size": 20},
            },
            "metadata": validation["metadata"],
            "assessment": validation["assessment"],
            "surrogate_id": "integration-surrogate",
            "seed": 18,
        },
    )
    assert executed.status_code == 200
    assert executed.json()["result"]["payload"]["facts"]["evidence_source"] == (
        "promoted_surrogate"
    )


def test_one_shot_sandbox_protocol() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "services.compute.cli", "catalog"],
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(completed.stdout)
    assert envelope["status"] == 200
    assert {entry["key"] for entry in envelope["body"]} >= {
        "monte_carlo",
        "sobol",
        "gpr",
    }
