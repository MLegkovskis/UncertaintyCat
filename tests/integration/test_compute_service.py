from __future__ import annotations

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


def test_public_error_is_structured() -> None:
    response = client.post(
        "/v1/validate",
        json={"source": "import os\nmodel = None\nproblem = None"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsafe_model"


def test_one_shot_sandbox_protocol() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "services.compute.cli", "catalog"],
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(completed.stdout)
    assert envelope["status"] == 200
    assert {entry["key"] for entry in envelope["body"]} >= {"monte_carlo", "sobol"}
