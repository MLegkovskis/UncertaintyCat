"""Exercise bounded subset sampling through HTTP and the Sandbox CLI protocol."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from services.compute.main import app

SOURCE = """import openturns as ot
model = ot.SymbolicFunction(["R", "S"], ["R-S"])
problem = ot.Normal([7., 2.], [1., 1.], ot.IdentityMatrix(2))
"""


def test_subset_service_auth_execution_and_bound_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNCERTAINTYCAT_INTERNAL_TOKEN", "local-test-token")
    client = TestClient(app)
    request = {
        "source": SOURCE,
        "analysis": {
            "analysis_key": "reliability",
            "config": {"method": "SUBSET_SAMPLING", "operator": "<", "threshold": 0.0},
        },
        "seed": 42,
    }
    assert client.post("/v1/execute", json=request).status_code == 401
    headers = {"Authorization": "Bearer local-test-token"}
    response = client.post("/v1/execute", json=request, headers=headers)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["plugin_version"] == "3.0.0"
    assert result["runtime"]["model_evaluations"] == 8000
    assert result["payload"]["tables"]["subset_levels"]["row_count"] == 4
    json.dumps(result, allow_nan=False)
    request = {
        **request,
        "analysis": {
            "analysis_key": "reliability",
            "config": {
                "method": "SUBSET_SAMPLING",
                "operator": "<",
                "threshold": 0.0,
                "maximum_evaluations": 1000,
                "subset_sample_size": 1000,
            },
        },
    }
    rejected = client.post("/v1/execute", json=request, headers=headers)
    assert rejected.status_code == 422
    assert rejected.json()["error"]["details"]["model_evaluations"] == 1000
    assert "result" not in rejected.json()


def test_subset_cli_retains_bounded_source_free_progress(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "source": SOURCE,
                "analysis": {
                    "analysis_key": "reliability",
                    "config": {"method": "SUBSET_SAMPLING", "operator": "<", "threshold": 0.0},
                },
                "seed": 42,
            }
        )
    )
    completed = subprocess.run(
        [sys.executable, "-m", "services.compute.cli", "execute", str(request_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    response = json.loads(completed.stdout)
    assert response["status"] == 200
    assert response["body"]["result"]["runtime"]["model_evaluations"] == 8000
    events = [
        json.loads(line.removeprefix("UNCERTAINTYCAT_PROGRESS "))
        for line in completed.stderr.splitlines()
        if line.startswith("UNCERTAINTYCAT_PROGRESS ")
    ]
    phases = [event for event in events if event["phase"] == "subset_population"]
    assert len(phases) == 4
    assert all(event["indeterminate"] for event in phases)
    assert "R-S" not in completed.stderr
    assert "SymbolicFunction" not in completed.stderr
    assert len(completed.stdout.encode()) < 20_000
