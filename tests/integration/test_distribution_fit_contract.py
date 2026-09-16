from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from services.compute.main import app


def test_seeded_distribution_fit_has_http_and_isolated_cli_parity(tmp_path: Path) -> None:
    csv = "x,y\n1,10\n2,13\n3,11\n4,20\n5,16\n6,31\n7,25\n8,38\n"
    request = {
        "content_base64": base64.b64encode(csv.encode()).decode(),
        "source_kind": "paste",
        "selected_columns": ["x", "y"],
        "candidates": ["Normal", "Uniform"],
        "selected_marginals": {"x": "Normal", "y": "Uniform"},
        "copula": "normal",
        "seed": 73,
    }
    client = TestClient(app)
    response = client.post("/v1/data/fit", json=request)
    repeated = client.post("/v1/data/fit", json=request)
    assert response.status_code == repeated.status_code == 200
    assert response.json() == repeated.json()
    assert response.json()["fit"]["seed"] == 73
    assert response.json()["fit"]["fittingVersion"] == "1.1.0"
    request_path = tmp_path / "fit-request.json"
    request_path.write_text(json.dumps(request))
    completed = subprocess.run(
        [sys.executable, "-m", "services.compute.cli", "fit-data", str(request_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(completed.stdout)
    assert envelope["status"] == 200
    assert envelope["body"] == response.json()
    json.dumps(envelope, allow_nan=False)
    invalid = client.post("/v1/data/fit", json={**request, "seed": 2_147_483_648})
    assert invalid.status_code == 422
