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

    target_hsic_execution = client.post(
        "/v1/execute",
        json={
            "source": source,
            "analysis": {
                "analysis_key": "target_hsic",
                "config": {
                    "sample_size": 100,
                    "permutations": 20,
                    "threshold": 0.0,
                    "operator": ">=",
                },
                "output_targets": [0],
            },
            "seed": 0,
        },
    )
    assert target_hsic_execution.status_code == 200
    target_hsic_result = target_hsic_execution.json()["result"]
    assert target_hsic_result["analysis_key"] == "target_hsic"
    assert target_hsic_result["runtime"]["model_evaluations"] == 100
    assert target_hsic_result["payload"]["tables"]["target_indices"]["row_count"] == 3

    dependent_source = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2^2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.4
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
    ancova_execution = client.post(
        "/v1/execute",
        json={
            "source": dependent_source,
            "analysis": {
                "analysis_key": "ancova",
                "config": {
                    "degree": 3,
                    "training_size": 128,
                    "validation_size": 64,
                    "ancova_sample_size": 256,
                },
                "output_targets": [0],
            },
            "seed": 42,
        },
    )
    assert ancova_execution.status_code == 200
    ancova_result = ancova_execution.json()["result"]
    assert ancova_result["analysis_key"] == "ancova"
    assert ancova_result["runtime"]["model_evaluations"] == 192
    assert ancova_result["payload"]["metrics"]["validation_q2"] > 0.99

    calibration_execution = client.post(
        "/v1/execute",
        json={
            "source": Path("examples/Calibration_Exponential.py").read_text(),
            "analysis": {
                "analysis_key": "calibration_nlls",
                "config": {
                    "parameter_indices": [0, 1, 2],
                    "starting_values": [1.0, 1.0, 1.0],
                    "observed_input_names": ["x"],
                    "observed_output_name": "y",
                    "observed_inputs": [[0.5 + index] for index in range(10)],
                    "observed_outputs": [
                        4.3712405825862275,
                        5.2770913648243774,
                        6.9664982679561884,
                        9.765797121248307,
                        14.076213741899407,
                        21.588660365352318,
                        33.73065754817239,
                        53.89716086558238,
                        86.9670282151489,
                        141.5407992331982,
                    ],
                    "maximum_calls": 250,
                },
                "output_targets": [0],
            },
            "seed": 0,
        },
    )
    assert calibration_execution.status_code == 200
    calibration_result = calibration_execution.json()["result"]
    assert calibration_result["analysis_key"] == "calibration_nlls"
    assert calibration_result["runtime"]["model_evaluations"] > 0
    assert calibration_result["payload"]["metrics"]["observations"] == 10
    assert calibration_result["payload"]["metrics"]["rmse_after"] < 0.05
    assert (
        calibration_result["payload"]["tables"]["observations_and_predictions"]["row_count"] == 10
    )


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

    surrogate_rows = ["x1,x2,response"]
    for index in range(40):
        x1 = -2.0 + 4.0 * index / 39.0
        x2 = ((index * 7) % 17) / 4.0 - 2.0
        surrogate_rows.append(f"{x1},{x2},{x1 * x1 + 0.3 * x2}")
    surrogate_content = base64.b64encode("\n".join(surrogate_rows).encode()).decode()
    data_surrogate = client.post(
        "/v1/data/surrogate",
        json={
            "content_base64": surrogate_content,
            "source_kind": "paste",
            "input_columns": ["x1", "x2"],
            "output_column": "response",
            "validation_fraction": 0.2,
            "kernel": "MATERN_2_5",
            "trend": "CONSTANT",
            "seed": 42,
        },
    )
    assert data_surrogate.status_code == 200
    data_surrogate_body = data_surrogate.json()["surrogate"]
    assert data_surrogate_body["validation"]["r2"] > 0.95
    assert data_surrogate_body["artifact"]["resultType"] == ("GaussianProcessRegressionResult")

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
        "ancova",
        "monte_carlo",
        "sobol",
        "gpr",
        "calibration_nlls",
        "target_hsic",
    }
