from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import openturns as ot
import pytest

from uncertaintycat_core.contracts import AnalysisRequest
from uncertaintycat_core.model import compile_model
from uncertaintycat_core.surrogate import (
    PromotedSurrogateExecutionRequest,
    SurrogateSerializationRequest,
    execute_promoted_surrogate,
    serialize_surrogate,
)


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("method", "config", "target_type"),
    [
        (
            "pce",
            {"degree": 2, "training_size": 60, "validation_size": 20},
            ot.FunctionalChaosResult,
        ),
        ("gpr", {"training_size": 32, "validation_size": 20}, ot.GaussianProcessRegressionResult),
    ],
)
def test_surrogate_study_xml_round_trip_and_explicit_execution(
    method: str, config: dict[str, object], target_type: type
) -> None:
    source = Path("examples/Ishigami.py").read_text()
    runtime = compile_model(source)
    serialized = serialize_surrogate(
        SurrogateSerializationRequest(
            source=source,
            method=method,  # type: ignore[arg-type]
            config=config,
            output_targets=[0],
            seed=7,
        )
    )
    assert serialized["pluginVersion"] == "2.0.0"
    assert serialized["sourceModelHash"] == runtime.metadata.source_hash
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "surrogate.xml"
        path.write_bytes(base64.b64decode(serialized["xmlBase64"]))
        study = ot.Study()
        study.setStorageManager(ot.XMLStorageManager(str(path)))
        study.load()
        restored = target_type()
        study.fillObject("surrogate_result", restored)
        assert restored.getMetaModel().getInputDimension() == 3

    output = execute_promoted_surrogate(
        PromotedSurrogateExecutionRequest(
            xml_base64=serialized["xmlBase64"],
            method=method,  # type: ignore[arg-type]
            analysis=AnalysisRequest(analysis_key="monte_carlo", config={"sample_size": 40}),
            metadata=runtime.metadata,
            assessment=runtime.assessment,
            surrogate_id="surrogate-under-test",
            seed=9,
        )
    )["result"]
    assert output["runtime"]["model_evaluations"] == 40
    assert output["payload"]["facts"]["evidence_source"] == "promoted_surrogate"


@pytest.mark.scientific
def test_promoted_surrogate_preserves_selected_multi_output_provenance() -> None:
    source = """import openturns as ot
problem = ot.Normal(1)
problem.setDescription([\"x\"])
model = ot.SymbolicFunction([\"x\"], [\"x\", \"x^2\"])
model.setOutputDescription([\"linear\", \"squared\"])
"""
    runtime = compile_model(source)
    serialized = serialize_surrogate(
        SurrogateSerializationRequest(
            source=source,
            method="pce",
            config={"degree": 2, "training_size": 50, "validation_size": 20},
            output_targets=[1],
            seed=7,
        )
    )
    output = execute_promoted_surrogate(
        PromotedSurrogateExecutionRequest(
            xml_base64=serialized["xmlBase64"],
            method="pce",
            analysis=AnalysisRequest(
                analysis_key="monte_carlo",
                config={"sample_size": 40},
                output_targets=[0],
            ),
            metadata=runtime.metadata,
            assessment=runtime.assessment,
            surrogate_id="multi-output-surrogate",
            surrogate_output_target=1,
            seed=9,
        )
    )["result"]
    assert output["payload"]["facts"]["source_output_index"] == 1
    assert output["payload"]["facts"]["source_output_name"] == "squared"
    assert output["payload"]["tables"]["summary"]["rows"][0][0] == "squared"
