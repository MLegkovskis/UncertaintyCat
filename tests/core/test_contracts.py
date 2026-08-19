from __future__ import annotations

import json
from pathlib import Path

from uncertaintycat_core.contracts import AnalysisRequest
from uncertaintycat_core.model import compile_model
from uncertaintycat_core.runner import run_analysis


def test_result_envelope_is_strict_json() -> None:
    runtime = compile_model(Path("examples/Beam.py").read_text())
    result = run_analysis(
        runtime,
        AnalysisRequest(
            analysis_key="monte_carlo",
            config={"sample_size": 20, "inline_sample_limit": 5},
        ),
    )
    encoded = result.model_dump_json()
    decoded = json.loads(encoded)
    assert decoded["analysis_key"] == "monte_carlo"
    assert decoded["payload"]["tables"]["samples"]["row_count"] == 20
    assert "Figure" not in encoded
    assert "DataFrame" not in encoded
