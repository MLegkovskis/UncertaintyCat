from __future__ import annotations

import math
from pathlib import Path

import pytest

from uncertaintycat_core.contracts import AnalysisRequest
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import compile_model
from uncertaintycat_core.runner import run_analysis

MULTI_OUTPUT_SOURCE = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2", "x1 * x2"])
model.setOutputDescription(["sum", "product"])
problem = ot.JointDistribution([ot.Normal(), ot.Uniform(-1.0, 1.0)])
problem.setDescription(["normal_input", "uniform_input"])
"""


def test_monte_carlo_is_reproducible_and_multi_output() -> None:
    runtime_a = compile_model(MULTI_OUTPUT_SOURCE)
    runtime_b = compile_model(MULTI_OUTPUT_SOURCE)
    request = AnalysisRequest(
        analysis_key="monte_carlo",
        config={"sample_size": 200, "inline_sample_limit": 20},
        output_targets=[0, 1],
    )
    result_a = run_analysis(runtime_a, request, seed=123)
    result_b = run_analysis(runtime_b, request, seed=123)
    assert result_a.payload.metrics["sum.mean"] == result_b.payload.metrics["sum.mean"]
    assert result_a.payload.tables["summary"].row_count == 2
    assert result_a.payload.tables["samples"].truncated is True
    assert len(result_a.payload.tables["samples"].rows) == 20


def test_eda_reuses_named_outputs_and_returns_correlation_matrices() -> None:
    runtime = compile_model(MULTI_OUTPUT_SOURCE)
    result = run_analysis(
        runtime,
        AnalysisRequest(analysis_key="eda", config={"sample_size": 300}),
        seed=7,
    )
    assert result.payload.matrices["pearson"].row_labels == ["sum", "product"]
    assert result.payload.matrices["pearson"].column_labels == [
        "normal_input",
        "uniform_input",
    ]
    assert result.payload.tables["summary"].row_count == 2


@pytest.mark.scientific
def test_ishigami_sobol_matches_known_first_order_structure() -> None:
    runtime = compile_model(Path("examples/Ishigami.py").read_text())
    result = run_analysis(
        runtime,
        AnalysisRequest(
            analysis_key="sobol",
            config={"base_sample_size": 2048, "second_order": True},
            output_targets=[0],
        ),
        seed=42,
    )
    rows = result.payload.tables["indices"].rows
    first_order = [float(row[1]) for row in rows]
    assert math.isclose(first_order[0], 0.314, abs_tol=0.12)
    assert math.isclose(first_order[1], 0.442, abs_tol=0.12)
    assert abs(first_order[2]) < 0.12


def test_sobol_rejects_dependent_inputs() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.5
problem = ot.JointDistribution([ot.Normal(), ot.Normal()], ot.NormalCopula(correlation))
"""
    runtime = compile_model(source)
    with pytest.raises(IncompatibleAnalysisError, match="independent"):
        run_analysis(runtime, AnalysisRequest(analysis_key="sobol"))


@pytest.mark.parametrize(
    ("analysis_key", "config"),
    [
        ("correlation", {"sample_size": 100}),
        ("fast", {"sample_size": 128}),
        ("hsic", {"sample_size": 50, "permutations": 5}),
        ("taylor", {"validation_size": 50}),
        ("morris", {"trajectories": 4, "levels": 4}),
        ("convergence", {"sample_size": 50, "max_points": 30}),
        (
            "reliability",
            {"method": "MONTE_CARLO", "threshold": 0.0, "operator": ">", "sample_size": 100},
        ),
        ("pce", {"degree": 2, "training_size": 100, "validation_size": 50}),
    ],
)
def test_extended_catalog_plugins_return_strict_results(
    analysis_key: str, config: dict[str, object]
) -> None:
    runtime = compile_model(Path("examples/Ishigami.py").read_text())
    result = run_analysis(
        runtime,
        AnalysisRequest(analysis_key=analysis_key, config=config, output_targets=[0]),
    )
    assert result.status == "succeeded"
    assert result.runtime.model_evaluations > 0
    assert "NaN" not in result.model_dump_json()


def test_form_probability_for_standard_normal_half_space() -> None:
    runtime = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["X"], ["X"])
problem = ot.Normal(0.0, 1.0)
problem.setDescription(["X"])
"""
    )
    result = run_analysis(
        runtime,
        AnalysisRequest(
            analysis_key="reliability",
            config={"method": "FORM", "threshold": 0.0, "operator": ">"},
            output_targets=[0],
        ),
    )
    assert result.payload.metrics["event_probability"] == pytest.approx(0.5, abs=1e-3)
