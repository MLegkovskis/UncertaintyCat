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
    assert result_a.payload.facts["sampling_design"] == "Monte Carlo"


def test_morris_handles_unbounded_borehole_marginals_without_endpoint_nan() -> None:
    runtime = compile_model(Path("examples/Borehole.py").read_text())
    result = run_analysis(
        runtime,
        AnalysisRequest(
            analysis_key="morris",
            config={"trajectories": 4, "levels": 6},
            output_targets=[0],
        ),
        seed=42,
    )
    assert result.plugin_version == "2.1.0"
    assert result.payload.metrics["tail_probability"] == 1.0e-6
    assert "NaN" not in result.model_dump_json()


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
        (
            "gpr",
            {
                "training_size": 64,
                "validation_size": 50,
                "kernel": "MATERN_2_5",
                "trend": "CONSTANT",
            },
        ),
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


@pytest.mark.scientific
def test_gpr_validates_a_smooth_response_and_is_reproducible() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["sin(x1) + 0.5 * x2^2"])
model.setOutputDescription(["smooth_response"])
problem = ot.JointDistribution([ot.Uniform(-2.0, 2.0), ot.Normal(0.0, 0.7)])
problem.setDescription(["x1", "x2"])
"""
    request = AnalysisRequest(
        analysis_key="gpr",
        config={
            "training_size": 64,
            "validation_size": 128,
            "kernel": "MATERN_2_5",
            "trend": "CONSTANT",
        },
        output_targets=[0],
    )
    result_a = run_analysis(compile_model(source), request, seed=19)
    result_b = run_analysis(compile_model(source), request, seed=19)

    assert result_a.payload.metrics == result_b.payload.metrics
    assert result_a.payload.metrics["validation_r2"] > 0.99
    assert result_a.payload.metrics["validation_rmse"] < 0.03
    assert 0.8 <= result_a.payload.metrics["nominal_95_interval_coverage"] <= 1.0
    assert result_a.runtime.model_evaluations == 192
    assert result_a.payload.tables["validation_predictions"].row_count == 128
    assert result_a.payload.tables["kernel_length_scales"].row_count == 2
    assert "NaN" not in result_a.model_dump_json()


def test_gpr_supports_dependent_continuous_inputs() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["2*x1 - 0.25*x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.6
problem = ot.JointDistribution(
    [ot.Normal(), ot.Normal()], ot.NormalCopula(correlation)
)
problem.setDescription(["x1", "x2"])
"""
    result = run_analysis(
        compile_model(source),
        AnalysisRequest(
            analysis_key="gpr",
            config={"training_size": 32, "validation_size": 64, "trend": "LINEAR"},
            output_targets=[0],
        ),
        seed=7,
    )
    assert result.payload.metrics["validation_r2"] > 0.999
    assert result.payload.facts["trend"] == "Linear"


def test_gpr_rejects_discrete_inputs_and_constant_outputs() -> None:
    discrete = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["x"], ["x"])
problem = ot.Poisson(3.0)
"""
    )
    with pytest.raises(IncompatibleAnalysisError, match="continuous input"):
        run_analysis(discrete, AnalysisRequest(analysis_key="gpr"))

    constant = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["x"], ["1.0"])
problem = ot.Uniform(-1.0, 1.0)
"""
    )
    with pytest.raises(IncompatibleAnalysisError, match="constant selected output"):
        run_analysis(constant, AnalysisRequest(analysis_key="gpr"))


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


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("method", "tolerance"),
    [
        ("SORM", 1e-3),
        ("MONTE_CARLO", 0.12),
        ("DIRECTIONAL_SAMPLING", 0.03),
        ("SUBSET_SAMPLING", 0.03),
    ],
)
def test_stable_reliability_methods_on_standard_normal_half_space(
    method: str, tolerance: float
) -> None:
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
            config={
                "method": method,
                "threshold": 0.0,
                "operator": ">",
                "maximum_evaluations": 1_000,
                "target_coefficient_of_variation": 0.1,
                "block_size": 10,
            },
            output_targets=[0],
        ),
        seed=31,
    )
    assert result.payload.metrics["event_probability"] == pytest.approx(0.5, abs=tolerance)
    assert 0 < result.runtime.model_evaluations <= 1_000
