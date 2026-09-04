from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

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

CALIBRATION_SOURCE = """
import openturns as ot
model = ot.SymbolicFunction(["a", "b", "c", "x"], ["a + b * exp(c * x)"])
model.setOutputDescription(["y"])
problem = ot.JointDistribution([
    ot.Uniform(0.0, 5.0),
    ot.Uniform(0.5, 2.0),
    ot.Uniform(0.1, 0.6),
    ot.Uniform(0.5, 9.5),
])
problem.setDescription(["a", "b", "c", "x"])
"""

CALIBRATION_INPUTS = [[0.5 + index] for index in range(10)]
CALIBRATION_OUTPUTS = [
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
]


def calibration_request(**overrides: object) -> AnalysisRequest:
    config: dict[str, object] = {
        "parameter_indices": [0, 1, 2],
        "starting_values": [1.0, 1.0, 1.0],
        "observed_input_names": ["x"],
        "observed_output_name": "y",
        "observed_inputs": CALIBRATION_INPUTS,
        "observed_outputs": CALIBRATION_OUTPUTS,
        "maximum_calls": 250,
    }
    config.update(overrides)
    return AnalysisRequest(analysis_key="calibration_nlls", config=config, output_targets=[0])


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


@pytest.mark.scientific
def test_hsic_matches_official_ishigami_global_sensitivity_benchmark() -> None:
    source = """
import math
import openturns as ot
model = ot.SymbolicFunction(
    ["X1", "X2", "X3"],
    ["sin(X1) + 5.0 * sin(X2)^2 + 0.1 * X3^4 * sin(X1)"],
)
model.setOutputDescription(["Y"])
problem = ot.JointDistribution([ot.Uniform(-math.pi, math.pi)] * 3)
problem.setDescription(["X1", "X2", "X3"])
"""
    result = run_analysis(
        compile_model(source),
        AnalysisRequest(
            analysis_key="hsic",
            config={"sample_size": 100, "permutations": 1000},
            output_targets=[0],
        ),
        seed=0,
    )
    rows = result.payload.tables["indices"].rows

    assert [float(row[1]) for row in rows] == pytest.approx(
        [0.29807297, 0.00344498, 0.07726572], abs=1.0e-8
    )
    assert [float(row[2]) for row in rows] == pytest.approx(
        [0.0, 0.29670330, 0.00199800], abs=1.0e-8
    )
    assert result.payload.facts["strongest_dependence_input"] == "X1"
    assert result.payload.metrics["estimated_quadratic_work_units"] == 40_160_000


def test_hsic_warns_that_dependent_inputs_confound_global_association() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.6
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
    result = run_analysis(
        compile_model(source),
        AnalysisRequest(
            analysis_key="hsic",
            config={"sample_size": 50, "permutations": 5},
        ),
        seed=17,
    )

    assert "transmitted through other inputs" in " ".join(result.warnings)


@pytest.mark.scientific
def test_target_hsic_matches_official_ishigami_benchmark_repeatably() -> None:
    source = """
import math
import openturns as ot
model = ot.SymbolicFunction(
    ["X1", "X2", "X3"],
    ["sin(X1) + 5.0 * sin(X2)^2 + 0.1 * X3^4 * sin(X1)"],
)
model.setOutputDescription(["Y"])
problem = ot.JointDistribution([ot.Uniform(-math.pi, math.pi)] * 3)
problem.setDescription(["X1", "X2", "X3"])
"""
    request = AnalysisRequest(
        analysis_key="target_hsic",
        config={
            "sample_size": 100,
            "permutations": 100,
            "threshold": 5.0,
            "operator": ">=",
            "smoothing_scale_fraction": 0.1,
        },
        output_targets=[0],
    )
    progress: list[tuple[str, int, str, bool]] = []
    result_a = run_analysis(
        compile_model(source),
        request,
        seed=0,
        progress_callback=lambda phase, percent, message, indeterminate: progress.append(
            (phase, percent, message, indeterminate)
        ),
    )
    result_b = run_analysis(compile_model(source), request, seed=0)
    rows = {str(row[0]): row for row in result_a.payload.tables["target_indices"].rows}

    assert [float(rows[name][1]) for name in ("X1", "X2", "X3")] == pytest.approx(
        [0.26863688, 0.00468423, 0.00339962], abs=1.0e-8
    )
    assert [float(rows[name][2]) for name in ("X1", "X2", "X3")] == pytest.approx(
        [0.00107494, 0.00001868, 0.00001411], abs=1.0e-8
    )
    assert [float(rows[name][4]) for name in ("X1", "X2", "X3")] == pytest.approx(
        [0.0, 0.26201467, 0.28227083], abs=1.0e-8
    )
    assert [float(rows[name][3]) for name in ("X1", "X2", "X3")] == pytest.approx(
        [0.0, 0.25742574, 0.21782178], abs=1.0e-8
    )
    assert result_a.payload.metrics == result_b.payload.metrics
    assert result_a.payload.tables == result_b.payload.tables
    assert result_a.runtime.model_evaluations == 100
    assert result_a.payload.metrics["model_evaluations"] == 100
    assert result_a.payload.metrics["target_observations"] == 25
    assert result_a.payload.metrics["estimated_quadratic_work_units"] == 4_160_000
    assert "permutations + 4" in result_a.payload.facts["quadratic_work_unit_definition"]
    assert result_a.payload.facts["strongest_target_association_input"] == "X1"
    assert "not a failure-probability" in " ".join(result_a.warnings)
    serialized = result_a.model_dump_json()
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert len(serialized.encode()) < 10_000
    phases = [item[0] for item in progress]
    assert phases == [
        "applicability",
        "openturns",
        "sampling",
        "target_domain",
        "kernel_construction",
        "observed_indices",
        "permutation_inference",
        "ranking",
        "serializing",
    ]
    assert [item[1] for item in progress] == sorted(item[1] for item in progress)
    assert next(item for item in progress if item[0] == "permutation_inference")[3]
    assert all("X1" not in item[2] for item in progress)


@pytest.mark.parametrize(
    ("source", "config", "message"),
    [
        (
            """
import openturns as ot
model = ot.SymbolicFunction(["x"], ["1.0"])
problem = ot.Uniform(-1.0, 1.0)
""",
            {"threshold": 0.0},
            "constant",
        ),
        (
            """
import openturns as ot
model = ot.SymbolicFunction(["x"], ["x"])
problem = ot.Poisson(3.0)
""",
            {"threshold": 3.0},
            "continuous input marginals",
        ),
        (
            """
import openturns as ot
model = ot.SymbolicFunction(["x"], ["x"])
problem = ot.Normal()
""",
            {"threshold": 20.0, "operator": ">="},
            "in the critical domain",
        ),
        (
            """
import openturns as ot
model = ot.SymbolicFunction(["x"], ["x"])
problem = ot.Normal()
""",
            {"threshold": -20.0, "operator": ">="},
            "outside the critical domain",
        ),
    ],
)
def test_target_hsic_rejects_invalid_or_degenerate_samples(
    source: str, config: dict[str, object], message: str
) -> None:
    with pytest.raises(IncompatibleAnalysisError, match=message):
        run_analysis(
            compile_model(source),
            AnalysisRequest(
                analysis_key="target_hsic",
                config={"sample_size": 50, "permutations": 20, **config},
                output_targets=[0],
            ),
            seed=11,
        )


def test_target_hsic_rejects_invalid_contract_and_excessive_work() -> None:
    source = """
import openturns as ot
names = [f"x{index}" for index in range(20)]
model = ot.SymbolicFunction(names, [" + ".join(names)])
problem = ot.Normal(20)
problem.setDescription(names)
"""
    with pytest.raises(ValidationError, match="finite number"):
        run_analysis(
            compile_model(source),
            AnalysisRequest(analysis_key="target_hsic", config={"threshold": float("inf")}),
        )
    with pytest.raises(IncompatibleAnalysisError, match="workload exceeds"):
        run_analysis(
            compile_model(source),
            AnalysisRequest(
                analysis_key="target_hsic",
                config={"sample_size": 500, "permutations": 200},
            ),
        )


def test_target_hsic_warns_that_dependent_inputs_can_confound_association() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.6
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
    result = run_analysis(
        compile_model(source),
        AnalysisRequest(
            analysis_key="target_hsic",
            config={
                "sample_size": 100,
                "permutations": 20,
                "threshold": 0.0,
                "operator": "<=",
            },
        ),
        seed=17,
    )
    assert "dependent" in " ".join(result.warnings).lower()


@pytest.mark.scientific
def test_ancova_matches_correlated_linear_analytical_contributions() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["X1", "X2"], ["4.0*X1 + 5.0*X2"])
model.setOutputDescription(["response"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.3
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
problem.setDescription(["X1", "X2"])
"""
    request = AnalysisRequest(
        analysis_key="ancova",
        config={
            "degree": 3,
            "training_size": 500,
            "validation_size": 300,
            "ancova_sample_size": 4000,
        },
        output_targets=[0],
    )
    result_a = run_analysis(compile_model(source), request, seed=42)
    result_b = run_analysis(compile_model(source), request, seed=42)
    rows = {str(row[0]): row for row in result_a.payload.tables["indices"].rows}

    assert result_a.plugin_version == "1.0.0"
    assert result_a.payload.metrics == result_b.payload.metrics
    assert result_a.payload.tables == result_b.payload.tables
    assert result_a.payload.metrics["validation_q2"] > 0.999
    assert float(rows["X1"][1]) == pytest.approx(22.0 / 53.0, abs=0.04)
    assert float(rows["X2"][1]) == pytest.approx(31.0 / 53.0, abs=0.04)
    assert float(rows["X1"][2]) == pytest.approx(16.0 / 53.0, abs=0.04)
    assert float(rows["X2"][2]) == pytest.approx(25.0 / 53.0, abs=0.04)
    assert float(rows["X1"][3]) == pytest.approx(6.0 / 53.0, abs=0.04)
    assert float(rows["X2"][3]) == pytest.approx(6.0 / 53.0, abs=0.04)
    assert result_a.runtime.model_evaluations == 800
    assert result_a.payload.facts["copula"] == "NormalCopula"
    assert "NaN" not in result_a.model_dump_json()
    recommendation = next(
        item
        for item in compile_model(source).assessment.recommendations
        if item.capability == "ancova"
    )
    assert recommendation.status == "recommended"


def test_ancova_rejects_invalid_applicability_and_untrusted_surrogates() -> None:
    independent = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
problem = ot.Normal(2)
"""
    )
    with pytest.raises(IncompatibleAnalysisError, match="dependent inputs"):
        run_analysis(independent, AnalysisRequest(analysis_key="ancova"))

    discrete = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.2
problem = ot.JointDistribution([ot.Poisson(2.0), ot.Poisson(3.0)], ot.NormalCopula(correlation))
"""
    )
    with pytest.raises(IncompatibleAnalysisError, match="continuous"):
        run_analysis(discrete, AnalysisRequest(analysis_key="ancova"))

    constant = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["1.0"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.2
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
    )
    with pytest.raises(IncompatibleAnalysisError, match="constant"):
        run_analysis(
            constant,
            AnalysisRequest(
                analysis_key="ancova",
                config={"training_size": 128, "validation_size": 64},
            ),
        )

    with pytest.raises(IncompatibleAnalysisError, match="output target"):
        run_analysis(
            compile_model(
                """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.2
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
            ),
            AnalysisRequest(analysis_key="ancova", output_targets=[3]),
        )
    with pytest.raises(IncompatibleAnalysisError, match="output target"):
        run_analysis(
            compile_model(
                """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.2
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
            ),
            AnalysisRequest(analysis_key="ancova", output_targets=[-1]),
        )


def test_ancova_selects_one_target_from_a_multi_output_model() -> None:
    runtime = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2", "2.0*x1 - x2"])
model.setOutputDescription(["sum", "difference"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.25
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
    )
    result = run_analysis(
        runtime,
        AnalysisRequest(
            analysis_key="ancova",
            config={
                "training_size": 128,
                "validation_size": 64,
                "ancova_sample_size": 256,
            },
            output_targets=[1],
        ),
    )

    assert result.payload.facts["output"] == "difference"
    assert result.payload.metrics["validation_q2"] > 0.99


def test_ancova_rejects_a_polynomial_basis_over_the_resource_cap() -> None:
    runtime = compile_model(
        """
import openturns as ot
names = [f"x{index}" for index in range(10)]
model = ot.SymbolicFunction(names, [" + ".join(names)])
correlation = ot.CorrelationMatrix(10)
correlation[0, 1] = 0.2
problem = ot.Normal([0.0] * 10, [1.0] * 10, correlation)
"""
    )

    with pytest.raises(IncompatibleAnalysisError, match="maximum is 500"):
        run_analysis(
            runtime,
            AnalysisRequest(analysis_key="ancova", config={"degree": 6}),
        )


@pytest.mark.scientific
def test_calibration_recovers_official_exponential_parameters_repeatably() -> None:
    runtime_a = compile_model(CALIBRATION_SOURCE)
    runtime_b = compile_model(CALIBRATION_SOURCE)
    calls_before = runtime_a.model.getEvaluationCallsNumber()
    result_a = run_analysis(runtime_a, calibration_request(), seed=0)
    calls_after = runtime_a.model.getEvaluationCallsNumber()
    result_b = run_analysis(runtime_b, calibration_request(), seed=0)

    rows = result_a.payload.tables["calibrated_parameters"].rows
    calibrated = [float(row[2]) for row in rows]
    for obtained, expected, tolerance in zip(
        calibrated, [2.8, 1.2, 0.5], [0.05, 0.02, 0.005], strict=True
    ):
        assert obtained == pytest.approx(expected, abs=tolerance)
    assert calibrated == pytest.approx(
        [2.7731136593401917, 1.2035076055520555, 0.49974911285083384],
        abs=1.0e-12,
    )
    assert result_a.payload.metrics == result_b.payload.metrics
    assert result_a.payload.tables == result_b.payload.tables
    assert result_a.payload.series == result_b.payload.series
    assert result_a.payload.matrices == result_b.payload.matrices
    assert result_a.runtime.model_evaluations == calls_after - calls_before
    assert result_a.payload.metrics["model_evaluations"] == (result_a.runtime.model_evaluations)
    assert result_a.payload.metrics["observations"] == 10
    assert result_a.payload.tables["observations_and_predictions"].row_count == 10
    assert result_a.payload.tables["observations_and_predictions"].truncated is False
    assert result_a.payload.metrics["rmse_after"] < 0.05
    assert result_a.payload.facts["bootstrap_size"] == 0
    assert result_a.payload.facts["report_payload_limit_bytes"] == 1_000_000
    assert "not an exact confidence guarantee" in str(
        result_a.payload.facts["parameter_uncertainty"]
    )
    assert "identifiability" in " ".join(result_a.warnings)
    serialized = result_a.model_dump_json()
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    assert len(serialized.encode()) < 100_000


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parameter_indices": []}, "at least 1 item"),
        ({"parameter_indices": [0, 0, 2]}, "must be unique"),
        ({"starting_values": [1.0, 1.0]}, "one starting value"),
        ({"starting_values": [1.0, float("inf"), 1.0]}, "finite number"),
        ({"observed_input_names": ["wrong"]}, "exactly match"),
        ({"observed_output_name": "wrong"}, "must be named 'y'"),
        ({"observed_output_name": "x"}, "column names must be unique"),
        ({"observed_inputs": [[0.5, 1.0]] * 10}, "named input columns"),
        ({"observed_outputs": CALIBRATION_OUTPUTS[:-1]}, "row counts must match"),
        ({"observed_outputs": [float("nan")] * 10}, "finite number"),
        ({"observed_outputs": [4.0] * 10}, "must vary"),
        ({"maximum_calls": 501}, "less than or equal to 500"),
    ],
)
def test_calibration_rejects_invalid_observation_contracts(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises((ValidationError, IncompatibleAnalysisError), match=message):
        run_analysis(compile_model(CALIBRATION_SOURCE), calibration_request(**overrides))


def test_calibration_rejects_too_few_observations_and_excessive_bounds() -> None:
    five_parameter_source = """
import openturns as ot
model = ot.SymbolicFunction(
    ["a", "b", "c", "d", "x"],
    ["a + b*x + c*x^2 + d*x^3"],
)
model.setOutputDescription(["y"])
problem = ot.Normal(5)
problem.setDescription(["a", "b", "c", "d", "x"])
"""
    with pytest.raises(IncompatibleAnalysisError, match="At least 6 observations"):
        run_analysis(
            compile_model(five_parameter_source),
            calibration_request(
                parameter_indices=[0, 1, 2, 3],
                starting_values=[0.0, 0.0, 0.0, 0.0],
                observed_inputs=CALIBRATION_INPUTS[:5],
                observed_outputs=CALIBRATION_OUTPUTS[:5],
            ),
        )

    with pytest.raises(ValidationError, match="at most 250 items"):
        run_analysis(
            compile_model(CALIBRATION_SOURCE),
            calibration_request(
                observed_inputs=[[float(index)] for index in range(251)],
                observed_outputs=[float(index) for index in range(251)],
            ),
        )

    excessive_parameters = list(range(9))
    with pytest.raises(ValidationError, match="at most 8 items"):
        run_analysis(
            compile_model(CALIBRATION_SOURCE),
            calibration_request(
                parameter_indices=excessive_parameters,
                starting_values=[0.0] * len(excessive_parameters),
            ),
        )


def test_calibration_retains_complete_predictions_within_the_report_byte_cap() -> None:
    inputs = [[0.5 + 9.0 * index / 249] for index in range(250)]
    outputs = [2.8 + 1.2 * math.exp(0.5 * row[0]) for row in inputs]
    runtime = compile_model(CALIBRATION_SOURCE)
    calls_before = runtime.model.getEvaluationCallsNumber()
    result = run_analysis(
        runtime,
        calibration_request(
            starting_values=[2.5, 1.1, 0.45],
            observed_inputs=inputs,
            observed_outputs=outputs,
        ),
        seed=0,
    )
    calls_after = runtime.model.getEvaluationCallsNumber()

    table = result.payload.tables["observations_and_predictions"]
    assert table.row_count == 250
    assert len(table.rows) == 250
    assert table.truncated is False
    assert len(result.payload.model_dump_json().encode()) < 1_000_000
    assert result.runtime.model_evaluations == calls_after - calls_before
    assert [
        float(row[2]) for row in result.payload.tables["calibrated_parameters"].rows
    ] == pytest.approx([2.8, 1.2, 0.5], abs=1.0e-6)


def test_calibration_rejects_discrete_and_locally_non_identifiable_parameters() -> None:
    discrete = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["a", "x"], ["a + x"])
model.setOutputDescription(["y"])
problem = ot.JointDistribution([ot.Poisson(2.0), ot.Normal()])
problem.setDescription(["a", "x"])
"""
    )
    with pytest.raises(IncompatibleAnalysisError, match="continuous model inputs"):
        run_analysis(
            discrete,
            calibration_request(
                parameter_indices=[0],
                starting_values=[1.0],
                observed_input_names=["x"],
            ),
        )

    non_identifiable = compile_model(
        """
import openturns as ot
model = ot.SymbolicFunction(["a", "b", "x"], ["a + b + x"])
model.setOutputDescription(["y"])
problem = ot.Normal(3)
problem.setDescription(["a", "b", "x"])
"""
    )
    with pytest.raises(IncompatibleAnalysisError, match="rank-deficient"):
        run_analysis(
            non_identifiable,
            calibration_request(
                parameter_indices=[0, 1],
                starting_values=[1.0, 1.0],
                observed_input_names=["x"],
            ),
        )


@pytest.mark.parametrize(
    ("analysis_key", "config"),
    [
        ("correlation", {"sample_size": 100}),
        ("fast", {"sample_size": 128}),
        ("hsic", {"sample_size": 50, "permutations": 5}),
        (
            "target_hsic",
            {"sample_size": 100, "permutations": 20, "threshold": 0.0},
        ),
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
                "block_size": 1 if method == "SUBSET_SAMPLING" else 10,
                **({"subset_sample_size": 1_000} if method == "SUBSET_SAMPLING" else {}),
            },
            output_targets=[0],
        ),
        seed=31,
    )
    assert result.payload.metrics["event_probability"] == pytest.approx(0.5, abs=tolerance)
    assert 0 < result.runtime.model_evaluations <= 1_000
