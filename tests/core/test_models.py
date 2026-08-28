from __future__ import annotations

from pathlib import Path

import pytest

from uncertaintycat_core.catalog import analysis_catalog
from uncertaintycat_core.errors import InvalidModelError, UnsafeModelError
from uncertaintycat_core.model import compile_model, recommend_workflow

EXAMPLES = sorted(Path("examples").glob("*.py"))


def test_public_example_inventory_contains_all_approved_models() -> None:
    assert len(EXAMPLES) == 24


@pytest.mark.scientific
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_all_bundled_examples_validate(example: Path) -> None:
    runtime = compile_model(example.read_text(), validation_sample_size=3)
    assert runtime.metadata.input_dimension == runtime.problem.getDimension()
    assert runtime.metadata.output_dimension == runtime.model.getOutputDimension()
    assert len(runtime.metadata.source_hash) == 64


def test_rejects_unapproved_import() -> None:
    with pytest.raises(UnsafeModelError, match="not in the curated"):
        compile_model("import os\nmodel = None\nproblem = None")


def test_rejects_dimension_mismatch() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x", "y"], ["x + y"])
problem = ot.Normal()
"""
    with pytest.raises(InvalidModelError, match="dimension"):
        compile_model(source)


@pytest.mark.parametrize(
    ("dimension", "projected_ms", "surrogate_eligible", "expected"),
    [
        (3, 120.0, True, "direct"),
        (15, 120.0, True, "dimensionality_reduction"),
        (20, 20_000.0, True, "dimensionality_reduction"),
        (6, 8_000.0, True, "surrogate"),
        (6, 8_000.0, False, "direct"),
    ],
)
def test_workflow_recommendation_is_deterministic(
    dimension: int,
    projected_ms: float,
    surrogate_eligible: bool,
    expected: str,
) -> None:
    recommendation = recommend_workflow(
        input_dimension=dimension,
        projected_1000_evaluation_runtime_ms=projected_ms,
        surrogate_eligible=surrogate_eligible,
    )
    assert recommendation.path == expected


def test_reference_models_receive_expected_workflow_routes() -> None:
    ishigami = compile_model(Path("examples/Ishigami.py").read_text(), validation_sample_size=3)
    morris = compile_model(
        Path("examples/Morris_Function.py").read_text(), validation_sample_size=3
    )
    assert ishigami.assessment.version == "1.4.0"
    assert ishigami.assessment.workflow.path == "direct"
    assert morris.assessment.workflow.path == "dimensionality_reduction"


def test_assessment_covers_every_plugin_with_model_specific_safe_configuration() -> None:
    runtime = compile_model(Path("examples/Ishigami.py").read_text())
    recommendations = {item.capability: item for item in runtime.assessment.recommendations}

    assert set(recommendations) == {
        *(entry.key for entry in analysis_catalog()),
        "distribution_fitting",
    }
    assert recommendations["hsic"].status == "available"
    assert recommendations["hsic"].safe_config == {
        "maximum_sample_size": 600,
        "permutations": 100,
    }
    assert recommendations["ancova"].status == "incompatible"
    assert "dependent input copula" in recommendations["ancova"].compatibility_warnings[0]


def test_dependent_input_assessment_disables_independence_methods_with_exact_reasons() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2"])
correlation = ot.CorrelationMatrix(2)
correlation[0, 1] = 0.5
problem = ot.Normal([0.0, 0.0], [1.0, 1.0], correlation)
"""
    recommendations = {
        item.capability: item for item in compile_model(source).assessment.recommendations
    }

    for key in ("sobol", "fast", "morris", "pce"):
        assert recommendations[key].status == "incompatible"
        assert recommendations[key].rationale_codes == ["INDEPENDENT_INPUTS_REQUIRED"]
        assert "dependent copula" in recommendations[key].compatibility_warnings[0]
    assert recommendations["ancova"].status == "recommended"
    assert recommendations["hsic"].status == "available"


def test_discrete_input_assessment_preserves_sampling_but_disables_kernel_metamodels() -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["x"], ["x * x"])
problem = ot.Poisson(3.0)
"""
    recommendations = {
        item.capability: item for item in compile_model(source).assessment.recommendations
    }

    assert recommendations["monte_carlo"].status == "recommended"
    assert recommendations["reliability"].status == "available"
    for key in ("hsic", "target_hsic", "gpr", "pce"):
        assert recommendations[key].status == "incompatible"
        assert "continuous" in recommendations[key].compatibility_warnings[0].lower()
