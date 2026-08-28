from __future__ import annotations

from pathlib import Path

import pytest

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
    assert ishigami.assessment.version == "1.2.0"
    assert ishigami.assessment.workflow.path == "direct"
    assert morris.assessment.workflow.path == "dimensionality_reduction"
