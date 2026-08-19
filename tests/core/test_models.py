from __future__ import annotations

from pathlib import Path

import pytest

from uncertaintycat_core.errors import InvalidModelError, UnsafeModelError
from uncertaintycat_core.model import compile_model

EXAMPLES = sorted(Path("examples").glob("*.py"))


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
