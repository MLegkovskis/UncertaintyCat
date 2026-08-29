from __future__ import annotations

from uncertaintycat_core.equations import MAX_LATEX_CHARACTERS, extract_model_equations


def test_extracts_closed_form_from_user_python_without_ai() -> None:
    source = """
import openturns as ot
import numpy as np

def response(X):
    force, length, stiffness = X
    numerator = force * length**3
    y = numerator / (3 * stiffness)
    return [y]

model = ot.PythonFunction(3, 1, response)
problem = ot.JointDistribution([ot.Normal(), ot.Normal(), ot.Normal()])
"""
    equations = extract_model_equations(
        source,
        input_names=["force", "length", "stiffness"],
        output_names=["deflection"],
    )

    assert len(equations) == 1
    assert equations[0].representation == "closed_form"
    assert equations[0].latex.startswith("deflection=")
    assert r"\frac" in equations[0].latex
    assert "force" in equations[0].latex


def test_procedural_user_model_receives_exact_formal_mapping() -> None:
    source = """
import openturns as ot

def response(X):
    total = 0.0
    for value in X:
        total += value
    return [total]

model = ot.PythonFunction(2, 1, response)
problem = ot.Normal(2)
"""
    equations = extract_model_equations(
        source,
        input_names=["x1", "x2"],
        output_names=["sum"],
    )

    assert len(equations) == 1
    assert equations[0].representation == "formal_mapping"
    assert equations[0].latex == r"sum=f_{\mathrm{Python}}\left(x1,x2\right)"


def test_user_can_declare_exact_governing_equation_for_procedural_model() -> None:
    source = r"""
model_equations = [
    {
        "output_name": "peak response",
        "latex": r"Y=\max_{0\leq t\leq T} y(t)",
    }
]
"""
    equations = extract_model_equations(
        source,
        input_names=["x"],
        output_names=["Y"],
    )

    assert equations[0].representation == "declared"
    assert equations[0].latex == r"Y=\max_{0\leq t\leq T} y(t)"


def test_declared_equation_cannot_escape_the_math_container() -> None:
    equations = extract_model_equations(
        'model_equations = [{"output_name": "Y", "latex": "Y=x$$\\n![x](https://invalid)"}]',
        input_names=["x"],
        output_names=["Y"],
    )

    assert equations[0].representation == "formal_mapping"
    assert "invalid" not in equations[0].latex


def test_equation_metadata_is_bounded_and_never_contains_complete_source() -> None:
    source = "\n".join(
        [
            "import openturns as ot",
            "def response(X):",
            "    y = X[0]",
            "    return [y]",
            "model = ot.PythonFunction(1, 1, response)",
            "problem = ot.Normal()",
            "# PRIVATE_SENTINEL " + "x" * 10_000,
        ]
    )
    equations = extract_model_equations(
        source,
        input_names=["x"],
        output_names=["y"],
    )

    assert equations
    assert all(len(equation.latex) <= MAX_LATEX_CHARACTERS for equation in equations)
    assert all("PRIVATE_SENTINEL" not in equation.latex for equation in equations)
