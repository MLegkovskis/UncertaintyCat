"""Bounded, deterministic equation extraction for authenticated model metadata."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from uncertaintycat_core.contracts import ModelEquation

MAX_EQUATIONS = 6
MAX_LATEX_CHARACTERS = 4_000
MAX_RENDER_DEPTH = 18

_GREEK = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\varepsilon",
    "eta": r"\eta",
    "theta": r"\theta",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "phi": r"\phi",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "omega": r"\omega",
    "zeta": r"\zeta",
}


class UnsupportedEquation(ValueError):
    """Raised when a source construct cannot be represented faithfully."""


def _identifier(name: str) -> str:
    if name in _GREEK:
        return _GREEK[name]
    parts = name.split("_", 1)
    base = _GREEK.get(parts[0], re.sub(r"[^A-Za-z0-9]", "", parts[0]) or "x")
    if len(parts) == 1 or not parts[1]:
        return base
    subscript = re.sub(r"[^A-Za-z0-9]", "", parts[1])
    return rf"{base}_{{{subscript}}}" if subscript else base


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


@dataclass
class _Renderer:
    assignments: dict[str, ast.expr]
    indexed_inputs: dict[int, str]
    _active: set[str] = field(default_factory=set)

    def render(self, node: ast.AST, depth: int = 0) -> str:
        if depth > MAX_RENDER_DEPTH:
            raise UnsupportedEquation("The expanded expression is too deeply nested.")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return r"\mathrm{true}" if node.value else r"\mathrm{false}"
            if isinstance(node.value, (int, float)):
                return f"{node.value:g}" if isinstance(node.value, float) else str(node.value)
            raise UnsupportedEquation("Non-numeric constants are not equations.")
        if isinstance(node, ast.Name):
            if node.id in self.assignments and node.id not in self._active:
                self._active.add(node.id)
                try:
                    return self.render(self.assignments[node.id], depth + 1)
                finally:
                    self._active.remove(node.id)
            return _identifier(node.id)
        if isinstance(node, ast.UnaryOp):
            operand = self.render(node.operand, depth + 1)
            if isinstance(node.op, ast.USub):
                return rf"-\left({operand}\right)"
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.Not):
                return rf"\neg\left({operand}\right)"
            raise UnsupportedEquation("Unsupported unary operator.")
        if isinstance(node, ast.BinOp):
            left = self.render(node.left, depth + 1)
            right = self.render(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return rf"\left({left}+{right}\right)"
            if isinstance(node.op, ast.Sub):
                return rf"\left({left}-{right}\right)"
            if isinstance(node.op, ast.Mult):
                return rf"\left({left}\,{right}\right)"
            if isinstance(node.op, ast.Div):
                return rf"\frac{{{left}}}{{{right}}}"
            if isinstance(node.op, ast.Pow):
                return rf"\left({left}\right)^{{{right}}}"
            if isinstance(node.op, ast.Mod):
                return rf"\operatorname{{mod}}\left({left},{right}\right)"
            raise UnsupportedEquation("Unsupported binary operator.")
        if isinstance(node, ast.Call):
            return self._render_call(node, depth + 1)
        if isinstance(node, ast.Attribute):
            if _call_name(node) in {"np.pi", "numpy.pi", "math.pi"}:
                return r"\pi"
            return rf"{self.render(node.value, depth + 1)}_{{{_identifier(node.attr)}}}"
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, int)
                and node.slice.value in self.indexed_inputs
            ):
                return _identifier(self.indexed_inputs[node.slice.value])
            value = self.render(node.value, depth + 1)
            index = self.render(node.slice, depth + 1)
            return rf"{value}_{{{index}}}"
        if isinstance(node, ast.IfExp):
            condition = self.render(node.test, depth + 1)
            body = self.render(node.body, depth + 1)
            otherwise = self.render(node.orelse, depth + 1)
            return (
                r"\begin{cases}"
                + body
                + rf",&{condition}\\{otherwise},&\text{{otherwise}}\end{{cases}}"
            )
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            left = self.render(node.left, depth + 1)
            right = self.render(node.comparators[0], depth + 1)
            symbols = {
                ast.Lt: "<",
                ast.LtE: r"\leq",
                ast.Gt: ">",
                ast.GtE: r"\geq",
                ast.Eq: "=",
                ast.NotEq: r"\neq",
            }
            symbol = symbols.get(type(node.ops[0]))
            if not symbol:
                raise UnsupportedEquation("Unsupported comparison.")
            return f"{left}{symbol}{right}"
        if isinstance(node, ast.BoolOp):
            joiner = r"\land" if isinstance(node.op, ast.And) else r"\lor"
            return joiner.join(self.render(value, depth + 1) for value in node.values)
        raise UnsupportedEquation(f"Unsupported Python expression: {type(node).__name__}.")

    def _render_call(self, node: ast.Call, depth: int) -> str:
        name = _call_name(node.func)
        short = name.rsplit(".", 1)[-1]
        arguments = [self.render(argument, depth + 1) for argument in node.args]
        if short == "sqrt" and len(arguments) == 1:
            return rf"\sqrt{{{arguments[0]}}}"
        if short in {"sin", "cos", "tan", "sinh", "cosh", "tanh", "log"} and len(arguments) == 1:
            operator = "ln" if short == "log" else short
            return rf"\{operator}\left({arguments[0]}\right)"
        if short == "exp" and len(arguments) == 1:
            return rf"e^{{{arguments[0]}}}"
        if short == "expm1" and len(arguments) == 1:
            return rf"\left(e^{{{arguments[0]}}}-1\right)"
        if short == "radians" and len(arguments) == 1:
            return rf"\frac{{\pi}}{{180}}{arguments[0]}"
        if short in {"abs", "absolute", "fabs"} and len(arguments) == 1:
            return rf"\left|{arguments[0]}\right|"
        if short in {"max", "maximum", "min", "minimum"} and arguments:
            operator = "max" if short in {"max", "maximum"} else "min"
            return rf"\{operator}\left({','.join(arguments)}\right)"
        if short == "pow" and len(arguments) == 2:
            return rf"\left({arguments[0]}\right)^{{{arguments[1]}}}"
        if short in {"float", "int"} and len(arguments) == 1:
            return arguments[0]
        if short in {"fsolve", "solve_ivp", "odeint", "quad"}:
            raise UnsupportedEquation(f"{short} is a procedural numerical solver.")
        if not short:
            raise UnsupportedEquation("Dynamic function calls cannot be rendered safely.")
        safe_name = re.sub(r"[^A-Za-z0-9_]", "", short)
        return rf"\operatorname{{{safe_name}}}\left({','.join(arguments)}\right)"


def _explicit_equations(tree: ast.Module) -> list[ModelEquation]:
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "model_equations" for target in targets
        ):
            continue
        if statement.value is None:
            return []
        try:
            raw = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError):
            return []
        if not isinstance(raw, list):
            return []
        equations: list[ModelEquation] = []
        for item in raw[:MAX_EQUATIONS]:
            if not isinstance(item, dict):
                continue
            output_name = item.get("output_name") or item.get("outputName")
            latex = item.get("latex")
            if not isinstance(output_name, str) or not isinstance(latex, str):
                continue
            if (
                not output_name.strip()
                or not latex.strip()
                or "$" in latex
                or len(latex) > MAX_LATEX_CHARACTERS
            ):
                continue
            equations.append(
                ModelEquation(
                    output_name=output_name.strip(),
                    latex=latex.strip(),
                    representation="declared",
                )
            )
        return equations
    return []


def _python_callback(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    callback_name: str | None = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "model" for target in statement.targets
        ):
            continue
        if not isinstance(statement.value, ast.Call):
            continue
        if _call_name(statement.value.func).rsplit(".", 1)[-1] != "PythonFunction":
            continue
        if len(statement.value.args) >= 3 and isinstance(statement.value.args[2], ast.Name):
            callback_name = statement.value.args[2].id
            break
    if not callback_name:
        return None
    return next(
        (
            statement
            for statement in tree.body
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == callback_name
        ),
        None,
    )


def _symbolic_equations(tree: ast.Module, output_names: list[str]) -> list[ModelEquation]:
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "model" for target in statement.targets
        ):
            continue
        call = statement.value
        if (
            not isinstance(call, ast.Call)
            or _call_name(call.func).rsplit(".", 1)[-1] != "SymbolicFunction"
        ):
            continue
        if len(call.args) < 2:
            return []
        try:
            formulas = ast.literal_eval(call.args[1])
        except (ValueError, TypeError, SyntaxError):
            return []
        if not isinstance(formulas, list):
            return []
        equations: list[ModelEquation] = []
        for index, formula in enumerate(formulas[:MAX_EQUATIONS]):
            if not isinstance(formula, str) or len(formula) > MAX_LATEX_CHARACTERS:
                continue
            try:
                expression = ast.parse(formula.replace("^", "**"), mode="eval").body
                latex = _Renderer({}, {}).render(expression)
            except (SyntaxError, UnsupportedEquation):
                latex = rf"\operatorname{{SymbolicFunction}}_{{{index + 1}}}(\mathbf{{x}})"
            name = output_names[index] if index < len(output_names) else f"Y{index + 1}"
            equations.append(
                ModelEquation(
                    output_name=name,
                    latex=f"{_identifier(name)}={latex}",
                    representation="closed_form",
                )
            )
        return equations
    return []


def _formal_mapping(input_names: list[str], output_names: list[str]) -> ModelEquation:
    inputs = ",".join(_identifier(name) for name in input_names)
    outputs = ",".join(_identifier(name) for name in output_names)
    left = outputs if len(output_names) == 1 else rf"\begin{{bmatrix}}{outputs}\end{{bmatrix}}"
    return ModelEquation(
        output_name="Validated Python mapping",
        latex=rf"{left}=f_{{\mathrm{{Python}}}}\left({inputs}\right)",
        representation="formal_mapping",
    )


def _python_equations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    input_names: list[str],
    output_names: list[str],
) -> list[ModelEquation]:
    if any(
        isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With))
        for node in ast.walk(function)
    ):
        return []
    argument_name = function.args.args[0].arg if function.args.args else "X"
    assignments: dict[str, ast.expr] = {}
    indexed_inputs = {index: name for index, name in enumerate(input_names)}
    return_node: ast.Return | None = None
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(statement.value, ast.Name)
                and statement.value.id == argument_name
            ):
                for index, element in enumerate(target.elts):
                    if isinstance(element, ast.Name) and index < len(input_names):
                        assignments[element.id] = ast.Name(id=input_names[index], ctx=ast.Load())
                continue
            if isinstance(target, ast.Name):
                assignments[target.id] = statement.value
                continue
        if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name):
            previous = assignments.get(
                statement.target.id, ast.Name(id=statement.target.id, ctx=ast.Load())
            )
            assignments[statement.target.id] = ast.BinOp(
                left=previous,
                op=statement.op,
                right=statement.value,
            )
            continue
        if isinstance(statement, ast.Return):
            return_node = statement
            break
        if isinstance(statement, (ast.If, ast.Match, ast.Raise)):
            return []
    if return_node is None or return_node.value is None:
        return []
    returned = (
        list(return_node.value.elts)
        if isinstance(return_node.value, (ast.List, ast.Tuple))
        else [return_node.value]
    )
    renderer = _Renderer(assignments, indexed_inputs)
    equations: list[ModelEquation] = []
    for index, expression in enumerate(returned[:MAX_EQUATIONS]):
        try:
            rendered = renderer.render(expression)
        except UnsupportedEquation:
            return []
        output_name = output_names[index] if index < len(output_names) else f"Y{index + 1}"
        latex = f"{_identifier(output_name)}={rendered}"
        if len(latex) > MAX_LATEX_CHARACTERS:
            return []
        equations.append(
            ModelEquation(
                output_name=output_name,
                latex=latex,
                representation="closed_form",
            )
        )
    return equations


def extract_model_equations(
    source: str,
    *,
    input_names: list[str],
    output_names: list[str],
) -> list[ModelEquation]:
    """Return safe derived equation metadata without retaining or transmitting source."""
    fallback = [_formal_mapping(input_names, output_names)]
    try:
        tree = ast.parse(source, filename="model.py")
        explicit = _explicit_equations(tree)
        if explicit:
            return explicit
        symbolic = _symbolic_equations(tree, output_names)
        if symbolic:
            return symbolic
        function = _python_callback(tree)
        if not function:
            return fallback
        extracted = _python_equations(function, input_names, output_names)
        return extracted or fallback
    except Exception:
        return fallback
