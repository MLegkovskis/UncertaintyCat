# ─────────────────────────────────────────────────────────────────────────────
# modules/pce_least_squares.py
# Fully self-contained LS-PCE builder with Plotly graphics + surrogate download
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import openturns as ot
import openturns.experimental as otexp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── internal helpers (kept local to avoid external imports) ──────────────────
def get_ot_model(model):
    """Ensure we have an ot.Function."""
    if isinstance(model, ot.Function):
        return model
    # otherwise wrap the callable
    dim = model.__code__.co_argcount
    return ot.PythonFunction(dim, 1, model)

# ╭──────────────────────── serialization helpers ───────────────────────────╮
class _SuperPoint:
    def __init__(self, pt: ot.Point): self.pt = pt
    def to_py(self): return "Point(["+",".join(map(str,self.pt))+"])"

class _SuperSample:
    def __init__(self, s: ot.Sample):
        self.s = s

    def to_py(self, var_name: str = "", indent: str = "") -> str:
        rows = [ "[" + ",".join(map(str, row)) + "]" for row in self.s ]
        code = indent
        if var_name:
            code += f"{var_name} = "
        code += "Sample([" + ",".join(rows) + "])"
        return code


class _SuperDistribution:
    """Serialize 1-D OT distribution or JointDistribution (independent copula)."""
    def __init__(self, d: ot.Distribution): self.d=d
    def to_py(self, var="distribution"):
        if self.d.getDimension()==1:
            impl = self.d.getImplementation()
            cls  = impl.getClassName()
            pars = ",".join(map(str,impl.getParameter()))
            return f"{var} = {cls}({pars})"
        # Joint (independent copula) – recurse
        code = "marginals=[]\n"
        for i in range(self.d.getDimension()):
            m = _SuperDistribution(self.d.getMarginal(i)).to_py("")
            code += f"marginals.append({m.split('=')[-1].strip()})\n"
        code += f"{var} = JointDistribution(marginals)"
        return code

class _SuperChaosResult:
    """Turn a FunctionalChaosResult into a runnable Python surrogate."""
    def __init__(self, res: ot.FunctionalChaosResult): self.r = res
    def to_python(self):
        dist_code = _SuperDistribution(self.r.getDistribution()).to_py()
        coeffs    = _SuperSample(self.r.getCoefficients()).to_py("coefficients")
        idx       = list(self.r.getIndices())
        code  = "from openturns import *\n\n"
        code += "# --- Input distribution\n" + dist_code + "\n\n"
        code += "# --- PCE coefficients and basis indices\n"
        code += f"indices = {idx}\n{coeffs}\n\n"
        code += """# --- Build orthonormal basis
input_dim = distribution.getDimension()
marginals = [distribution.getMarginal(i) for i in range(input_dim)]
basis = OrthogonalProductPolynomialFactory(marginals)
funcs = [basis.build(i) for i in indices]
meta  = DualLinearCombinationFunction(funcs, coefficients)
measure = basis.getMeasure()
transform = DistributionTransformation(distribution, measure)
metaModel = ComposedFunction(meta, transform)

def function_of_interest(X):
    return [ metaModel(X)[0] ]

model = PythonFunction(distribution.getDimension(), 1, function_of_interest)
"""
        return code
# ╰──────────────────────────────────────────────────────────────────────────╯

# ─────────────────────────────────────────────────────────────────────────────
def _fit_pce(
    f: ot.Function,
    dist: ot.Distribution,
    n_train: int,
    basis_frac: float,
    use_selection: bool,
) -> ot.FunctionalChaosResult:
    X = dist.getSample(n_train)
    Y = f(X)
    basis = ot.OrthogonalProductPolynomialFactory(
        [dist.getMarginal(i) for i in range(dist.getDimension())]
    )
    size = int(basis_frac * n_train)
    proj = ot.LeastSquaresStrategy(
        X, Y,
        ot.LeastSquaresMetaModelSelectionFactory() if use_selection
        else ot.PenalizedLeastSquaresAlgorithmFactory()
    )
    algo = ot.FunctionalChaosAlgorithm(
        X, Y, dist,
        ot.FixedStrategy(basis, size),
        proj
    )
    algo.run()
    return algo.getResult()

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
def pce_least_squares_analysis(
    model,
    problem,
    code_str: str | None = None,
    *,
    N_train: int = 1000,
    N_validate: int = 1000,
    basis_size_factor: float = 0.5,
    use_model_selection: bool = False,
    language_model: str | None = None,
    display_results: bool = True,
):
    """Fit a least-squares PCE, validate, plot (Plotly) and optionally download surrogate."""
    f_ot = get_ot_model(model)
    res  = _fit_pce(
        f_ot, problem,
        n_train=N_train,
        basis_frac=basis_size_factor,
        use_selection=use_model_selection,
    )

    # ── 1. Validation -------------------------------------------------------
    Xv = problem.getSample(N_validate)      # independent sample
    Yv = f_ot(Xv)                           # true model outputs
    Yh = res.getMetaModel()(Xv)             # PCE predictions

    val   = ot.MetaModelValidation(Yv, Yh)
    r2    = val.computeR2Score()[0]

    r2_loo = otexp.FunctionalChaosValidation(
        res, ot.LeaveOneOutSplitter(N_train)
    ).computeR2Score()[0]

    # ── 2. Sobol indices ----------------------------------------------------
    sens  = ot.FunctionalChaosSobolIndices(res)
    names = list(problem.getDescription())
    S1    = [sens.getSobolIndex(i)      for i in range(problem.getDimension())]
    ST    = [sens.getSobolTotalIndex(i) for i in range(problem.getDimension())]

    # ── 3. Plotly figures ----------------------------------------------------
    figs = {}

    # Validation scatter
    figs["validation"] = px.scatter(
        {
            "Model": [float(y[0]) for y in Yv],
            "PCE":   [float(y[0]) for y in Yh],
        },
        x="Model",
        y="PCE",
        title=f"PCE validation — R² = {r2:.3f}",
    ).update_traces(marker=dict(size=6, opacity=0.7))

    # Sobol bar chart
    figs["sobol"] = go.Figure([
        go.Bar(name="S1", x=names, y=S1),
        go.Bar(name="ST", x=names, y=ST),
    ]).update_layout(
        barmode="group",
        title="Sobol indices from PCE",
        yaxis_title="Sensitivity index",
    )

# ── 4. Display ────────────────────────────────────────────────────────────
    if display_results:
        st.plotly_chart(
            figs["validation"],
            use_container_width=True,
            key="pce_validation_immediate",
        )
        st.plotly_chart(
            figs["sobol"],
            use_container_width=True,
            key="pce_sobol_immediate",
        )

    # ── 5. Download surrogate ────────────────────────────────────────────────
    surrogate_py = _SuperChaosResult(res).to_python()
    if display_results:
        st.download_button(
            "💾 Download Python surrogate",
            data=surrogate_py,
            file_name="pce_surrogate.py",
            mime="text/x-python",
        )

    # ── 6. Optional LLM commentary ------------------------------------------
    ai_md = None
    if language_model and code_str:
        from utils.core_utils import call_groq_api  # lazy import
        prompt = (
            f"PCE built with N_train={N_train}, R²={r2:.3f}, R²_LOO={r2_loo:.3f}\n"
            + "\n".join(
                f"{n}: S1={s1:.3f}, ST={st:.3f}"
                for n, s1, st in zip(names, S1, ST)
            )
        )
        ai_md = call_groq_api(prompt, model_name=language_model)
        if display_results:
            st.markdown(ai_md)

    # ── 7. Return structured results ----------------------------------------
    return {
        "chaos_result": res,
        "r2":           r2,
        "r2_loo":       r2_loo,
        "sobol_S1":     S1,
        "sobol_ST":     ST,
        "figs":         figs,
        "surrogate_py": surrogate_py,
        "ai_md":        ai_md,
    }

# ─────────────────────────────────────────────────────────────────────────────

def display_pce_results(results: dict, container=None):
    if container is None:
        container = st
    if not results:
        return

    if "validation" in results["figs"]:
        container.plotly_chart(
            results["figs"]["validation"],
            use_container_width=True,
            key="pce_validation_report",
        )
    if "sobol" in results["figs"]:
        container.plotly_chart(
            results["figs"]["sobol"],
            use_container_width=True,
            key="pce_sobol_report",
        )

    if results.get("ai_md"):
        container.markdown(results["ai_md"])
