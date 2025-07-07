# ─────────────────────────────────────────────────────────────────────────────
# modules/pce_least_squares.py
# Least-squares PCE builder (Plotly + surrogate download; no AI insights)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import openturns as ot
import openturns.experimental as otexp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ╭──────────────────────── helpers ─────────────────────────────────────────╮
def _get_ot_model(model):
    """Return an ot.Function whatever the user passed."""
    if isinstance(model, ot.Function):
        return model
    n_in = model.__code__.co_argcount
    return ot.PythonFunction(n_in, 1, model)


class _SuperSample:
    def __init__(self, s: ot.Sample): self.s = s
    def to_py(self, var: str, indent: str = "") -> str:
        rows = [ "[" + ",".join(map(str, row)) + "]" for row in self.s ]
        return f"{indent}{var} = Sample([{','.join(rows)}])"


class _SuperDistribution:
    """Serialise a (possibly multivariate independent) OT distribution."""
    def __init__(self, d: ot.Distribution): self.d = d
    def to_py(self, var: str = "distribution") -> str:
        if self.d.getDimension() == 1:
            impl  = self.d.getImplementation()
            cls   = impl.getClassName()
            pars  = ",".join(map(str, impl.getParameter()))
            return f"{var} = {cls}({pars})"
        # independent joint
        code = "marginals = []\n"
        for i in range(self.d.getDimension()):
            m = _SuperDistribution(self.d.getMarginal(i)).to_py("")
            code += f"marginals.append({m.split('=')[-1].strip()})\n"
        code += f"{var} = JointDistribution(marginals)"
        return code


class _SuperChaosResult:
    """Turn a FunctionalChaosResult into a ready-to-use surrogate file."""
    def __init__(self, res: ot.FunctionalChaosResult): self.r = res

    def to_python(self) -> str:
        dist_code = _SuperDistribution(self.r.getDistribution()).to_py("distribution")
        coeffs    = _SuperSample(self.r.getCoefficients()).to_py("coefficients")
        indices   = list(self.r.getIndices())

        code  = "from openturns import *\n\n"
        code += "# --- Input distribution\n" + dist_code + "\n\n"
        code += "# --- PCE coefficients / basis indices\n"
        code += f"indices = {indices}\n{coeffs}\n\n"
        code += """# --- Build orthonormal basis and the surrogate
input_dim  = distribution.getDimension()
marginals  = [distribution.getMarginal(i) for i in range(input_dim)]
basis      = OrthogonalProductPolynomialFactory(marginals)
funcs      = [basis.build(i) for i in indices]
meta       = DualLinearCombinationFunction(funcs, coefficients)
measure    = basis.getMeasure()
transform  = DistributionTransformation(distribution, measure)
metaModel  = ComposedFunction(meta, transform)

def function_of_interest(X):
    return [ metaModel(X)[0] ]

model   = PythonFunction(input_dim, 1, function_of_interest)
problem = distribution  # ← required by the main app
"""
        return code
# ╰──────────────────────────────────────────────────────────────────────────╯


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
        else ot.PenalizedLeastSquaresAlgorithmFactory(),
    )
    algo = ot.FunctionalChaosAlgorithm(X, Y, dist, ot.FixedStrategy(basis, size), proj)
    algo.run()
    return algo.getResult()


# ─────────────────────────────────────────────────────────────────────────────
def pce_least_squares_analysis(
    model,
    problem,
    code_str: str | None = None,        # kept for signature consistency
    *,
    N_train: int = 1000,
    N_validate: int = 1000,
    basis_size_factor: float = 0.5,
    use_model_selection: bool = False,
    language_model: str | None = None,  # <─ ignored (no AI now)
    display_results: bool = True,
):
    """Fit LS-PCE, validate, plot with Plotly, return results dict."""
    f_ot = _get_ot_model(model)
    res  = _fit_pce(
        f_ot, problem,
        n_train=N_train,
        basis_frac=basis_size_factor,
        use_selection=use_model_selection,
    )

    # ── validation
    Xv = problem.getSample(N_validate)
    Yv = f_ot(Xv)
    Yh = res.getMetaModel()(Xv)
    r2 = ot.MetaModelValidation(Yv, Yh).computeR2Score()[0]
    r2_loo = otexp.FunctionalChaosValidation(
        res, ot.LeaveOneOutSplitter(N_train)
    ).computeR2Score()[0]

    # ── Sobol
    sens  = ot.FunctionalChaosSobolIndices(res)
    names = list(problem.getDescription())
    S1    = [sens.getSobolIndex(i)      for i in range(problem.getDimension())]
    ST    = [sens.getSobolTotalIndex(i) for i in range(problem.getDimension())]

    # ── Plotly figs
    figs = {}
    figs["validation"] = px.scatter(
        {"Model": [float(y[0]) for y in Yv], "PCE": [float(y[0]) for y in Yh]},
        x="Model",
        y="PCE",
        title=f"PCE validation — R² = {r2:.3f}",
    ).update_traces(marker=dict(size=6, opacity=0.7))

    figs["sobol"] = go.Figure([
        go.Bar(name="S1", x=names, y=S1),
        go.Bar(name="ST", x=names, y=ST),
    ]).update_layout(
        barmode="group",
        title="Sobol indices from PCE",
        yaxis_title="Sensitivity index",
    )

    # ── display
    if display_results:
        st.plotly_chart(figs["validation"], use_container_width=True, key="pce_val")
        st.plotly_chart(figs["sobol"],      use_container_width=True, key="pce_sob")

    # ── download surrogate
    surrogate_py = _SuperChaosResult(res).to_python()
    if display_results:
        st.download_button(
            "💾 Download Python surrogate",
            data=surrogate_py,
            file_name="pce_surrogate.py",
            mime="text/x-python",
            key="pce_dl",
        )

    # ── return
    return {
        "chaos_result": res,
        "r2": r2,
        "r2_loo": r2_loo,
        "sobol_S1": S1,
        "sobol_ST": ST,
        "figs": figs,
        "surrogate_py": surrogate_py,
    }


def display_pce_results(results: dict, container=None):
    if container is None:
        container = st
    if not results:
        return

    if "validation" in results["figs"]:
        container.plotly_chart(
            results["figs"]["validation"],
            use_container_width=True,
            key="pce_val_report",
        )
    if "sobol" in results["figs"]:
        container.plotly_chart(
            results["figs"]["sobol"],
            use_container_width=True,
            key="pce_sob_report",
        )

    # download button also in cached report
    if "surrogate_py" in results:
        container.download_button(
            "💾 Download Python surrogate",
            data=results["surrogate_py"],
            file_name="pce_surrogate.py",
            mime="text/x-python",
            key="pce_dl_report",
        )
