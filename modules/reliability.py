"""Reliability analysis helpers (FORM, SORM, simulation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import openturns as ot
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

_OPERATOR_MAP = {
    ">": ot.Greater(),
    "<": ot.Less(),
    ">=": ot.GreaterOrEqual(),
    "<=": ot.LessOrEqual(),
}


@dataclass
class ReliabilityResult:
    method: str
    probability: float
    reliability_index: Optional[float] = None
    cov: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    evaluations: Optional[int] = None
    design_point: Optional[pd.DataFrame] = None
    importance_factors: Optional[pd.DataFrame] = None
    convergence_sample: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "probability": self.probability,
            "reliability_index": self.reliability_index,
            "cov": self.cov,
            "confidence_interval": self.confidence_interval,
            "evaluations": self.evaluations,
            "design_point": self.design_point,
            "importance_factors": self.importance_factors,
            "convergence_sample": self.convergence_sample,
        }


def _build_event(model: ot.Function, problem: ot.Distribution, operator: str, threshold: float) -> ot.Event:
    input_vector = ot.RandomVector(problem)
    output_vector = ot.CompositeRandomVector(model, input_vector)
    comparator = _OPERATOR_MAP.get(operator, ot.Greater())
    return ot.ThresholdEvent(output_vector, comparator, threshold)


def _importance_dataframe(result: ot.FORMResult, input_names: list[str]) -> pd.DataFrame:
    try:
        factors = list(result.getImportanceFactors())
    except Exception:
        return None
    data = {
        "Variable": input_names,
        "Importance": factors,
    }
    return pd.DataFrame(data).sort_values("Importance", ascending=False).reset_index(drop=True)


def _design_point_dataframe(result: ot.FORMResult, input_names: list[str]) -> pd.DataFrame:
    try:
        design = list(result.getPhysicalSpaceDesignPoint())
    except Exception:
        return None
    data = {
        "Variable": input_names,
        "Design Point": design,
    }
    return pd.DataFrame(data)


def _graph_to_sample(graph: ot.Graph) -> Optional[pd.DataFrame]:
    try:
        drawable = graph.getDrawable(0)
        data = drawable.getData()
    except Exception:
        return None
    if data is None or data.getSize() == 0:
        return None
    array = [[pt[i] for i in range(data.getDimension())] for pt in data]
    columns = ["Sampling", "Probability"] if data.getDimension() >= 2 else ["Value"]
    return pd.DataFrame(array, columns=columns[: data.getDimension()])


def compute_output_distribution_plot(
    model: ot.Function,
    problem: ot.Distribution,
    threshold: float,
    operator_str: str,
    sample_size: int = 1000,
) -> go.Figure:
    """Sample model output and plot histogram + KDE with threshold."""
    input_sample = problem.getSample(sample_size)
    output_sample = model(input_sample)
    data = np.array(output_sample).flatten()
    if data.size == 0:
        data = np.array([0.0])

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data,
            histnorm="probability density",
            name="Sample PDF",
            marker_color="#1f77b4",
            opacity=0.6,
        )
    )

    try:
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde(x_range),
                mode="lines",
                name="Estimated PDF",
                line=dict(color="#1f77b4", width=2),
            )
        )
    except Exception:
        pass

    fig.add_vline(
        x=threshold,
        line_width=3,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="Threshold",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Output Distribution (N={sample_size})",
        xaxis_title="Model Output",
        yaxis_title="Density",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def format_probability(probability: float) -> str:
    """Return user-friendly odds string."""
    if probability is None:
        return "—"
    if probability <= 0:
        return "0 (Impossible)"
    if probability >= 1:
        return "1.0 (Certain)"
    one_in = 1.0 / probability
    if one_in >= 1000:
        return f"1 in {int(one_in):,}"
    return f"{probability:.2%}"


def run_reliability_analysis(
    model: ot.Function,
    problem: ot.Distribution,
    method: str,
    threshold: float,
    operator: str = ">",
    max_iter: int = 10_000,
    target_cov: float = 0.05,
) -> ReliabilityResult:
    event = _build_event(model, problem, operator, threshold)
    input_names = [problem.getDescription()[i] if problem.getDescription()[i] else f"X{i+1}" for i in range(problem.getDimension())]

    method_upper = method.upper()

    start_point = problem.getMean()

    if method_upper == "FORM":
        optimizer = ot.Cobyla()
        optimizer.setMaximumCallsNumber(10000)
        optimizer.setMaximumAbsoluteError(1e-10)
        optimizer.setMaximumRelativeError(1e-10)
        optimizer.setMaximumResidualError(1e-10)
        optimizer.setMaximumConstraintError(1e-10)
        if start_point.getDimension() == problem.getDimension():
            optimizer.setStartingPoint(start_point)
        algo = ot.FORM(optimizer, event)
        algo.run()
        form_res = algo.getResult()
        importance_df = _importance_dataframe(form_res, input_names)
        design_df = _design_point_dataframe(form_res, input_names)
        return ReliabilityResult(
            method="FORM",
            probability=form_res.getEventProbability(),
            reliability_index=form_res.getHasoferReliabilityIndex(),
            design_point=design_df,
            importance_factors=importance_df,
        )

    if method_upper == "SORM":
        optimizer = ot.Cobyla()
        optimizer.setMaximumCallsNumber(10000)
        optimizer.setMaximumAbsoluteError(1e-10)
        optimizer.setMaximumRelativeError(1e-10)
        optimizer.setMaximumResidualError(1e-10)
        optimizer.setMaximumConstraintError(1e-10)
        if start_point.getDimension() == problem.getDimension():
            optimizer.setStartingPoint(start_point)
        algo = ot.SORM(optimizer, event)
        algo.run()
        sorm_res = algo.getResult()
        return ReliabilityResult(
            method="SORM",
            probability=sorm_res.getEventProbabilityBreitung(),
            reliability_index=sorm_res.getGeneralisedReliabilityIndexBreitung(),
        )

    if method_upper == "DIRECTIONAL SAMPLING":
        algo = ot.DirectionalSampling(event)
        algo.setMaximumOuterSampling(max_iter)
        algo.setMaximumCoefficientOfVariation(target_cov)
        algo.run()
        sim_res = algo.getResult()
        graph = algo.drawProbabilityConvergence()
        convergence_df = _graph_to_sample(graph)
        return ReliabilityResult(
            method="Directional Sampling",
            probability=sim_res.getProbabilityEstimate(),
            reliability_index=sim_res.getGeneralisedReliabilityIndex(),
            cov=sim_res.getCoefficientOfVariation(),
            evaluations=sim_res.getOuterSampling(),
            convergence_sample=convergence_df,
        )

    # default Monte Carlo
    experiment = ot.MonteCarloExperiment()
    algo = ot.ProbabilitySimulationAlgorithm(event, experiment)
    algo.setMaximumOuterSampling(max_iter)
    algo.setMaximumCoefficientOfVariation(target_cov)
    algo.run()
    mc_res = algo.getResult()
    convergence_df = _graph_to_sample(algo.drawProbabilityConvergence())
    ci_half = mc_res.getConfidenceLength()
    probability = mc_res.getProbabilityEstimate()
    confidence_interval = None
    if ci_half is not None:
        confidence_interval = (probability - 0.5 * ci_half, probability + 0.5 * ci_half)
    return ReliabilityResult(
        method="Monte Carlo",
        probability=probability,
        reliability_index=mc_res.getGeneralisedReliabilityIndex(),
        cov=mc_res.getCoefficientOfVariation(),
        evaluations=mc_res.getOuterSampling(),
        confidence_interval=confidence_interval,
        convergence_sample=convergence_df,
    )
