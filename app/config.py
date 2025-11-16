"""Application-level configuration for analyses."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import pandas as pd

from app.displays import (
    display_ancova_view,
    display_correlation_view,
    display_expectation_view,
    display_exploratory_view,
    display_fast_view,
    display_hsic_view,
    display_overall_summary,
    display_model_understanding_view,
    display_shapley_view,
    display_sobol_view,
    display_taylor_view,
)
from app.state import get_simulation_data
from modules.ancova_analysis import ancova_analysis
from modules.correlation_analysis import correlation_analysis
from modules.expectation_convergence_analysis import (
    compute_expectation_convergence_analysis,
    generate_ai_insights,
)
from modules.exploratory_data_analysis import exploratory_data_analysis
from modules.fast_analysis import fast_analysis
from modules.hsic_analysis import compute_hsic_analysis
from modules.ml_analysis import ml_analysis
from modules.model_understanding import model_understanding
from modules.monte_carlo import create_monte_carlo_dataframe
from modules.overall_analysis import generate_overall_summary
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis
from modules.taylor_analysis import taylor_analysis

AnalysisRunner = Callable[[Any, Any, str, str], Dict[str, Any]]


def _run_model_understanding(model, problem, model_code, language_model):
    return model_understanding(
        model,
        problem,
        model_code,
        language_model=language_model,
    )


def _run_overall_summary(
    model,
    problem,
    model_code,
    language_model,
    *,
    context_text,
    sources,
):
    return generate_overall_summary(
        model,
        problem,
        context_text=context_text,
        sources=sources,
        language_model=language_model,
    )


def _run_expectation_convergence(model, problem, model_code, language_model):
    results = compute_expectation_convergence_analysis(
        model, problem, model_code, N_samples=8000
    )
    insights = generate_ai_insights(results, language_model=language_model)
    results["ai_insights"] = insights
    fit_df = results.get("fit_df")
    if fit_df is not None and not isinstance(fit_df, pd.DataFrame):
        fit_df = pd.DataFrame(fit_df)
    if fit_df is not None:
        if "OT_Distribution" in fit_df.columns:
            fit_df = fit_df.drop(columns=["OT_Distribution"])
        results["fit_df"] = fit_df
    return results


def _run_exploratory_data_analysis(model, problem, model_code, language_model):
    sim_data = get_simulation_data()
    if sim_data is None:
        raise ValueError("Monte Carlo simulation data not available for EDA.")
    df = create_monte_carlo_dataframe(sim_data)
    return exploratory_data_analysis(
        df,
        1000,
        model,
        problem,
        model_code,
        language_model=language_model,
        display_results=False,
    )


def _run_sobol_analysis(model, problem, model_code, language_model):
    return sobol_sensitivity_analysis(
        1024,
        model,
        problem,
        model_code,
        language_model=language_model,
    )


def _run_fast_analysis(model, problem, model_code, language_model):
    return fast_analysis(
        model,
        problem,
        model_code_str=model_code,
        language_model=language_model,
        display_results=False,
    )


def _run_ancova_analysis(model, problem, model_code, language_model):
    return ancova_analysis(
        model,
        problem,
        model_code_str=model_code,
        language_model=language_model,
        display_results=False,
    )


def _run_taylor_analysis(model, problem, model_code, language_model):
    return taylor_analysis(
        model,
        problem,
        model_code,
        language_model=language_model,
        display_results=False,
    )


def _run_correlation_analysis(model, problem, model_code, language_model):
    return correlation_analysis(
        model,
        problem,
        model_code,
        size=1024,
        language_model=language_model,
        display_results=False,
    )


def _run_hsic_analysis(model, problem, model_code, language_model):
    return compute_hsic_analysis(
        hsic_size=200,
        model=model,
        problem=problem,
        model_code_str=model_code,
        language_model=language_model,
    )


def _run_shapley_analysis(model, problem, model_code, language_model):
    return ml_analysis(
        model,
        problem,
        size=1000,
        model_code_str=model_code,
        language_model=language_model,
        display_results=False,
    )


ANALYSIS_REGISTRY: List[Dict[str, Any]] = [
    {
        "key": "overall_summary",
        "name": "🚀 Overall Summary",
        "button_label": "🚀 Overall Summary",
        "run": _run_overall_summary,
        "display": display_overall_summary,
        "requires_all_results": True,
    },
    {
        "key": "model_understanding",
        "name": "Model Understanding",
        "button_label": "Model Understanding",
        "run": _run_model_understanding,
        "display": display_model_understanding_view,
    },
    {
        "key": "exploratory_data_analysis",
        "name": "Exploratory Data Analysis",
        "button_label": "Exploratory Data Analysis",
        "run": _run_exploratory_data_analysis,
        "display": display_exploratory_view,
    },
    {
        "key": "expectation_convergence",
        "name": "Expectation Convergence Analysis",
        "button_label": "Expectation Convergence",
        "run": _run_expectation_convergence,
        "display": display_expectation_view,
    },
    {
        "key": "correlation_analysis",
        "name": "Correlation Analysis",
        "button_label": "Correlation Analysis",
        "run": _run_correlation_analysis,
        "display": display_correlation_view,
    },
    {
        "key": "taylor_analysis",
        "name": "Taylor Analysis",
        "button_label": "Taylor Analysis (Local)",
        "run": _run_taylor_analysis,
        "display": display_taylor_view,
    },
    {
        "key": "hsic_analysis",
        "name": "HSIC Analysis",
        "button_label": "HSIC Analysis",
        "run": _run_hsic_analysis,
        "display": display_hsic_view,
    },
    {
        "key": "sobol_analysis",
        "name": "Sobol Analysis",
        "button_label": "Sobol Analysis (Variance)",
        "run": _run_sobol_analysis,
        "display": display_sobol_view,
    },
    {
        "key": "fast_analysis",
        "name": "FAST Analysis",
        "button_label": "FAST Analysis (Variance)",
        "run": _run_fast_analysis,
        "display": display_fast_view,
    },
    {
        "key": "ancova_analysis",
        "name": "ANCOVA Analysis",
        "button_label": "ANCOVA Analysis",
        "run": _run_ancova_analysis,
        "display": display_ancova_view,
    },
    {
        "key": "shapley_analysis",
        "name": "Shapley Analysis",
        "button_label": "Shapley Analysis (ML)",
        "run": _run_shapley_analysis,
        "display": display_shapley_view,
    },
]
