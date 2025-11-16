"""Utilities for building chat context from analysis results."""

from __future__ import annotations

from typing import Any, Dict

from modules.ancova_analysis import get_ancova_context_for_chat
from modules.expectation_convergence_analysis import (
    get_expectation_convergence_context_for_chat,
)
from modules.exploratory_data_analysis import get_eda_context_for_chat
from modules.fast_analysis import get_fast_context_for_chat
from modules.ml_analysis import get_ml_context_for_chat
from modules.sobol_sensitivity_analysis import get_sobol_context_for_chat
from modules.taylor_analysis import get_taylor_context_for_chat


CHAT_PREAMBLE = (
    "You are an expert assistant helping users understand a comprehensive uncertainty "
    "quantification and sensitivity analysis report."
)


def build_global_chat_context(results_dict: Dict[str, Any], model_code: str) -> str:
    """Create a combined markdown summary for the sidebar chat prompt."""
    analysis_names = list(results_dict.keys())
    summary = ", ".join(analysis_names) if analysis_names else "No analyses yet"
    context = f"""
{CHAT_PREAMBLE}

The report includes results from the following analyses: {summary}

The model being analyzed is defined as:
```python
{model_code}
```
""".strip()

    if results_dict:
        context += "\n\nThe following analyses have been performed:\n"

        if "Sobol Analysis" in results_dict:
            context += get_sobol_context_for_chat(results_dict["Sobol Analysis"])
        if "FAST Analysis" in results_dict:
            context += get_fast_context_for_chat(results_dict["FAST Analysis"])
        if "ANCOVA Analysis" in results_dict:
            context += get_ancova_context_for_chat(results_dict["ANCOVA Analysis"])
        if "Taylor Analysis" in results_dict:
            context += get_taylor_context_for_chat(results_dict["Taylor Analysis"])
        if "Correlation Analysis" in results_dict:
            corr_results = results_dict["Correlation Analysis"].get("all_correlation_results")
            if corr_results is not None:
                for output_name, corr_df in corr_results.items():
                    context += f"\n\n### Correlation Analysis Results for {output_name}\n"
                    context += corr_df.to_markdown()
        if "HSIC Analysis" in results_dict:
            hsic_results = results_dict["HSIC Analysis"].get("hsic_df")
            if hsic_results is not None:
                context += "\n\n### HSIC Sensitivity Analysis Results\n"
                context += hsic_results.to_markdown(index=False)
        if "Shapley Analysis" in results_dict:
            context += get_ml_context_for_chat(results_dict["Shapley Analysis"])
        if "Expectation Convergence Analysis" in results_dict:
            context += get_expectation_convergence_context_for_chat(
                results_dict["Expectation Convergence Analysis"]
            )
        if "Exploratory Data Analysis" in results_dict:
            context += get_eda_context_for_chat(results_dict["Exploratory Data Analysis"])

    return context
