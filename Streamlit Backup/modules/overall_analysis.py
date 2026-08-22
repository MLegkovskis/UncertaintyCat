"""Meta-analysis module that synthesizes existing results into a summary."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from utils.constants import RETURN_INSTRUCTION
from utils.core_utils import call_groq_api


def generate_overall_summary(
    model: Any,
    problem: Any,
    *,
    context_text: str,
    sources: Iterable[str],
    language_model: str,
) -> Dict[str, Any]:
    """Use the LLM to create a high-level summary of available analyses."""
    source_list = list(sources)
    if not source_list:
        return {
            "summary_text": "No analysis results found. Run one or more modules to generate a summary.",
            "sources": [],
        }

    prompt = f"""
{RETURN_INSTRUCTION}

You are an enterprise-grade uncertainty quantification expert. Synthesize the
following analysis reports into a concise executive summary:

{context_text}

Provide 2-3 well-structured paragraphs capturing key findings, dominant
uncertainties, and recommended next actions.
""".strip()

    try:
        summary_text = call_groq_api(prompt, model_name=language_model)
    except Exception as exc:
        summary_text = f"Error generating summary: {exc}"

    return {
        "summary_text": summary_text,
        "sources": source_list,
    }
