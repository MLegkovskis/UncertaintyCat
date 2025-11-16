"""Application-level orchestration helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import streamlit as st

from app.chat_utils import build_global_chat_context
from app.config import ANALYSIS_REGISTRY
from app.state import (
    clear_all_results,
    get_all_results,
    get_simulation_data,
    set_analyses_completed,
    set_simulation_data,
    update_analysis_result,
)
from modules.monte_carlo import monte_carlo_simulation

ProgressCallback = Callable[[str, int, int], None]


@st.cache_resource(show_spinner=False)
def get_compiled_model(code_str: str) -> Tuple[Any, Any]:
    """Evaluate the provided code and return the ``model`` and ``problem`` objects."""
    eval_globals: dict[str, Any] = {}
    exec(code_str, eval_globals)
    model = eval_globals.get("model")
    problem = eval_globals.get("problem")
    if model is None or problem is None:
        raise ValueError("Model code must define 'model' and 'problem' variables.")
    return model, problem


def ensure_simulation_data(model: Any, problem: Any, sample_size: int = 1000) -> Any:
    """Run a Monte Carlo simulation once and cache in session state."""
    data = get_simulation_data()
    if data is None:
        data = monte_carlo_simulation(model, problem, sample_size)
        set_simulation_data(data)
    return data


def run_single_analysis(
    key: str,
    model: Any,
    problem: Any,
    model_code: str,
    language_model: str,
) -> dict:
    """Execute a single analysis defined in the registry."""
    entry = next((item for item in ANALYSIS_REGISTRY if item["key"] == key), None)
    if entry is None:
        raise ValueError(f"Unknown analysis key: {key}")

    set_analyses_completed(False)
    extra_kwargs: dict[str, Any] = {}
    if entry.get("requires_all_results"):
        snapshot = dict(get_all_results())
        context_text = build_global_chat_context(snapshot, model_code)
        extra_kwargs = {
            "context_text": context_text,
            "sources": list(snapshot.keys()),
        }

    try:
        results = entry["run"](model, problem, model_code, language_model, **extra_kwargs)
    except Exception as exc:
        results = {"error": str(exc)}
    update_analysis_result(entry["name"], results)
    return results


def run_all_analyses(
    model: Any,
    problem: Any,
    model_code: str,
    language_model: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Execute every analysis sequentially, optionally reporting progress."""
    clear_all_results()
    set_analyses_completed(False)

    immediate_entries = [e for e in ANALYSIS_REGISTRY if not e.get("requires_all_results")]
    deferred_entries = [e for e in ANALYSIS_REGISTRY if e.get("requires_all_results")]

    total = len(immediate_entries) + len(deferred_entries)
    step = 0

    for entry in immediate_entries:
        step += 1
        if progress_callback:
            progress_callback(entry["name"], step, total)
        run_single_analysis(entry["key"], model, problem, model_code, language_model)

    for entry in deferred_entries:
        step += 1
        if progress_callback:
            progress_callback(entry["name"], step, total)
        run_single_analysis(entry["key"], model, problem, model_code, language_model)

    set_analyses_completed(True)
