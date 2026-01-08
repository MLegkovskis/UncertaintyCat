"""Centralized Streamlit session state helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

DEFAULT_LANGUAGE_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_STATE_FACTORIES = {
    "analyses_ran": lambda: False,
    "simulation_data": lambda: None,
    "all_results": dict,
    "global_chat_messages": list,
    "sidebar_global_chat_messages": list,
    "model_code": lambda: "",
    "language_model": lambda: DEFAULT_LANGUAGE_MODEL,
    "selected_model_name": lambda: "",
    "pce_chaos_result": lambda: None,
    "pce_build_results": lambda: None,
    "pce_diag_results": dict,
    "reliability_results": lambda: None,
}


def init_session_state() -> None:
    """Ensure all known keys exist in ``st.session_state``."""
    for key, factory in _STATE_FACTORIES.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


def get_model_code() -> str:
    init_session_state()
    return st.session_state.model_code


def set_model_code(code: str) -> None:
    init_session_state()
    st.session_state.model_code = code


def get_selected_model_name() -> str:
    init_session_state()
    return st.session_state.selected_model_name


def set_selected_model_name(name: str) -> None:
    init_session_state()
    st.session_state.selected_model_name = name


def get_language_model() -> str:
    init_session_state()
    return st.session_state.language_model or DEFAULT_LANGUAGE_MODEL


def set_language_model(model_name: str) -> None:
    init_session_state()
    st.session_state.language_model = model_name


def get_all_results() -> Dict[str, Any]:
    init_session_state()
    return st.session_state.all_results


def update_analysis_result(name: str, data: Dict[str, Any]) -> None:
    init_session_state()
    st.session_state.all_results[name] = data


def clear_all_results() -> None:
    init_session_state()
    st.session_state.all_results = {}
    st.session_state.analyses_ran = False


def get_analysis_result(name: str) -> Optional[Dict[str, Any]]:
    return get_all_results().get(name)


def get_simulation_data() -> Any:
    init_session_state()
    return st.session_state.simulation_data


def set_simulation_data(data: Any) -> None:
    init_session_state()
    st.session_state.simulation_data = data


def analyses_completed() -> bool:
    init_session_state()
    return bool(st.session_state.analyses_ran)


def set_analyses_completed(value: bool) -> None:
    init_session_state()
    st.session_state.analyses_ran = value


def get_sidebar_chat_messages() -> List[Dict[str, str]]:
    init_session_state()
    return st.session_state.sidebar_global_chat_messages


def append_sidebar_chat_message(role: str, content: str) -> None:
    init_session_state()
    st.session_state.sidebar_global_chat_messages.append(
        {"role": role, "content": content}
    )


def clear_sidebar_chat() -> None:
    init_session_state()
    st.session_state.sidebar_global_chat_messages = []


def get_pce_chaos_result() -> Any:
    init_session_state()
    return st.session_state.pce_chaos_result


def set_pce_chaos_result(result: Any) -> None:
    init_session_state()
    st.session_state.pce_chaos_result = result
    st.session_state.pce_diag_results = {}


def get_pce_build_results() -> Optional[Dict[str, Any]]:
    init_session_state()
    return st.session_state.pce_build_results


def set_pce_build_results(results: Optional[Dict[str, Any]]) -> None:
    init_session_state()
    st.session_state.pce_build_results = results


def get_pce_diag_results() -> Dict[str, Any]:
    init_session_state()
    return st.session_state.pce_diag_results


def update_pce_diag_result(key: str, data: Any) -> None:
    init_session_state()
    st.session_state.pce_diag_results[key] = data


def get_reliability_results() -> Any:
    init_session_state()
    return st.session_state.reliability_results


def set_reliability_results(data: Any) -> None:
    init_session_state()
    st.session_state.reliability_results = data
