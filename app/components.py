"""Reusable UI components for the Streamlit application."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import streamlit as st

from app.chat_utils import build_global_chat_context
from app.state import (
    analyses_completed,
    append_sidebar_chat_message,
    get_all_results,
    get_language_model,
    get_model_code,
    get_reliability_results,
    get_sidebar_chat_messages,
    get_selected_model_name,
    init_session_state,
    reset_analysis_state,
    set_language_model,
    set_model_code,
    set_selected_model_name,
)
from modules.monte_carlo import monte_carlo_simulation
from utils.core_utils import call_groq_api, check_code_safety, get_model_options
from utils.css_styles import load_css
from utils.model_loader import load_model_code


def render_app_shell(*, interactive: bool = True) -> Tuple[str, str]:
    """Render global UI chrome and return the current code and language model."""
    init_session_state()

    st.markdown(load_css(), unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .running-indicator {
            background-color: #f0f2f6;
            border-left: 5px solid #ff9800;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        .running-indicator:before {
            content: '';
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #ff9800;
            margin-right: 10px;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.4; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("UncertaintyCat | Version 5.50")
    st.caption("Advanced Uncertainty Quantification and Sensitivity Analysis Platform")
    st.info(
        "📌 **Tip:** This application works best in fullscreen mode to view all analysis tabs properly."
    )

    st.sidebar.image("logo.jpg", width=250)
    st.sidebar.header("Configuration Panel")
    st.sidebar.caption("Select or upload a model and choose an LLM for AI insights.")
    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    model_options = ["(Select or define your own model)"] + get_model_options()
    selected_option = st.sidebar.selectbox(
        "Select a Model File:",
        model_options,
        index=0,
        key="model_file_select",
        disabled=not interactive,
    )
    previous_selection = get_selected_model_name()
    current_code = get_model_code()

    if interactive:
        if selected_option == model_options[0]:
            if previous_selection:
                reset_analysis_state()
                set_selected_model_name("")
        elif selected_option != previous_selection:
            model_code = load_model_code(selected_option)
            reset_analysis_state()
            set_model_code(model_code)
            set_selected_model_name(selected_option)
            if "model_code_editor" in st.session_state:
                st.session_state.model_code_editor = model_code
            current_code = model_code

    uploaded_file = st.sidebar.file_uploader(
        "or Upload a Python Model File",
        type=["py"],
        key="uploaded_model_file",
        disabled=not interactive,
    )
    if interactive and uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        if st.sidebar.button("Apply Uploaded File", key="apply_uploaded_file"):
            reset_analysis_state()
            set_model_code(file_contents)
            set_selected_model_name("(Uploaded file)")
            if "model_code_editor" in st.session_state:
                st.session_state.model_code_editor = file_contents
            current_code = file_contents

    groq_model_options = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "qwen-2.5-32b",
        "deepseek-r1-distill-llama-70b",
    ]
    stored_language_model = get_language_model()
    default_index = (
        groq_model_options.index(stored_language_model)
        if stored_language_model in groq_model_options
        else 0
    )
    selected_language_model = st.sidebar.selectbox(
        "Select Language Model:",
        options=groq_model_options,
        index=default_index,
        key="language_model_select",
        disabled=not interactive,
    )
    if interactive:
        set_language_model(selected_language_model)
    else:
        selected_language_model = get_language_model()

    return get_model_code(), selected_language_model


def render_code_editor(current_code: str | None = None) -> str:
    """Render the model definition editor and return the updated code."""
    init_session_state()
    if current_code is None:
        current_code = get_model_code()

    st.header("Model Definition")
    if "model_code_editor" not in st.session_state:
        st.session_state.model_code_editor = current_code

    with st.expander("Model Code Editor & Preview", expanded=True):
        col_code, col_preview = st.columns(2)

        with col_code:
            st.subheader("Model Code Editor")
            st.caption(
                "Define your model using Python 3.12. You have access to numpy, scipy, and openturns libraries. "
                "Your code must define 'model' (an OpenTURNS Function) and 'problem' (an OpenTURNS Distribution)."
            )
            code_area_value = st.text_area(
                label=".",
                label_visibility="collapsed",
                height=300,
                key="model_code_editor",
            )
            if code_area_value != current_code:
                set_model_code(code_area_value)
                current_code = code_area_value

            if st.button("Validate Model Input", key="validate_model_input"):
                if not current_code:
                    st.error("Please provide a model first or select one of the example models to start with.")
                else:
                    try:
                        is_safe, safety_message = check_code_safety(current_code)
                        if not is_safe:
                            st.error(f"Security Error: {safety_message}")
                        else:
                            eval_globals: dict[str, object] = {}
                            exec(current_code, eval_globals)
                            model = eval_globals.get("model")
                            problem = eval_globals.get("problem")

                            if not model or not problem:
                                st.error("Model code must define 'model' and 'problem' variables.")
                            else:
                                reset_analysis_state()
                                set_model_code(current_code)
                                with st.spinner("Running 10 Monte Carlo samples..."):
                                    try:
                                        results = monte_carlo_simulation(model, problem, N=10, seed=42)
                                        mean_value = results["mean"]
                                        std_value = results["std"]
                                        if isinstance(mean_value, np.ndarray):
                                            mean_str = f"{mean_value[0]:.4f}"
                                            std_str = f"{std_value[0]:.4f}"
                                        else:
                                            mean_str = f"{mean_value:.4f}"
                                            std_str = f"{std_value:.4f}"
                                        st.success(
                                            f"Model validated and state reset successfully! Sample mean: {mean_str}, std: {std_str}"
                                        )
                                    except Exception as exc:
                                        st.error(f"Error running model: {exc}")
                    except Exception as exc:
                        st.error(f"Error evaluating model code: {exc}")

        with col_preview:
            st.subheader("Syntax-Highlighted Preview")
            if current_code.strip():
                st.code(current_code, language="python")
            else:
                st.info("No code to display. Please select or upload a model.")
    return current_code


def render_sidebar_chat(current_code: str | None, selected_language_model: str | None) -> None:
    """Render the sidebar chat assistant."""
    init_session_state()
    current_code = current_code or get_model_code()
    selected_language_model = selected_language_model or get_language_model()

    st.sidebar.header("Chat about Results")
    all_results_data = get_all_results()
    reliability_result = get_reliability_results()
    if all_results_data or reliability_result:
        result_names = list(all_results_data.keys())
        is_full_run = analyses_completed()
        if reliability_result and not result_names:
            info_message = "Ask questions about your reliability analysis."
        elif is_full_run and result_names:
            info_message = "Ask questions about the full analysis suite."
        elif len(result_names) == 1:
            info_message = f"Ask questions about your '{result_names[0]}' results."
        else:
            info_message = f"Ask questions about your {len(result_names)} run analyses."
        st.sidebar.info(info_message)
        messages = get_sidebar_chat_messages()
        for message in messages:
            with st.sidebar.chat_message(message["role"]):
                st.sidebar.write(message["content"])

        sidebar_prompt = st.sidebar.chat_input(
            "Ask about your analysis results...", key="sidebar_chat_input"
        )
        if sidebar_prompt:
            append_sidebar_chat_message("user", sidebar_prompt)
            context = build_global_chat_context(
                all_results_data, current_code, reliability_result
            )
            chat_history = ""
            history_messages = get_sidebar_chat_messages()[:-1]
            if history_messages:
                chat_history = "Previous conversation:\n"
                for msg in history_messages:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n\n"
            chat_prompt = (
                f"{context}\n\n{chat_history}\n\nCurrent user question: {sidebar_prompt}\n\n"
                "Please provide a helpful, accurate response to this question."
            )
            with st.spinner("Thinking..."):
                try:
                    response_text = call_groq_api(
                        chat_prompt, model_name=selected_language_model
                    )
                except Exception as exc:
                    st.sidebar.error(f"Error calling API: {exc}")
                    response_text = (
                        "I'm sorry, I encountered an error while processing your question. "
                        "Please try again."
                    )
            append_sidebar_chat_message("assistant", response_text)
            st.rerun()
    else:
        st.sidebar.warning("Chat will be available after you run at least one analysis.")
