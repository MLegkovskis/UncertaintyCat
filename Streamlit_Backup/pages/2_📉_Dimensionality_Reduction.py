import streamlit as st

from app.components import render_app_shell, render_code_editor, render_sidebar_chat
from app.state import init_session_state
from modules.morris_analysis import dimensionality_reduction_page
from utils.core_utils import check_code_safety


def main():
    init_session_state()

    current_code, selected_language_model = render_app_shell()
    current_code = render_code_editor(current_code)

    if not current_code:
        st.info("Please select or upload a model file to perform dimensionality reduction.")
        return

    is_safe, safety_message = check_code_safety(current_code)
    if not is_safe:
        st.error(f"Security Error: {safety_message}")
        return

    try:
        eval_globals = {}
        exec(current_code, eval_globals)
        model = eval_globals.get("model")
        problem = eval_globals.get("problem")

        if not model or not problem:
            st.error("Model code must define 'model' and 'problem' variables.")
            return

        dimensionality_reduction_page(current_code, model, problem, selected_language_model)
    except Exception as e:
        st.error(f"Error evaluating model code for Dimensionality Reduction: {e}")

    render_sidebar_chat(current_code, selected_language_model)


if __name__ == "__main__":
    main()
