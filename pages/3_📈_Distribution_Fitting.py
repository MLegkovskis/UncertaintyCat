import streamlit as st

from app.components import render_app_shell, render_code_editor, render_sidebar_chat
from app.state import init_session_state
from modules.distribution_fitting import distribution_fitting_page


def main():
    init_session_state()

    current_code, selected_language_model = render_app_shell()
    current_code = render_code_editor(current_code)

    distribution_fitting_page()

    render_sidebar_chat(current_code, selected_language_model)


if __name__ == "__main__":
    main()
