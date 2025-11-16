import streamlit as st

from app.components import render_app_shell, render_sidebar_chat
from app.state import init_session_state


st.set_page_config(
    page_title="UncertaintyCat | UQ Made Easy",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

current_code, selected_language_model = render_app_shell(interactive=False)
render_sidebar_chat(current_code, selected_language_model)

st.title("Welcome to UncertaintyCat")
st.caption("Advanced Uncertainty Quantification and Sensitivity Analysis Platform")

st.info(
    "Use the Streamlit sidebar to choose an analysis workspace. The multipage navigation"
    " provides dedicated views for the UQ Dashboard, Dimensionality Reduction, and Distribution"
    " Fitting workflows."
)

st.write(
    "Select a page from the left to begin exploring models, running analyses, and generating"
    " comprehensive reports."
)
