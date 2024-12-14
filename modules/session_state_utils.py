import streamlit as st


def initialize_session_state():
    """Initializes the required keys in Streamlit session state."""
    if "model_file" not in st.session_state:
        st.session_state.model_file = "Beam.py"
    if "code_editor" not in st.session_state:
        st.session_state.code_editor = open(
            "examples/" + st.session_state.model_file
        ).read()
    if "code_editor_counter" not in st.session_state:
        st.session_state.code_editor_counter = 0
    if "simulation_results" not in st.session_state:
        st.session_state.simulation_results = None
    if "run_simulation" not in st.session_state:
        st.session_state.run_simulation = False
    if "code_updated" not in st.session_state:
        st.session_state.code_updated = False
    if "markdown_output" not in st.session_state:
        st.session_state.markdown_output = None


def reset_analysis_results():
    """Resets the analysis results in session state."""
    keys_to_reset = [
        "correlation_response_markdown",
        "correlation_fig",
        "expectation_response_markdown",
        "expectation_fig",
        "eda_response_markdown",
        "eda_fig",
        "hsic_response_markdown",
        "hsic_fig",
        "ml_response_markdown",
        "ml_shap_summary_fig",
        "ml_dependence_fig",
        "model_understanding_response_markdown",
        "sobol_response_markdown",
        "sobol_fig",
        "taylor_response_markdown",
        "taylor_fig",
        "pce_validation_fig",
        "pce_sobol_fig",
        "pce_treemap_fig",
        "pce_sobol_markdown",
        "pce_sobol_response_markdown",
        "pce_sobol_radial_fig",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def get_session_state(key, default=None):
    """Returns the value for the given key from session state, or the default value if the key doesn't exist."""
    return st.session_state.get(key, default)
