import streamlit as st

def initialize_session_state():
    """Initialize minimal required session state variables."""
    defaults = {
        "code": "",
        "morris_results": None,
        "fixed_values": {},
        "reduced_model": None,
        "model_file": '(Select or define your own model)',
        "simulation_results": None,
        "run_simulation": False,
        "markdown_output": None,
        "morris_analysis_done": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_analysis():
    """Reset analysis results."""
    st.session_state.morris_results = None
    st.session_state.fixed_values = {}
    st.session_state.reduced_model = None

def get_session_state(key, default=None):
    """Returns the value for the given key from session state, or the default value if the key doesn't exist."""
    return st.session_state.get(key, default)
