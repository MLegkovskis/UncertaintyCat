import streamlit as st

def initialize_session_state():
    """Initializes the required keys in Streamlit session state."""
    # By default, no model is selected
    if 'model_file' not in st.session_state:
        st.session_state.model_file = '(Select or define your own model)'
    # By default, the code area is empty
    if 'code' not in st.session_state:
        st.session_state.code = ''
    # Store the simulation results
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    # Flag to track if the user clicked 'Run Simulation'
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    # Will store the AI-based interpretation
    if 'markdown_output' not in st.session_state:
        st.session_state.markdown_output = None

def reset_analysis_results():
    """Resets the analysis results in session state."""
    keys_to_reset = [
        'correlation_response_markdown', 'correlation_fig',
        'expectation_response_markdown', 'expectation_fig',
        'eda_response_markdown', 'eda_fig',
        'hsic_response_markdown', 'hsic_fig',
        'ml_response_markdown', 'ml_shap_summary_fig', 'ml_dependence_fig',
        'model_understanding_response_markdown',
        'sobol_response_markdown', 'sobol_fig',
        'taylor_response_markdown', 'taylor_fig',
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def get_session_state(key, default=None):
    """Returns the value for the given key from session state, or the default value if the key doesn't exist."""
    return st.session_state.get(key, default)
