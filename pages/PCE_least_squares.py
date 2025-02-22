import os
import streamlit as st
import openturns as ot
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from modules.code_safety import check_code_safety
from modules.pce_least_squares_utils import pce_sobol, reset_pce_least_squares_results
from modules.model_options_list import model_options
from modules.session_state_utils import initialize_session_state, reset_analysis

# Initialize session state
initialize_session_state()

# Page setup
st.title("Polynomial Chaos Expansion - Least Squares")
st.markdown("""
This page performs Polynomial Chaos Expansion (PCE) using the Least Squares method:
1. First, it creates a PCE metamodel using training data
2. Then, it validates the metamodel using test data
3. Finally, it computes and displays Sobol sensitivity indices
""")

# Model selection
def load_model_code(filename):
    """Load code from examples folder."""
    try:
        with open(f'examples/{filename}', 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

# Model dropdown
model_options_list = ["(Select or define your own model)"] + model_options
selected_model = st.selectbox(
    "Select a Model File:",
    model_options_list,
    key="model_file",
    on_change=reset_analysis
)

# Get model code
if selected_model != "(Select or define your own model)":
    st.session_state.code = load_model_code(selected_model)
    st.code(st.session_state.code, language='python')
else:
    st.session_state.code = st.text_area(
        "Or paste your model code here:",
        value=st.session_state.code,
        height=300,
        on_change=reset_analysis
    )

# Execute code if present
if st.session_state.code:
    if not check_code_safety(st.session_state.code):
        st.error("Code contains unsafe operations. Please review and modify.")
    else:
        try:
            # Execute code to get model and problem
            local_dict = {}
            exec(st.session_state.code, globals(), local_dict)
            
            if 'model' not in local_dict or 'problem' not in local_dict:
                st.error("Code must define 'model' and 'problem' variables.")
            else:
                # PCE parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    degree = st.number_input("Maximum polynomial degree", min_value=1, max_value=15, value=3)
                with col2:
                    n_train = st.number_input("Training samples", min_value=10, max_value=10000, value=100)
                with col3:
                    n_validate = st.number_input("Validation samples", min_value=10, max_value=10000, value=1000)

                # Run PCE analysis
                if st.button("Run PCE Analysis"):
                    with st.spinner("Running PCE Analysis..."):
                        pce_sobol(
                            train_sample_size=n_train,
                            validation_sample_size=n_validate,
                            model=local_dict['model'],
                            problem=local_dict['problem'],
                            model_code_str=st.session_state.code,
                            language_model="llama-3.3-70b-versatile",
                            basis_size_factor=1.0,
                            use_model_selection=True
                        )

        except Exception as e:
            st.error(f"Error executing code: {e}")
