import os
import streamlit as st
import numpy as np
import openturns as ot
import pandas as pd

# Import modules
from modules.monte_carlo import monte_carlo_simulation, create_monte_carlo_dataframe
from modules.model_understanding import model_understanding
from modules.exploratory_data_analysis import exploratory_data_analysis
from modules.expectation_convergence_analysis import expectation_convergence_analysis_joint
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis
from modules.taylor_analysis import taylor_analysis
from modules.correlation_analysis import correlation_analysis
from modules.hsic_analysis import hsic_analysis
from modules.ml_analysis import ml_analysis

# Import utils
from utils.core_utils import (
    get_model_options,
    call_groq_api
)
from utils.model_utils import (
    validate_problem_structure,
    test_model,
    sample_inputs
)
from utils.constants import RETURN_INSTRUCTION

###############################################################################
# 1) SURROGATE DETECTION & SNIPPET EXTRACTION
###############################################################################
def is_surrogate_model(code_str: str) -> bool:
    """Returns True if the code snippet includes 'Y = metaModel(X)'."""
    return "Y = metaModel(X)" in code_str

def extract_surrogate_snippet(full_code: str) -> str:
    """
    If code is a surrogate, keep only lines from:
      def function_of_interest(X): ... up to ... model = function_of_interest
    Otherwise, return entire code.
    """
    if not is_surrogate_model(full_code):
        return full_code

    lines = full_code.splitlines(keepends=False)
    start_idx, end_idx = None, None

    for i, line in enumerate(lines):
        if "def function_of_interest(" in line:
            start_idx = i
            break

    for j in range(len(lines) - 1, -1, -1):
        if "model = function_of_interest" in lines[j]:
            end_idx = j
            break

    if (start_idx is None) or (end_idx is None) or (start_idx > end_idx):
        return full_code

    return "\n".join(lines[start_idx : end_idx + 1])

###############################################################################
# 2) LOAD MODEL CODE FROM EXAMPLES
###############################################################################
def load_model_code(selected_model_name: str) -> str:
    """
    Loads code from 'examples/' folder if a valid model is selected.
    Otherwise returns an empty string.
    """
    try:
        file_path = os.path.join('examples', selected_model_name)
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

###############################################################################
# 3) STREAMLIT APP START
###############################################################################
st.set_page_config(layout="wide")

col1, col2 = st.columns([1,4])
with col1:
    st.image("logo.jpg", width=100)
with col2:
    st.title("UncertaintyCat | v4.0")

###############################################################################
# 4) MODEL SELECT / UPLOAD
###############################################################################
# Insert placeholder item at index 0 for "no model selected"
dropdown_items = ["(Select or define your own model)"] + get_model_options()

# Variables to store current state
current_code = ""
simulation_results = None

col_select, col_upload = st.columns(2)

with col_select:
    selected_model = st.selectbox(
        "Select a Model File or Enter your own Model:",
        dropdown_items,
        index=0
    )
    
    if selected_model != "(Select or define your own model)":
        current_code = load_model_code(selected_model)

with col_upload:
    uploaded_file = st.file_uploader("or Choose a Python model file")
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        if st.button("Apply Uploaded File"):
            current_code = file_contents

###############################################################################
# 5) CODE EDITOR & SYNTAX-PREVIEW SIDE-BY-SIDE
###############################################################################
st.markdown("### Model Code Editor & Preview")

col_code, col_preview = st.columns(2)

with col_code:
    st.markdown("**Model Code Editor**")
    code_area_value = st.text_area(
        label="",
        value=current_code,
        height=300
    )
    # Update current code if changed in the editor
    current_code = code_area_value

with col_preview:
    st.markdown("**Syntax-Highlighted Preview (Read Only)**")
    if current_code.strip():
        st.code(current_code, language="python")
    else:
        st.info("No code to display.")

###############################################################################
# 6) LANGUAGE MODEL & ANALYSES
###############################################################################
groq_model_options = [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "qwen-2.5-32b",
    "deepseek-r1-distill-llama-70b"
]
selected_language_model = st.selectbox(
    "Select Language Model:",
    options=groq_model_options,
    index=0
)

st.markdown("### Select Analyses to Run")
analysis_options = {
    "Sobol Sensitivity Analysis": True,
    "Taylor Analysis": True,
    "Correlation Analysis": True,
    "HSIC Analysis": True,
    "SHAP Analysis": True
}
for key in analysis_options:
    analysis_options[key] = st.checkbox(key, value=True)

run_button = st.button("Run Simulation")

###############################################################################
# 7) MAIN SIMULATION LOGIC
###############################################################################
if run_button:
    if not current_code:
        st.error("Please provide a model first.")
    else:
        try:
            # Execute the model code in a fresh namespace
            local_namespace = {}
            exec(current_code, local_namespace)
            
            # Check that both model and problem are defined
            if 'model' not in local_namespace or 'problem' not in local_namespace:
                st.error("The model code must define both 'model' and 'problem'.")
            else:
                model = local_namespace['model']
                problem = local_namespace['problem']

                # Validate the problem structure
                validate_problem_structure(problem)

                # Get number of samples from UI
                N = 2000
                
                # Run Monte Carlo simulation
                results = monte_carlo_simulation(model, problem, N=N, seed=42)
                data = create_monte_carlo_dataframe(results)
                
                # Store results for use in analyses
                simulation_results = {
                    "data": data,
                    "model": model,
                    "problem": problem,
                    "code": current_code,
                    "selected_language_model": selected_language_model,
                    "N": N,
                    "analysis_options": analysis_options
                }
                
                # Display results
                st.markdown("---")
                st.header("Model Understanding")
                with st.spinner("Running Model Understanding..."):
                    snippet_for_modules = extract_surrogate_snippet(current_code)
                    model_understanding(
                        model,
                        problem,
                        snippet_for_modules,
                        language_model=selected_language_model
                    )

                st.markdown("---")
                st.header("Exploratory Data Analysis")
                with st.spinner("Running Exploratory Data Analysis..."):
                    exploratory_data_analysis(
                        data, N, model, problem, snippet_for_modules,
                        language_model=selected_language_model
                    )

                st.markdown("---")
                st.header("Expectation Convergence Analysis")
                with st.spinner("Running Expectation Convergence Analysis..."):
                    expectation_convergence_analysis_joint(
                        model, problem, snippet_for_modules,
                        language_model=selected_language_model
                    )

                if analysis_options["Sobol Sensitivity Analysis"]:
                    st.markdown("---")
                    st.header("Sobol Sensitivity Analysis")
                    with st.spinner("Running Sobol Sensitivity Analysis..."):
                        sobol_sensitivity_analysis(
                            1024, model, problem, snippet_for_modules,
                            language_model=selected_language_model
                        )

                if analysis_options["Taylor Analysis"]:
                    st.markdown("---")
                    st.header("Taylor Analysis")
                    with st.spinner("Running Taylor Analysis..."):
                        taylor_analysis(
                            model, problem, snippet_for_modules,
                            language_model=selected_language_model
                        )

                if analysis_options["Correlation Analysis"]:
                    st.markdown("---")
                    st.header("Correlation Analysis")
                    with st.spinner("Running Correlation Analysis..."):
                        correlation_analysis(
                            model, problem, snippet_for_modules,
                            language_model=selected_language_model
                        )

                if analysis_options["HSIC Analysis"]:
                    st.markdown("---")
                    st.header("HSIC Analysis")
                    with st.spinner("Running HSIC Analysis..."):
                        hsic_analysis(
                            model, problem, snippet_for_modules,
                            language_model=selected_language_model
                        )

                if analysis_options["SHAP Analysis"]:
                    st.markdown("---")
                    st.header("SHAP Analysis")
                    with st.spinner("Running SHAP Analysis..."):
                        ml_analysis(
                            data, problem, snippet_for_modules,
                            language_model=selected_language_model
                        )
                
        except Exception as e:
            st.error(f"Error during simulation: {str(e)}")
