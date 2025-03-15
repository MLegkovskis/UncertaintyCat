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
from modules.morris_analysis import morris_analysis

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

# Create pages
pages = {
    "Main Analysis": "Run standard uncertainty analysis",
    "Dimensionality Reduction": "Reduce model complexity by identifying non-influential variables"
}
selected_page = st.sidebar.radio("Select Page", list(pages.keys()))
st.sidebar.write(pages[selected_page])

###############################################################################
# 4) MODEL SELECT / UPLOAD
###############################################################################
# Insert placeholder item at index 0 for "no model selected"
dropdown_items = ["(Select or define your own model)"] + get_model_options()

# Variables to store current state
current_code = ""
simulation_results = None
model = None
problem = None
code_snippet = None

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
        if st.button("Apply Uploaded File", key="apply_uploaded_file"):
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

###############################################################################
# 7) PAGE-SPECIFIC CONTENT
###############################################################################
if selected_page == "Main Analysis":
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

    run_button = st.button("Run Simulation", key="run_main_simulation")

    ###############################################################################
    # 8) MAIN SIMULATION LOGIC
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
                        model_understanding(
                            model,
                            problem,
                            current_code,
                            language_model=selected_language_model
                        )

                    st.markdown("---")
                    st.header("Exploratory Data Analysis")
                    with st.spinner("Running Exploratory Data Analysis..."):
                        exploratory_data_analysis(
                            data, N, model, problem, current_code,
                            language_model=selected_language_model
                        )

                    st.markdown("---")
                    st.header("Expectation Convergence Analysis")
                    with st.spinner("Running Expectation Convergence Analysis..."):
                        expectation_convergence_analysis_joint(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )

                    if analysis_options["Sobol Sensitivity Analysis"]:
                        st.markdown("---")
                        st.header("Sobol Sensitivity Analysis")
                        with st.spinner("Running Sobol Sensitivity Analysis..."):
                            sobol_sensitivity_analysis(
                                1024, model, problem, current_code,
                                language_model=selected_language_model
                            )

                    if analysis_options["Taylor Analysis"]:
                        st.markdown("---")
                        st.header("Taylor Analysis")
                        with st.spinner("Running Taylor Analysis..."):
                            taylor_analysis(
                                model, problem, current_code,
                                language_model=selected_language_model
                            )

                    if analysis_options["Correlation Analysis"]:
                        st.markdown("---")
                        st.header("Correlation Analysis")
                        with st.spinner("Running Correlation Analysis..."):
                            correlation_analysis(
                                model, problem, current_code,
                                language_model=selected_language_model
                            )

                    if analysis_options["HSIC Analysis"]:
                        st.markdown("---")
                        st.header("HSIC Analysis")
                        with st.spinner("Running HSIC Analysis..."):
                            hsic_analysis(
                                model, problem, current_code,
                                language_model=selected_language_model
                            )

                    if analysis_options["SHAP Analysis"]:
                        st.markdown("---")
                        st.header("SHAP Analysis")
                        with st.spinner("Running SHAP Analysis..."):
                            ml_analysis(
                                data, problem, current_code,
                                language_model=selected_language_model
                            )
                    
            except Exception as e:
                st.error(f"Error during simulation: {str(e)}")

elif selected_page == "Dimensionality Reduction":
    st.title("Dimensionality Reduction with Morris Method")
    
    # First, check if we have a model from the code editor
    if not current_code and model is None:
        st.info("Please define your model in the Model Definition page first.")
        st.markdown("""
        ### How to use the Morris Analysis:
        
        1. First, go to the **Model Definition** page
        2. Define or load your model and problem
        3. Return to this page to run the Morris analysis
        
        The Morris method helps identify which input variables have minimal impact on your model's output.
        This allows you to create simplified models by fixing non-influential variables at nominal values.
        """)
    else:
        # If we have code but no model yet, try to execute it
        if current_code and model is None:
            try:
                # Execute the model code in a fresh namespace
                local_namespace = {}
                exec(current_code, local_namespace)
                
                # Check that both model and problem are defined
                if 'model' in local_namespace and 'problem' in local_namespace:
                    model = local_namespace['model']
                    problem = local_namespace['problem']
                    
                    # Run Morris analysis
                    morris_analysis(model, problem, current_code, selected_language_model)
                else:
                    st.error("The model code must define both 'model' and 'problem'.")
            except Exception as e:
                st.error(f"Error executing model code: {str(e)}")
        # If we already have a model, just run the analysis
        elif model is not None:
            try:
                # Run Morris analysis
                morris_analysis(model, problem, current_code, selected_language_model)
            except Exception as e:
                st.error(f"Error during Morris analysis: {str(e)}")
