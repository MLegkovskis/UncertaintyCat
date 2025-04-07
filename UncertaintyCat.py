import os
import streamlit as st
import numpy as np
import openturns as ot
import pandas as pd
import plotly.express as px
import re
import traceback

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
from modules.morris_analysis import morris_analysis, dimensionality_reduction_page
from modules.fast_analysis import fast_analysis
from modules.ancova_analysis import ancova_analysis
from modules.distribution_fitting import distribution_fitting_page, run_distribution_fitting_analysis

# Import utils
from utils.core_utils import (
    get_model_options,
    call_groq_api,
    check_code_safety
)
from utils.model_utils import (
    validate_problem_structure,
    test_model,
    sample_inputs
)
from utils.constants import RETURN_INSTRUCTION
from utils.css_styles import load_css

# Initialize session state variables
if 'analyses_ran' not in st.session_state:
    st.session_state.analyses_ran = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'model_understanding_ran' not in st.session_state:
    st.session_state.model_understanding_ran = False
if 'eda_ran' not in st.session_state:
    st.session_state.eda_ran = False
if 'expectation_convergence_ran' not in st.session_state:
    st.session_state.expectation_convergence_ran = False
if 'sobol_ran' not in st.session_state:
    st.session_state.sobol_ran = False
if 'fast_ran' not in st.session_state:
    st.session_state.fast_ran = False
if 'ancova_ran' not in st.session_state:
    st.session_state.ancova_ran = False
if 'taylor_ran' not in st.session_state:
    st.session_state.taylor_ran = False
if 'correlation_ran' not in st.session_state:
    st.session_state.correlation_ran = False
if 'hsic_ran' not in st.session_state:
    st.session_state.hsic_ran = False
if 'shap_ran' not in st.session_state:
    st.session_state.shap_ran = False
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'fitted_distributions' not in st.session_state:
    st.session_state.fitted_distributions = {}
if 'selected_distributions' not in st.session_state:
    st.session_state.selected_distributions = {}
if 'problem_distribution' not in st.session_state:
    st.session_state.problem_distribution = None

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

# Page configuration
st.set_page_config(
    page_title="UncertaintyCat | UQ Made Easy",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and apply custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Add CSS for the custom status indicator
st.markdown("""
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
    0% {
        opacity: 1;
    }
    50% {
        opacity: 0.4;
    }
    100% {
        opacity: 1;
    }
}
</style>
""", unsafe_allow_html=True)

# Function to show immediate visual feedback
def show_running_indicator(container, analysis_name):
    container.markdown(f'<div class="running-indicator">Running {analysis_name}...</div>', unsafe_allow_html=True)

# Header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.jpg", width=100)
with col2:
    st.markdown('<h1 class="main-header">UncertaintyCat | Version 5.20</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #7F8C8D;">Advanced Uncertainty Quantification and Sensitivity Analysis Platform</p>', unsafe_allow_html=True)

# Fullscreen recommendation
st.info("📌 **Tip:** This application works best in fullscreen mode to view all analysis tabs properly.")

# Sidebar styling and navigation
st.sidebar.markdown('<div style="text-align: center; padding: 10px 0;"><h3>Navigation Panel</h3></div>', unsafe_allow_html=True)

# Create pages with icons
pages = {
    "📊 Main Analysis": "Comprehensive uncertainty quantification and sensitivity analysis",
    "📉 Dimensionality Reduction": "Reduce model complexity by identifying non-influential variables",
    "📈 Distribution Fitting": "Fit probability distributions to your data for UQ analysis"
}
selected_page = st.sidebar.radio("", list(pages.keys()))
st.sidebar.markdown(f'<div class="info-box status-box">{pages[selected_page.strip()]}</div>', unsafe_allow_html=True)

# Sidebar divider
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

###############################################################################
# 4) MODEL SELECT / UPLOAD
###############################################################################
st.sidebar.markdown('<h3>Model Configuration</h3>', unsafe_allow_html=True)

# Insert placeholder item at index 0 for "no model selected"
dropdown_items = ["(Select or define your own model)"] + get_model_options()

# Variables to store current state
current_code = ""
simulation_results = None
model = None
problem = None
code_snippet = None

# Model selection in sidebar
selected_model = st.sidebar.selectbox(
    "Select a Model File:",
    dropdown_items,
    index=0
)

if selected_model != "(Select or define your own model)":
    current_code = load_model_code(selected_model)

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("or Upload a Python Model File")
if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    if st.sidebar.button("Apply Uploaded File", key="apply_uploaded_file"):
        current_code = file_contents

# LLM model selection in sidebar
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<h3>AI Configuration</h3>', unsafe_allow_html=True)

groq_model_options = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
    "qwen-2.5-32b",
    "deepseek-r1-distill-llama-70b"
]
selected_language_model = st.sidebar.selectbox(
    "Select Language Model:",
    options=groq_model_options,
    index=0
)

###############################################################################
# 5) CODE EDITOR & SYNTAX-PREVIEW
###############################################################################
st.markdown('<h2 class="sub-header">Model Definition</h2>', unsafe_allow_html=True)

with st.expander("Model Code Editor & Preview", expanded=True):
    col_code, col_preview = st.columns(2)

    with col_code:
        st.markdown('<p style="font-weight: bold; margin-bottom: 10px;">Model Code Editor</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 0.8rem; margin-bottom: 10px; color: #666;">
        Define your model using Python 3.12. You have access to numpy, scipy, and openturns libraries.
        Your code must define 'model' (an OpenTURNS Function) and 'problem' (an OpenTURNS Distribution).
        </div>
        """, unsafe_allow_html=True)
        code_area_value = st.text_area(
            label="",
            value=current_code,
            height=300
        )
        # Update current code if changed in the editor
        current_code = code_area_value
        
        # Add a model validation button
        if st.button("Validate Model Input"):
            if not current_code:
                st.error("Please provide a model first or select one of the example models to start with.")
            else:
                try:
                    # First check if the code is safe to execute
                    is_safe, safety_message = check_code_safety(current_code)
                    
                    if not is_safe:
                        st.error(f"Security Error: {safety_message}")
                    else:
                        # Execute the code
                        eval_globals = {}
                        exec(current_code, eval_globals)
                        
                        # Get model and problem
                        model = eval_globals.get('model')
                        problem = eval_globals.get('problem')
                        
                        if not model or not problem:
                            st.error("Model code must define 'model' and 'problem' variables.")
                        else:
                            # Run a small Monte Carlo simulation
                            with st.spinner("Running 10 Monte Carlo samples..."):
                                try:
                                    results = monte_carlo_simulation(model, problem, N=10, seed=42)
                                    # Check if mean is a numpy array or scalar
                                    mean_value = results['mean']
                                    std_value = results['std']
                                    if isinstance(mean_value, np.ndarray):
                                        mean_str = f"{mean_value[0]:.4f}"
                                        std_str = f"{std_value[0]:.4f}"
                                    else:
                                        mean_str = f"{mean_value:.4f}"
                                        std_str = f"{std_value:.4f}"
                                    st.success(f"Model validated successfully! Sample mean: {mean_str}, std: {std_str}")
                                except Exception as e:
                                    st.error(f"Error running model: {e}")
                except Exception as e:
                    st.error(f"Error evaluating model code: {e}")

    with col_preview:
        st.markdown('<p style="font-weight: bold; margin-bottom: 10px;">Syntax-Highlighted Preview</p>', unsafe_allow_html=True)
        if current_code.strip():
            st.code(current_code, language="python")
        else:
            st.markdown('<div class="info-box status-box">No code to display. Please select or upload a model.</div>', unsafe_allow_html=True)

###############################################################################
# 7) PAGE-SPECIFIC CONTENT
###############################################################################
if "📊 Main Analysis" in selected_page:
    st.subheader("Uncertainty Analysis Dashboard")
    
    # Check if we have a model
    if not current_code:
        st.markdown('<div class="error-box status-box">Please provide a model first.</div>', unsafe_allow_html=True)
    else:
        # First check if the code is safe to execute
        is_safe, safety_message = check_code_safety(current_code)
        
        if not is_safe:
            st.markdown(f'<div class="error-box status-box">Security Error: {safety_message}</div>', unsafe_allow_html=True)
        else:
            # Try to execute the code
            try:
                # Execute the code
                eval_globals = {}
                exec(current_code, eval_globals)
                
                # Get model and problem
                model = eval_globals.get('model')
                problem = eval_globals.get('problem')
                
                if not model or not problem:
                    st.markdown('<div class="error-box status-box">Model code must define \'model\' and \'problem\' variables.</div>', unsafe_allow_html=True)
                else:
                    # Run Monte Carlo simulation
                    with st.spinner("Running Monte Carlo Simulation..."):
                        N = 1000
                        data = monte_carlo_simulation(model, problem, N)
                        st.session_state.simulation_data = data
                        st.session_state.analyses_ran = True
                    
                    # Create analysis tabs
                    tabs = st.tabs([
                        "Model Understanding", 
                        "Convergence and Output Analysis",
                        "Exploratory Data Analysis", 
                        "Sobol Sensitivity", 
                        "FAST Sensitivity", 
                        "ANCOVA Sensitivity", 
                        "Taylor Analysis", 
                        "Correlation Analysis", 
                        "HSIC Analysis", 
                        "ML Analysis"
                    ])
                    
                    # Model Understanding Tab
                    with tabs[0]:
                        # Add a button to run this analysis
                        if st.button("Run Model Understanding", key="run_model_understanding"):
                            # Create a placeholder for immediate feedback
                            model_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(model_status, "Model Understanding")
                            
                            with st.spinner("Running Model Understanding..."):
                                try:
                                    model_understanding(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.model_understanding_ran = True
                                    model_status.empty()
                                except Exception as e:
                                    model_status.empty()
                                    st.error(f"Error in Model Understanding: {str(e)}")
                        elif st.session_state.model_understanding_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading Model Understanding Results..."):
                                try:
                                    model_understanding(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading Model Understanding results: {str(e)}")

                    # Convergence and Output Analysis Tab (moved before EDA)
                    with tabs[1]:
                        if st.button("Run Convergence and Output Analysis", key="run_expectation"):
                            # Create a placeholder for immediate feedback
                            conv_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(conv_status, "Convergence and Output Analysis")
                            
                            with st.spinner("Running Convergence and Output Analysis..."):
                                try:
                                    expectation_convergence_analysis_joint(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.expectation_convergence_ran = True
                                    conv_status.empty()
                                except Exception as e:
                                    conv_status.empty()
                                    st.error(f"Error in Convergence Analysis: {str(e)}")
                        elif st.session_state.expectation_convergence_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading Convergence and Output Analysis Results..."):
                                try:
                                    expectation_convergence_analysis_joint(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading Convergence Analysis results: {str(e)}")
                    
                    # Exploratory Data Analysis Tab (moved after Convergence)
                    with tabs[2]:
                        if st.button("Run Exploratory Data Analysis", key="run_eda"):
                            # Create a placeholder for immediate feedback
                            eda_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(eda_status, "Exploratory Data Analysis")
                            
                            with st.spinner("Running Exploratory Data Analysis..."):
                                try:
                                    # Convert Monte Carlo results to DataFrame if needed
                                    if isinstance(data, dict):
                                        df = create_monte_carlo_dataframe(data)
                                    else:
                                        df = data
                                        
                                    exploratory_data_analysis(
                                        df, N, model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.eda_ran = True
                                    eda_status.empty()
                                except Exception as e:
                                    eda_status.empty()
                                    st.error(f"Error in Exploratory Data Analysis: {str(e)}")
                        elif st.session_state.eda_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading Exploratory Data Analysis Results..."):
                                try:
                                    # Convert Monte Carlo results to DataFrame if needed
                                    if isinstance(data, dict):
                                        df = create_monte_carlo_dataframe(data)
                                    else:
                                        df = data
                                        
                                    exploratory_data_analysis(
                                        df, N, model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading Exploratory Data Analysis results: {str(e)}")
                    
                    # Sobol Sensitivity Analysis Tab
                    with tabs[3]:
                        # Add a slider for sample size
                        sobol_n = st.slider("Number of Sobol Samples", min_value=256, max_value=4096, value=1024, step=256)
                        # Add a button to run this analysis
                        if st.button("Run Sobol Sensitivity Analysis", key="run_sobol"):
                            # Create a placeholder for immediate feedback
                            sobol_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(sobol_status, "Sobol Sensitivity Analysis")
                            
                            with st.spinner("Running Sobol Sensitivity Analysis..."):
                                try:
                                    sobol_sensitivity_analysis(
                                        sobol_n, model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.sobol_ran = True
                                    sobol_status.empty()
                                except Exception as e:
                                    sobol_status.empty()
                                    st.error(f"Error in Sobol Analysis: {str(e)}")
                        elif st.session_state.sobol_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading Sobol Sensitivity Analysis Results..."):
                                try:
                                    sobol_sensitivity_analysis(
                                        1024, model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading Sobol Analysis results: {str(e)}")
                    
                    # FAST Sensitivity Analysis Tab
                    with tabs[4]:
                        # Add a button to run this analysis
                        if st.button("Run FAST Sensitivity Analysis", key="run_fast"):
                            # Create a placeholder for immediate feedback
                            fast_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(fast_status, "FAST Sensitivity Analysis")
                            
                            with st.spinner("Running FAST Sensitivity Analysis..."):
                                try:
                                    fast_analysis(
                                        model, problem, model_code_str=current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.fast_ran = True
                                    fast_status.empty()
                                except Exception as e:
                                    fast_status.empty()
                                    st.error(f"Error in FAST Analysis: {str(e)}")
                        elif st.session_state.fast_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading FAST Sensitivity Analysis Results..."):
                                try:
                                    fast_analysis(
                                        model, problem, model_code_str=current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading FAST Analysis results: {str(e)}")
                    
                    # ANCOVA Sensitivity Analysis Tab
                    with tabs[5]:
                        # Add a button to run this analysis
                        if st.button("Run ANCOVA Sensitivity Analysis", key="run_ancova"):
                            # Create a placeholder for immediate feedback
                            ancova_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(ancova_status, "ANCOVA Sensitivity Analysis")
                            
                            with st.spinner("Running ANCOVA Sensitivity Analysis..."):
                                try:
                                    ancova_analysis(
                                        model, problem, model_code_str=current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.ancova_ran = True
                                    ancova_status.empty()
                                except Exception as e:
                                    ancova_status.empty()
                                    st.error(f"Error in ANCOVA Analysis: {str(e)}")
                        elif st.session_state.ancova_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading ANCOVA Sensitivity Analysis Results..."):
                                try:
                                    ancova_analysis(
                                        model, problem, model_code_str=current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading ANCOVA Analysis results: {str(e)}")
                    
                    # Taylor Analysis Tab
                    with tabs[6]:
                        # Add a button to run this analysis
                        if st.button("Run Taylor Analysis", key="run_taylor"):
                            # Create a placeholder for immediate feedback
                            taylor_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(taylor_status, "Taylor Analysis")
                            
                            with st.spinner("Running Taylor Analysis..."):
                                try:
                                    taylor_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.taylor_ran = True
                                    taylor_status.empty()
                                except Exception as e:
                                    taylor_status.empty()
                                    st.error(f"Error in Taylor Analysis: {str(e)}")
                        elif st.session_state.taylor_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading Taylor Analysis Results..."):
                                try:
                                    taylor_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading Taylor Analysis results: {str(e)}")
                    
                    # Correlation Analysis Tab
                    with tabs[7]:
                        # Add a button to run this analysis
                        if st.button("Run Correlation Analysis", key="run_correlation"):
                            # Create a placeholder for immediate feedback
                            corr_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(corr_status, "Correlation Analysis")
                            
                            with st.spinner("Running Correlation Analysis..."):
                                try:
                                    correlation_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.correlation_ran = True
                                    corr_status.empty()
                                except Exception as e:
                                    corr_status.empty()
                                    st.error(f"Error in Correlation Analysis: {str(e)}")
                        elif st.session_state.correlation_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading Correlation Analysis Results..."):
                                try:
                                    correlation_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading Correlation Analysis results: {str(e)}")
                    
                    # HSIC Analysis Tab
                    with tabs[8]:
                        # Add a button to run this analysis
                        if st.button("Run HSIC Analysis", key="run_hsic"):
                            # Create a placeholder for immediate feedback
                            hsic_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(hsic_status, "HSIC Analysis")
                            
                            with st.spinner("Running HSIC Analysis..."):
                                try:
                                    hsic_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.hsic_ran = True
                                    hsic_status.empty()
                                except Exception as e:
                                    hsic_status.empty()
                                    st.error(f"Error in HSIC Analysis: {str(e)}")
                        elif st.session_state.hsic_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading HSIC Analysis Results..."):
                                try:
                                    hsic_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading HSIC Analysis results: {str(e)}")
                    
                    # ML Analysis Tab
                    with tabs[9]:
                        # Add a button to run this analysis
                        if st.button("Run ML Analysis", key="run_ml"):
                            # Create a placeholder for immediate feedback
                            ml_status = st.empty()
                            # Show immediate visual indicator
                            show_running_indicator(ml_status, "ML Analysis")
                            
                            with st.spinner("Running ML Analysis..."):
                                try:
                                    ml_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.shap_ran = True
                                    ml_status.empty()
                                except Exception as e:
                                    ml_status.empty()
                                    st.error(f"Error in ML Analysis: {str(e)}")
                                    st.error(traceback.format_exc())
                        elif st.session_state.shap_ran:
                            # Re-display the previously run analysis
                            with st.spinner("Loading ML Analysis Results..."):
                                try:
                                    ml_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model
                                    )
                                except Exception as e:
                                    st.error(f"Error loading ML Analysis results: {str(e)}")
                                    st.error(traceback.format_exc())
            except Exception as e:
                st.error(f"Error evaluating model code: {e}")

elif "📉 Dimensionality Reduction" in selected_page:
    # Use the modular dimensionality reduction page function from morris_analysis module
    dimensionality_reduction_page(current_code, model, problem, selected_language_model)

elif "📈 Distribution Fitting" in selected_page:
    # Use the distribution fitting page function
    distribution_fitting_page()
