import os
import streamlit as st
import numpy as np
import openturns as ot
import pandas as pd
import plotly.express as px

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
from modules.fast_ancova_analysis import fast_analysis, ancova_analysis

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
# Set page configuration with custom theme
st.set_page_config(
    layout="wide",
    page_title="UncertaintyCat | Enterprise",
    page_icon="🐱",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise look and feel
st.markdown("""
<style>
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #2C3E50;
        padding-bottom: 15px;
        border-bottom: 2px solid #3498DB;
    }
    .sub-header {
        font-family: 'Arial', sans-serif;
        color: #34495E;
        padding: 10px 0;
        margin-top: 20px;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #EBF5FB;
        border-left: 5px solid #3498DB;
    }
    .success-box {
        background-color: #E9F7EF;
        border-left: 5px solid #2ECC71;
    }
    .warning-box {
        background-color: #FEF9E7;
        border-left: 5px solid #F1C40F;
    }
    .error-box {
        background-color: #FDEDEC;
        border-left: 5px solid #E74C3C;
    }
    .stTextArea>div>div>textarea {
        font-family: 'Courier New', monospace;
        background-color: #F8F9FA;
    }
    .section-divider {
        height: 3px;
        background-color: #F0F3F4;
        margin: 30px 0;
        border-radius: 2px;
    }
    /* Custom CSS for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: 1px solid #dee2e6;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #3498DB;
        font-weight: bold;
    }
    .tab-content {
        padding: 16px;
        border: 1px solid #dee2e6;
        border-top: none;
        border-radius: 0 0 4px 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.jpg", width=100)
with col2:
    st.markdown('<h1 class="main-header">UncertaintyCat | Enterprise Edition</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #7F8C8D;">Advanced Uncertainty Quantification and Sensitivity Analysis Platform</p>', unsafe_allow_html=True)

# Sidebar styling and navigation
st.sidebar.markdown('<div style="text-align: center; padding: 10px 0;"><h3>Navigation Panel</h3></div>', unsafe_allow_html=True)

# Create pages with icons
pages = {
    "📊 Main Analysis": "Comprehensive uncertainty quantification and sensitivity analysis",
    "📉 Dimensionality Reduction": "Reduce model complexity by identifying non-influential variables"
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
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
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
        code_area_value = st.text_area(
            label="",
            value=current_code,
            height=300
        )
        # Update current code if changed in the editor
        current_code = code_area_value

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
    st.markdown('<h2 class="sub-header">Uncertainty Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Create a card for the run button
    st.markdown('<p style="font-weight: bold;">Run Comprehensive Analysis</p>', unsafe_allow_html=True)
    
    # Default number of samples
    N = st.slider("Number of Monte Carlo Samples", min_value=100, max_value=10000, value=2000, step=100)
    
    # Add run button
    run_button = st.button("Run Analysis", key="run_main_analysis")
    
    # Initialize session state for analyses
    if 'analyses_ran' not in st.session_state:
        st.session_state.analyses_ran = False
        st.session_state.simulation_data = None
        st.session_state.active_tab = None
        st.session_state.model_understanding_ran = False
        st.session_state.eda_ran = False
        st.session_state.expectation_convergence_ran = False
        st.session_state.sobol_ran = False
        st.session_state.fast_ran = False
        st.session_state.ancova_ran = False
        st.session_state.taylor_ran = False
        st.session_state.correlation_ran = False
        st.session_state.hsic_ran = False
        st.session_state.shap_ran = False
    
    if run_button:
        if not current_code:
            st.markdown('<div class="error-box status-box">Please provide a model first.</div>', unsafe_allow_html=True)
        else:
            try:
                # Execute the model code in a fresh namespace
                local_namespace = {}
                exec(current_code, local_namespace)
                
                if 'model' not in local_namespace or 'problem' not in local_namespace:
                    st.markdown('<div class="error-box status-box">The model code must define both \'model\' and \'problem\'.</div>', unsafe_allow_html=True)
                else:
                    model = local_namespace['model']
                    problem = local_namespace['problem']

                    # Validate the problem structure
                    validate_problem_structure(problem)
                    
                    # Run Monte Carlo simulation (always needed)
                    with st.spinner("Running Monte Carlo Simulation..."):
                        st.markdown('<div class="info-box status-box">Running Monte Carlo simulation with ' + str(N) + ' samples...</div>', unsafe_allow_html=True)
                        results = monte_carlo_simulation(model, problem, N=N, seed=42)
                        data = create_monte_carlo_dataframe(results)
                        st.markdown('<div class="success-box status-box">Monte Carlo simulation completed successfully.</div>', unsafe_allow_html=True)
                    
                    # Store results for use in analyses
                    st.session_state.simulation_data = {
                        "data": data,
                        "model": model,
                        "problem": problem,
                        "code": current_code,
                        "selected_language_model": selected_language_model,
                        "N": N
                    }
                    
                    # Mark analyses as ready to run
                    st.session_state.analyses_ran = True
                    # Set active tab to the first one
                    st.session_state.active_tab = "Model Understanding"
                    
                    # Force rerun to show tabs
                    st.rerun()
                    
            except Exception as e:
                st.markdown(f'<div class="error-box status-box">Error during simulation: {str(e)}</div>', unsafe_allow_html=True)
    
    # Display tabs only if analyses have been run
    if st.session_state.analyses_ran and st.session_state.simulation_data is not None:
        # Create analysis tabs
        tabs = st.tabs([
            "Model Understanding", 
            "Exploratory Data Analysis", 
            "Expectation Convergence", 
            "Sobol Sensitivity", 
            "FAST Sensitivity", 
            "ANCOVA Sensitivity", 
            "Taylor Analysis", 
            "Correlation Analysis", 
            "HSIC Analysis", 
            "SHAP Analysis"
        ])
        
        # Get the data
        data = st.session_state.simulation_data["data"]
        model = st.session_state.simulation_data["model"]
        problem = st.session_state.simulation_data["problem"]
        current_code = st.session_state.simulation_data["code"]
        selected_language_model = st.session_state.simulation_data["selected_language_model"]
        N = st.session_state.simulation_data["N"]
        
        # Model Understanding Tab
        with tabs[0]:
            if st.session_state.active_tab == "Model Understanding":
                with st.spinner("Running Model Understanding..."):
                    model_understanding(
                        model,
                        problem,
                        current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.model_understanding_ran = True
            elif st.session_state.model_understanding_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading Model Understanding Results..."):
                    model_understanding(
                        model,
                        problem,
                        current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a button to run this analysis
                if st.button("Run Model Understanding Analysis", key="run_model_understanding"):
                    with st.spinner("Running Model Understanding..."):
                        model_understanding(
                            model,
                            problem,
                            current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.model_understanding_ran = True
        
        # Exploratory Data Analysis Tab
        with tabs[1]:
            if st.session_state.active_tab == "Exploratory Data Analysis":
                with st.spinner("Running Exploratory Data Analysis..."):
                    exploratory_data_analysis(
                        data, N, model, problem, current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.eda_ran = True
            elif st.session_state.eda_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading Exploratory Data Analysis Results..."):
                    exploratory_data_analysis(
                        data, N, model, problem, current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a button to run this analysis
                if st.button("Run Exploratory Data Analysis", key="run_eda"):
                    with st.spinner("Running Exploratory Data Analysis..."):
                        exploratory_data_analysis(
                            data, N, model, problem, current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.eda_ran = True
        
        # Expectation Convergence Analysis Tab
        with tabs[2]:
            if st.session_state.active_tab == "Expectation Convergence":
                with st.spinner("Running Expectation Convergence Analysis..."):
                    expectation_convergence_analysis_joint(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.expectation_convergence_ran = True
            elif st.session_state.expectation_convergence_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading Expectation Convergence Analysis Results..."):
                    expectation_convergence_analysis_joint(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a button to run this analysis
                if st.button("Run Expectation Convergence Analysis", key="run_expectation"):
                    with st.spinner("Running Expectation Convergence Analysis..."):
                        expectation_convergence_analysis_joint(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.expectation_convergence_ran = True
        
        # Sobol Sensitivity Analysis Tab
        with tabs[3]:
            if st.session_state.active_tab == "Sobol Sensitivity":
                with st.spinner("Running Sobol Sensitivity Analysis..."):
                    sobol_sensitivity_analysis(
                        1024, model, problem, current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.sobol_ran = True
            elif st.session_state.sobol_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading Sobol Sensitivity Analysis Results..."):
                    sobol_sensitivity_analysis(
                        1024, model, problem, current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a slider for sample size
                sobol_n = st.slider("Number of Sobol Samples", min_value=256, max_value=4096, value=1024, step=256)
                # Add a button to run this analysis
                if st.button("Run Sobol Sensitivity Analysis", key="run_sobol"):
                    with st.spinner("Running Sobol Sensitivity Analysis..."):
                        sobol_sensitivity_analysis(
                            sobol_n, model, problem, current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.sobol_ran = True
        
        # FAST Sensitivity Analysis Tab
        with tabs[4]:
            if st.session_state.active_tab == "FAST Sensitivity":
                fast_analysis(
                    model, problem, size=400, model_code_str=current_code,
                    language_model=selected_language_model
                )
                st.session_state.active_tab = None
                st.session_state.fast_ran = True
            elif st.session_state.fast_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading FAST Sensitivity Analysis Results..."):
                    fast_analysis(
                        model, problem, size=400, model_code_str=current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a slider for sample size
                fast_size = st.slider("Number of FAST Samples", min_value=100, max_value=1000, value=400, step=100)
                # Add a button to run this analysis
                if st.button("Run FAST Sensitivity Analysis", key="run_fast"):
                    fast_analysis(
                        model, problem, size=fast_size, model_code_str=current_code,
                        language_model=selected_language_model
                    )
                    st.session_state.fast_ran = True
        
        # ANCOVA Sensitivity Analysis Tab
        with tabs[5]:
            if st.session_state.active_tab == "ANCOVA Sensitivity":
                ancova_analysis(
                    model, problem, size=2000, model_code_str=current_code,
                    language_model=selected_language_model
                )
                st.session_state.active_tab = None
                st.session_state.ancova_ran = True
            elif st.session_state.ancova_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading ANCOVA Sensitivity Analysis Results..."):
                    ancova_analysis(
                        model, problem, size=2000, model_code_str=current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a slider for sample size
                ancova_size = st.slider("Number of ANCOVA Samples", min_value=500, max_value=5000, value=2000, step=500)
                # Add a button to run this analysis
                if st.button("Run ANCOVA Sensitivity Analysis", key="run_ancova"):
                    ancova_analysis(
                        model, problem, size=ancova_size, model_code_str=current_code,
                        language_model=selected_language_model
                    )
                    st.session_state.ancova_ran = True
        
        # Taylor Analysis Tab
        with tabs[6]:
            if st.session_state.active_tab == "Taylor Analysis":
                with st.spinner("Running Taylor Analysis..."):
                    taylor_analysis(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.taylor_ran = True
            elif st.session_state.taylor_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading Taylor Analysis Results..."):
                    taylor_analysis(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a button to run this analysis
                if st.button("Run Taylor Analysis", key="run_taylor"):
                    with st.spinner("Running Taylor Analysis..."):
                        taylor_analysis(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.taylor_ran = True
        
        # Correlation Analysis Tab
        with tabs[7]:
            if st.session_state.active_tab == "Correlation Analysis":
                with st.spinner("Running Correlation Analysis..."):
                    correlation_analysis(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.correlation_ran = True
            elif st.session_state.correlation_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading Correlation Analysis Results..."):
                    correlation_analysis(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a button to run this analysis
                if st.button("Run Correlation Analysis", key="run_correlation"):
                    with st.spinner("Running Correlation Analysis..."):
                        correlation_analysis(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.correlation_ran = True
        
        # HSIC Analysis Tab
        with tabs[8]:
            if st.session_state.active_tab == "HSIC Analysis":
                with st.spinner("Running HSIC Analysis..."):
                    hsic_analysis(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.hsic_ran = True
            elif st.session_state.hsic_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading HSIC Analysis Results..."):
                    hsic_analysis(
                        model, problem, current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a button to run this analysis
                if st.button("Run HSIC Analysis", key="run_hsic"):
                    with st.spinner("Running HSIC Analysis..."):
                        hsic_analysis(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.hsic_ran = True
        
        # ML Analysis Tab
        with tabs[9]:
            if st.session_state.active_tab == "SHAP Analysis":
                with st.spinner("Running SHAP Analysis..."):
                    ml_analysis(
                        data, problem, current_code,
                        language_model=selected_language_model
                    )
                st.session_state.active_tab = None
                st.session_state.shap_ran = True
            elif st.session_state.shap_ran:
                # Re-display the previously run analysis
                with st.spinner("Loading SHAP Analysis Results..."):
                    ml_analysis(
                        data, problem, current_code,
                        language_model=selected_language_model
                    )
            else:
                # Add a button to run this analysis
                if st.button("Run SHAP Analysis", key="run_ml"):
                    with st.spinner("Running SHAP Analysis..."):
                        ml_analysis(
                            data, problem, current_code,
                            language_model=selected_language_model
                        )
                    st.session_state.shap_ran = True

elif "📉 Dimensionality Reduction" in selected_page:
    # Use the modular dimensionality reduction page function from morris_analysis module
    dimensionality_reduction_page(current_code, model, problem, selected_language_model)
