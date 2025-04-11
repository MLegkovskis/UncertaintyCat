import os
import streamlit as st
import numpy as np
import openturns as ot
import pandas as pd
import plotly.express as px
import re
import traceback
import os

# Import modules
from modules.monte_carlo import monte_carlo_simulation, create_monte_carlo_dataframe
from modules.model_understanding import model_understanding, display_model_understanding
from modules.exploratory_data_analysis import exploratory_data_analysis, display_exploratory_data_results
from modules.expectation_convergence_analysis import expectation_convergence_analysis_joint, display_expectation_convergence_results
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis, display_sobol_results
from modules.taylor_analysis import taylor_analysis, display_taylor_results
from modules.correlation_analysis import correlation_analysis, display_correlation_results
from modules.hsic_analysis import hsic_analysis, display_hsic_results
from modules.ml_analysis import ml_analysis, display_ml_results
from modules.morris_analysis import morris_analysis, dimensionality_reduction_page
from modules.fast_analysis import fast_analysis, display_fast_results
from modules.ancova_analysis import ancova_analysis, display_ancova_results
from modules.distribution_fitting import distribution_fitting_page, run_distribution_fitting_analysis

# Import utils
from utils.core_utils import (
    get_model_options,
    check_code_safety,
    call_groq_api,
    create_chat_interface
)
from utils.css_styles import load_css
from utils.model_utils import (
    validate_problem_structure,
    test_model,
    sample_inputs
)
from utils.constants import RETURN_INSTRUCTION

def run_all_analyses(model, problem, current_code, selected_language_model):
    """
    Run all analyses in batch mode and store results in a unified dictionary.
    
    This function runs each analysis with display_results=False to avoid cluttering the UI,
    and stores the results in a single dictionary in session state for display in the unified report.
    
    Parameters
    ----------
    model : callable
        The model function
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    current_code : str
        String representation of the model code
    selected_language_model : str
        Language model to use for AI insights
    """
    # Initialize the all_results dictionary if it doesn't exist
    if 'all_results' not in st.session_state:
        st.session_state.all_results = {}
    
    # Assume you want to run 10 analyses
    total_analyses = 10
    completed = 0

    # Initialize a global progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(name):
        nonlocal completed
        status_text.markdown(f'<div class="running-indicator">Running {name}...</div>', unsafe_allow_html=True)
        completed += 1
        progress_bar.progress(completed / total_analyses)
        
    # Helper function to add a pause between analyses
    def pause_between_analyses():
        import time
        pause_text = st.empty()
        pause_text.info("Pausing for 5 seconds to respect API rate limits...")
        time.sleep(5)  # 5 second pause
        pause_text.empty()

    # Run each analysis in "silent" mode
    try:
        # Model Understanding
        update_progress("Model Understanding")
        st.session_state.all_results["Model Understanding"] = model_understanding(
            model, problem, current_code,
            language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # Convergence Analysis
        update_progress("Convergence and Output Analysis")
        st.session_state.all_results["Convergence Analysis"] = expectation_convergence_analysis_joint(
            model, problem, current_code,
            N_samples=8000,
            language_model=selected_language_model, 
            display_results=False
        )
        pause_between_analyses()

        # Exploratory Data Analysis
        update_progress("Exploratory Data Analysis")
        # Convert Monte Carlo results to DataFrame if needed
        if 'simulation_data' in st.session_state:
            df = create_monte_carlo_dataframe(st.session_state.simulation_data)
            st.session_state.all_results["Exploratory Data Analysis"] = exploratory_data_analysis(
                df, 1000, model, problem, current_code,
                language_model=selected_language_model, display_results=False
            )
        pause_between_analyses()

        # Sobol Sensitivity Analysis
        update_progress("Sobol Sensitivity Analysis")
        st.session_state.all_results["Sobol Analysis"] = sobol_sensitivity_analysis(
            1024, model, problem, current_code,
            language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # FAST Sensitivity Analysis
        update_progress("FAST Sensitivity Analysis")
        st.session_state.all_results["FAST Analysis"] = fast_analysis(
            model, problem, model_code_str=current_code,
            language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # ANCOVA Sensitivity Analysis
        update_progress("ANCOVA Sensitivity Analysis")
        st.session_state.all_results["ANCOVA Analysis"] = ancova_analysis(
            model, problem, model_code_str=current_code,
            language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # Taylor Analysis
        update_progress("Taylor Analysis")
        st.session_state.all_results["Taylor Analysis"] = taylor_analysis(
            model, problem, current_code,
            language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # Correlation Analysis
        update_progress("Correlation Analysis")
        st.session_state.all_results["Correlation Analysis"] = correlation_analysis(
            model, problem, current_code,
            language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # HSIC Analysis
        update_progress("HSIC Analysis")
        st.session_state.all_results["HSIC Analysis"] = hsic_analysis(
            model, problem, current_code,
            language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # ML Analysis
        update_progress("ML Analysis")
        st.session_state.all_results["ML Analysis"] = ml_analysis(
            model, problem, current_code,
            language_model=selected_language_model, display_results=False
        )
        
        # Set overall flag that all analyses have been run
        st.session_state.analyses_ran = True
        
    except Exception as e:
        st.error(f"Error during run-all analyses: {e}")
        traceback.print_exc()  # Print the full traceback for debugging
    
    # Final update
    progress_bar.progress(1.0)
    status_text.markdown('<div style="background-color: #d4edda; border-left: 5px solid #28a745; padding: 10px; border-radius: 4px;">All analyses completed! You can now view the comprehensive report below.</div>', unsafe_allow_html=True)

# Initialize session state variables
if 'analyses_ran' not in st.session_state:
    st.session_state.analyses_ran = False
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'all_results' not in st.session_state:
    st.session_state.all_results = {}
if 'global_chat_messages' not in st.session_state:
    st.session_state.global_chat_messages = []

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
    st.image("logo.jpg", width=300)
with col2:
    st.header("UncertaintyCat | Version 5.20")
    st.caption("Advanced Uncertainty Quantification and Sensitivity Analysis Platform")

# Fullscreen recommendation
st.info("📌 **Tip:** This application works best in fullscreen mode to view all analysis tabs properly.")

# Sidebar styling and navigation
st.sidebar.header("Navigation Panel")

# Create pages with icons
pages = {
    "📊 Main Analysis": "Comprehensive uncertainty quantification and sensitivity analysis",
    "📉 Dimensionality Reduction": "Reduce model complexity by identifying non-influential variables",
    "📈 Distribution Fitting": "Fit probability distributions to your data for UQ analysis"
}
selected_page = st.sidebar.radio("", list(pages.keys()))
st.sidebar.caption(pages[selected_page.strip()])

# Sidebar divider
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

###############################################################################
# 4) MODEL SELECT / UPLOAD
###############################################################################
st.sidebar.header("Model Configuration")

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
st.sidebar.header("AI Configuration")

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
st.header("Model Definition")

with st.expander("Model Code Editor & Preview", expanded=True):
    col_code, col_preview = st.columns(2)

    with col_code:
        st.subheader("Model Code Editor")
        st.caption("Define your model using Python 3.12. You have access to numpy, scipy, and openturns libraries. Your code must define 'model' (an OpenTURNS Function) and 'problem' (an OpenTURNS Distribution).")
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
        st.subheader("Syntax-Highlighted Preview")
        if current_code.strip():
            st.code(current_code, language="python")
        else:
            st.info("No code to display. Please select or upload a model.")

###############################################################################
# 7) PAGE-SPECIFIC CONTENT
###############################################################################
if "📊 Main Analysis" in selected_page:
    st.subheader("Uncertainty Analysis Dashboard")
    
    # Check if we have a model
    if not current_code:
        st.info("Please select or upload a model file to begin analysis.")
    else:
        # First check if the code is safe to execute
        is_safe, safety_message = check_code_safety(current_code)
        
        if not is_safe:
            st.error(f"Security Error: {safety_message}")
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
                    st.error("Model code must define 'model' and 'problem' variables.")
                else:
                    # Run Monte Carlo simulation
                    with st.spinner("Running Monte Carlo Simulation..."):
                        N = 1000
                        data = monte_carlo_simulation(model, problem, N)
                        st.session_state.simulation_data = data
                    
                    # Add a "Run All Analyses" button
                    st.subheader("Run All Analyses")
                    st.write("Click the button below to run all analyses sequentially. This may take some time depending on the complexity of your model.")
                    
                    if st.button("🚀 Run All Analyses", key="run_all_analyses", help="This will run all analyses sequentially and generate a comprehensive report."):
                        run_all_analyses(model, problem, current_code, selected_language_model)
                    
                    # Display the comprehensive report if analyses have been run
                    if st.session_state.analyses_ran and 'all_results' in st.session_state:
                        st.header("Comprehensive Analysis Report")
                        st.caption("This report contains the results of all analyses run on your model. Click on each section to expand and view detailed results.")
                        
                        # Create a two-column layout: main content (2/3) and chat interface (1/3)
                        report_col, chat_col = st.columns([2, 1])
                        
                        with report_col:
                            # Iterate through all analysis results and create expandable sections
                            for module_name, results in st.session_state.all_results.items():
                                if results:  # Only display if results exist
                                    # Create an expandable container for each module
                                    with st.expander(f"📊 {module_name}", expanded=False):
                                        st.subheader(f"{module_name} Results")
                                        
                                        # Display module-specific results
                                        if module_name == "Model Understanding":
                                            # Create a container for the results
                                            with st.container():
                                                # Display the results directly
                                                if isinstance(results, dict):
                                                    if 'explanation' in results:
                                                        st.write("##### Model Explanation")
                                                        st.markdown(results['explanation'])
                                                    
                                                    # Display input distributions
                                                    if 'inputs_df' in results:
                                                        st.write("##### Input Distributions")
                                                        try:
                                                            # Display the dataframe with styling
                                                            st.dataframe(
                                                                results['inputs_df'],
                                                                use_container_width=True,
                                                                hide_index=False
                                                            )
                                                        except Exception as e:
                                                            st.error(f"Error displaying input distributions: {e}")
                                                    
                                                    # Display model code
                                                    if 'model_code_str' in results:
                                                        st.write("##### Model Code")
                                                        st.code(results['model_code_str'], language="python")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "Convergence Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # Mean convergence
                                                    if 'mean_convergence_fig' in results:
                                                        st.write("##### Mean Convergence")
                                                        st.plotly_chart(results['mean_convergence_fig'], use_container_width=True)
                                                    
                                                    # Standard deviation convergence
                                                    if 'std_convergence_fig' in results:
                                                        st.write("##### Standard Deviation Convergence")
                                                        st.plotly_chart(results['std_convergence_fig'], use_container_width=True)
                                                    
                                                    # Output distribution
                                                    if 'output_distribution_fig' in results:
                                                        st.write("##### Output Distribution")
                                                        st.plotly_chart(results['output_distribution_fig'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['mean_convergence_fig', 'std_convergence_fig', 'output_distribution_fig']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "Exploratory Data Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # Pairplot
                                                    if 'pairplot' in results:
                                                        st.write("##### Pairwise Relationships")
                                                        st.plotly_chart(results['pairplot'], use_container_width=True)
                                                    
                                                    # Correlation heatmap
                                                    if 'correlation_heatmap' in results:
                                                        st.write("##### Correlation Heatmap")
                                                        st.plotly_chart(results['correlation_heatmap'], use_container_width=True)
                                                    
                                                    # Summary statistics
                                                    if 'summary_stats' in results:
                                                        st.write("##### Summary Statistics")
                                                        st.dataframe(results['summary_stats'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['pairplot', 'correlation_heatmap', 'summary_stats']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "Sobol Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # Bar chart
                                                    if 'fig_bar' in results:
                                                        st.write("##### Sobol Sensitivity Indices")
                                                        st.plotly_chart(results['fig_bar'], use_container_width=True)
                                                    
                                                    # Indices dataframe
                                                    if 'indices_df' in results:
                                                        st.write("##### Sensitivity Indices Table")
                                                        st.dataframe(results['indices_df'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['fig_bar', 'indices_df']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "FAST Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # Bar chart
                                                    if 'fig_bar' in results:
                                                        st.write("##### FAST Sensitivity Indices")
                                                        st.plotly_chart(results['fig_bar'], use_container_width=True)
                                                    
                                                    # Indices dataframe
                                                    if 'indices_df' in results:
                                                        st.write("##### Sensitivity Indices Table")
                                                        st.dataframe(results['indices_df'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['fig_bar', 'indices_df']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "ANCOVA Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # Indices figure
                                                    if 'indices_fig' in results:
                                                        st.write("##### ANCOVA Sensitivity Indices")
                                                        st.plotly_chart(results['indices_fig'], use_container_width=True)
                                                    
                                                    # Indices dataframe
                                                    if 'indices_df' in results:
                                                        st.write("##### Sensitivity Indices Table")
                                                        st.dataframe(results['indices_df'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['indices_fig', 'indices_df']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "Taylor Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # Bar chart
                                                    if 'bar_chart' in results:
                                                        st.write("##### Taylor Sensitivity Indices")
                                                        st.plotly_chart(results['bar_chart'], use_container_width=True)
                                                    
                                                    # Taylor dataframe
                                                    if 'taylor_df' in results:
                                                        st.write("##### Taylor Coefficients Table")
                                                        st.dataframe(results['taylor_df'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['bar_chart', 'taylor_df']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "Correlation Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # Correlation heatmap
                                                    if 'correlation_heatmap' in results:
                                                        st.write("##### Correlation Heatmap")
                                                        st.plotly_chart(results['correlation_heatmap'], use_container_width=True)
                                                    
                                                    # Correlation dataframe
                                                    if 'correlation_df' in results:
                                                        st.write("##### Correlation Coefficients Table")
                                                        st.dataframe(results['correlation_df'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['correlation_heatmap', 'correlation_df']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "HSIC Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # HSIC figure
                                                    if 'fig' in results:
                                                        st.write("##### HSIC Sensitivity Indices")
                                                        st.plotly_chart(results['fig'], use_container_width=True)
                                                    
                                                    # HSIC dataframe
                                                    if 'hsic_df' in results:
                                                        st.write("##### HSIC Indices Table")
                                                        st.dataframe(results['hsic_df'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['fig', 'hsic_df']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        elif module_name == "ML Analysis":
                                            with st.container():
                                                if isinstance(results, dict):
                                                    # SHAP summary plot
                                                    if 'shap_summary_plot' in results:
                                                        st.write("##### SHAP Summary Plot")
                                                        st.plotly_chart(results['shap_summary_plot'], use_container_width=True)
                                                    
                                                    # SHAP bar plot
                                                    if 'shap_bar_plot' in results:
                                                        st.write("##### SHAP Feature Importance")
                                                        st.plotly_chart(results['shap_bar_plot'], use_container_width=True)
                                                    
                                                    # Validation plot
                                                    if 'validation_plot' in results:
                                                        st.write("##### Model Validation")
                                                        st.plotly_chart(results['validation_plot'], use_container_width=True)
                                                    
                                                    # Feature importance
                                                    if 'feature_importance' in results:
                                                        st.write("##### Feature Importance Table")
                                                        st.dataframe(results['feature_importance'], use_container_width=True)
                                                    
                                                    # Model metrics
                                                    if 'model_metrics' in results:
                                                        st.write("##### Model Performance Metrics")
                                                        st.dataframe(results['model_metrics'], use_container_width=True)
                                                    
                                                    # Display any other available results
                                                    for key, value in results.items():
                                                        if key not in ['shap_summary_plot', 'shap_bar_plot', 'validation_plot', 'feature_importance', 'model_metrics']:
                                                            if hasattr(value, 'to_html'):
                                                                st.write(f"##### {key.replace('_', ' ').title()}")
                                                                st.dataframe(value, use_container_width=True)
                                                            elif isinstance(value, (str, int, float)):
                                                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                                else:
                                                    st.info("No detailed results available for this analysis.")
                                        
                                        # Add a separator between modules
                                        st.write("---")
                        
                        # Create a unified chat interface in the right column
                        with chat_col:
                            st.header("Ask Questions About the Entire Report")
                            st.caption("Use this chat interface to ask questions about any part of the analysis report. The AI assistant will provide insights based on all the analyses performed.")
                            
                            # Create a context generator function for the global chat
                            def global_context_generator(prompt):
                                # Create a summary of available analyses
                                analysis_summary = ", ".join(list(st.session_state.all_results.keys()))
                                
                                # Build a comprehensive context
                                context = f"""
                                You are an expert assistant helping users understand a comprehensive uncertainty quantification and sensitivity analysis report.
                                
                                The report includes results from the following analyses: {analysis_summary}
                                
                                The model being analyzed is defined as:
                                ```python
                                {current_code}
                                ```
                                
                                Please answer the user's question in the context of the full report. If necessary, refer to specific analyses by name.
                                
                                User question: {prompt}
                                """
                                return context
                            
                            # Add a container with styling for the chat interface
                            chat_container = st.container()
                            with chat_container:
                                st.write("""
                                <style>
                                .chat-container {
                                    border: 1px solid #e0e0e0;
                                    border-radius: 10px;
                                    padding: 10px;
                                    background-color: #f9f9f9;
                                    height: 400px;
                                    overflow-y: auto;
                                    margin-bottom: 15px;
                                }
                                </style>
                                <div class="chat-container" id="chat-messages">
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Create a scrollable chat interface
                            create_chat_interface(
                                "global_chat",
                                global_context_generator,
                                input_placeholder="Ask a question about any part of the analysis...",
                                disclaimer_text="Ask questions about any part of the analysis report.",
                                language_model=selected_language_model
                            )
                    else:
                        st.info("Please run all analyses first by clicking the 'Run All Analyses' button.")
                        
            except Exception as e:
                st.error(f"Error evaluating model code: {e}")

elif "📉 Dimensionality Reduction" in selected_page:
    # Use the modular dimensionality reduction page function from morris_analysis module
    dimensionality_reduction_page(current_code, model, problem, selected_language_model)

elif "📈 Distribution Fitting" in selected_page:
    # Use the distribution fitting page function
    distribution_fitting_page()
