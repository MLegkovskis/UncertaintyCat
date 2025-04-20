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
from modules.exploratory_data_analysis import exploratory_data_analysis, display_exploratory_data_analysis_results
from modules.expectation_convergence_analysis import expectation_convergence_analysis_joint, display_expectation_convergence_results
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis, display_sobol_results
from modules.taylor_analysis import taylor_analysis, display_taylor_results
from modules.correlation_analysis import correlation_analysis, display_correlation_results
from modules.hsic_analysis import compute_hsic_analysis, display_hsic_results
from modules.ml_analysis import ml_analysis, display_ml_results
from modules.morris_analysis import morris_analysis, dimensionality_reduction_page
from modules.fast_analysis import fast_analysis, fast_sensitivity_analysis, display_fast_results
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
        time.sleep(1)  # 5 second pause
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

        # Expectation Convergence Analysis
        update_progress("Expectation Convergence Analysis")
        from modules.expectation_convergence_analysis import compute_expectation_convergence_analysis, generate_ai_insights
        
        try:
            # Compute the expectation convergence analysis
            expectation_results = compute_expectation_convergence_analysis(
                model, problem, current_code, N_samples=8000
            )
            
            # Generate AI insights
            with st.spinner("Generating AI insights for Expectation Convergence Analysis..."):
                insights = generate_ai_insights(expectation_results, language_model=selected_language_model)
                expectation_results['ai_insights'] = insights
            
            # Ensure the fit_df is properly handled
            if 'fit_df' in expectation_results and not expectation_results['fit_df'].empty:
                # Make sure the fit_df is a proper pandas DataFrame
                if not isinstance(expectation_results['fit_df'], pd.DataFrame):
                    expectation_results['fit_df'] = pd.DataFrame(expectation_results['fit_df'])
                
                # Remove any non-serializable objects from the fit_df
                if 'OT_Distribution' in expectation_results['fit_df'].columns:
                    expectation_results['fit_df'] = expectation_results['fit_df'].drop(columns=['OT_Distribution'])
                
            # Store the results
            st.session_state.all_results["Expectation Convergence Analysis"] = expectation_results
        except Exception as e:
            st.error(f"Error in Expectation Convergence Analysis: {str(e)}")
            st.session_state.all_results["Expectation Convergence Analysis"] = {"error": str(e)}
            
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
            size=1024, language_model=selected_language_model, display_results=False
        )
        pause_between_analyses()

        # HSIC Analysis
        update_progress("HSIC Analysis")
        st.session_state.all_results["HSIC Analysis"] = compute_hsic_analysis(
            hsic_size=200, model=model, problem=problem, model_code_str=current_code,
            language_model=selected_language_model
        )
        pause_between_analyses()

        # ML Analysis
        update_progress("ML Analysis")
        st.session_state.all_results["ML Analysis"] = ml_analysis(
            model, problem, size=1000, model_code_str=current_code,
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
    
    # Force a rerun to update the sidebar chat interface
    st.rerun()

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

# Header with app title (centered, no logo column)
st.header("UncertaintyCat | Version 5.20")
st.caption("Advanced Uncertainty Quantification and Sensitivity Analysis Platform")

# Fullscreen recommendation
st.info("📌 **Tip:** This application works best in fullscreen mode to view all analysis tabs properly.")

# Sidebar styling and navigation
st.sidebar.image("logo.jpg", width=250)
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

# Add the global chat interface to the sidebar
st.sidebar.header("Global Chat")

# Check if analyses have been run
if st.session_state.analyses_ran and 'all_results' in st.session_state:
    st.sidebar.info("Ask questions about any part of the analysis report here.")
    
    # Initialize session state for sidebar chat messages if not already done
    if "sidebar_global_chat_messages" not in st.session_state:
        st.session_state.sidebar_global_chat_messages = []
    
    # Create a context generator function for the global chat in sidebar
    def sidebar_global_context_generator(prompt):
        # If analyses have been run, include the analysis summary
        analysis_summary = ", ".join(list(st.session_state.all_results.keys()))
        
        # Start building a comprehensive context
        context = f"""
        You are an expert assistant helping users understand a comprehensive uncertainty quantification and sensitivity analysis report.
        
        The report includes results from the following analyses: {analysis_summary}
        
        The model being analyzed is defined as:
        ```python
        {current_code}
        ```
        """
        
        # Add basic information about each analysis if available
        if 'all_results' in st.session_state:
            context += "\n\nThe following analyses have been performed:\n\n"
            
            # ANCOVA Analysis
            if "ANCOVA Analysis" in st.session_state.all_results:
                context += "- ANCOVA Analysis: Analyzes variance decomposition for correlated inputs\n"
            
            # Sobol Analysis
            if "Sobol Analysis" in st.session_state.all_results:
                context += "- Sobol Analysis: Quantifies the contribution of each input variable to the output variance\n"
            
            # ML Analysis
            if "ML Analysis" in st.session_state.all_results:
                context += "- ML Analysis: Uses machine learning to analyze variable importance\n"
            
            # Taylor Analysis
            if "Taylor Analysis" in st.session_state.all_results:
                context += "- Taylor Analysis: Uses first-order Taylor expansion to estimate sensitivity\n"
            
            # Correlation Analysis
            if "Correlation Analysis" in st.session_state.all_results:
                context += "- Correlation Analysis: Examines linear and rank correlations between inputs and outputs\n"
            
            # FAST Analysis
            if "FAST Analysis" in st.session_state.all_results:
                context += "- FAST Analysis: Fourier Amplitude Sensitivity Test for global sensitivity analysis\n"
            
            # HSIC Analysis
            if "HSIC Analysis" in st.session_state.all_results:
                context += "- HSIC Analysis: Hilbert-Schmidt Independence Criterion for detecting non-linear dependencies\n"
            
            # Add Sobol results summary if available and relevant
            if "Sobol Analysis" in st.session_state.all_results:
                sobol_results = st.session_state.all_results["Sobol Analysis"]
                indices_df = sobol_results.get("indices_df")
                sum_first_order = sobol_results.get("sum_first_order")
                sum_total_order = sobol_results.get("sum_total_order")
                interaction_effect = sobol_results.get("interaction_effect")
                
                if indices_df is not None:
                    context += "\n\n### Sobol Sensitivity Analysis Results\n"
                    # Show main indices as a Markdown table
                    context += indices_df.to_markdown(index=False)
                    context += f"\n\n- **Sum of First-Order Indices:** {sum_first_order:.4f}"
                    context += f"\n- **Sum of Total-Order Indices:** {sum_total_order:.4f}"
                    context += f"\n- **Interaction Effect:** {interaction_effect:.4f}"
        
        # Add FAST results summary if available and relevant
        if "FAST Analysis" in st.session_state.all_results:
            fast_results = st.session_state.all_results["FAST Analysis"]
            fast_indices_df = fast_results.get("indices_df")
            if fast_indices_df is not None:
                context += "\n\n### FAST Sensitivity Analysis Results\n"
                context += fast_indices_df.to_markdown(index=False)
                # Optionally add more summary stats if needed

        # Add ANCOVA results summary if available and relevant
        if "ANCOVA Analysis" in st.session_state.all_results:
            ancova_results = st.session_state.all_results["ANCOVA Analysis"]
            ancova_indices_df = ancova_results.get("indices_df")
            if ancova_indices_df is not None:
                context += "\n\n### ANCOVA Sensitivity Analysis Results\n"
                context += ancova_indices_df.to_markdown(index=False)
                # Optionally add more summary stats if needed

        # Add Taylor results summary if available and relevant
        if "Taylor Analysis" in st.session_state.all_results:
            taylor_results = st.session_state.all_results["Taylor Analysis"]
            taylor_indices_df = None
            # Try to get a DataFrame from the results
            if "indices_df" in taylor_results:
                taylor_indices_df = taylor_results["indices_df"]
            elif "input_names" in taylor_results and "sensitivity_indices" in taylor_results:
                # Fallback: create a DataFrame if only names and indices are present
                import pandas as pd
                taylor_indices_df = pd.DataFrame({
                    'Variable': taylor_results['input_names'],
                    'Sensitivity Index': taylor_results['sensitivity_indices']
                })
                taylor_indices_df = taylor_indices_df.sort_values('Sensitivity Index', ascending=False)
            if taylor_indices_df is not None:
                context += "\n\n### Taylor Sensitivity Analysis Results\n"
                context += taylor_indices_df.to_markdown(index=False)
                # Optionally add more summary stats if needed

        # Add Correlation Analysis results summary if available and relevant
        if "Correlation Analysis" in st.session_state.all_results:
            corr_results = st.session_state.all_results["Correlation Analysis"]
            all_corr = corr_results.get("all_correlation_results")
            if all_corr is not None:
                for output_name, corr_df in all_corr.items():
                    context += f"\n\n### Correlation Analysis Results for {output_name}\n"
                    context += corr_df.to_markdown()
                # Optionally add more summary stats if needed

        # Add HSIC Analysis results summary if available and relevant
        if "HSIC Analysis" in st.session_state.all_results:
            hsic_results = st.session_state.all_results["HSIC Analysis"]
            hsic_df = hsic_results.get("hsic_df")
            if hsic_df is not None:
                context += "\n\n### HSIC Sensitivity Analysis Results\n"
                context += hsic_df.to_markdown(index=False)
                # Optionally add more summary stats if needed

        # Add ML Analysis results summary if available and relevant
        if "ML Analysis" in st.session_state.all_results:
            ml_results = st.session_state.all_results["ML Analysis"]
            # SHAP summary (feature importance)
            shap_summary_df = None
            if "shap_results" in ml_results and "shap_summary_df" in ml_results["shap_results"]:
                shap_summary_df = ml_results["shap_results"]["shap_summary_df"]
            elif "feature_importance" in ml_results:
                shap_summary_df = ml_results["feature_importance"]
            if shap_summary_df is not None:
                context += "\n\n### ML Analysis: SHAP Feature Importance\n"
                context += shap_summary_df.to_markdown(index=False)
            # Model metrics
            model_metrics_df = ml_results.get("model_metrics")
            if model_metrics_df is not None:
                context += "\n\n### ML Model Performance Metrics\n"
                context += model_metrics_df.to_markdown(index=False)
            # Optionally add more summary stats if needed

        # Add Expectation Convergence Analysis results summary if available and relevant
        if "Expectation Convergence Analysis" in st.session_state.all_results:
            exp_results = st.session_state.all_results["Expectation Convergence Analysis"]
            # Main statistics
            mean_Y = exp_results.get("mean_Y")
            std_Y = exp_results.get("std_Y")
            conf_int = exp_results.get("conf_int")
            skewness = exp_results.get("skewness")
            kurtosis = exp_results.get("kurtosis")
            quantiles = exp_results.get("quantiles") if "quantiles" in exp_results else None
            best_distribution = exp_results.get("best_distribution_name")
            best_params = exp_results.get("best_params")
            prob_exceedance = exp_results.get("prob_exceedance") if "prob_exceedance" in exp_results else None
            ai_insights = exp_results.get("ai_insights")
            
            context += "\n\n### Expectation Convergence Analysis Results\n"
            if mean_Y is not None and std_Y is not None:
                context += f"- **Estimated Mean Output:** {mean_Y:.4f}\n"
                context += f"- **Estimated Std Dev:** {std_Y:.4f}\n"
            if conf_int is not None:
                context += f"- **95% Confidence Interval:** [{conf_int[0]:.4f}, {conf_int[1]:.4f}]\n"
            if skewness is not None:
                context += f"- **Skewness:** {skewness:.4f}\n"
            if kurtosis is not None:
                context += f"- **Kurtosis:** {kurtosis:.4f}\n"
            if quantiles is not None:
                context += "- **Quantiles:**\n"
                for q, v in quantiles.items():
                    context += f"    - {q}: {v:.4f}\n"
            if best_distribution is not None:
                context += f"- **Best Fit Distribution:** {best_distribution}\n"
            if best_params is not None:
                context += f"- **Best Fit Parameters:** {best_params}\n"
            if prob_exceedance is not None:
                context += f"- **Probability of Exceeding Threshold:** {prob_exceedance}\n"
            if ai_insights is not None:
                context += f"\n#### AI Insights\n{ai_insights}\n"
            # Add output distribution table if available
            if "fit_df" in exp_results and exp_results["fit_df"] is not None:
                fit_df = exp_results["fit_df"]
                try:
                    context += "\n**Distribution Fit Results:**\n"
                    context += fit_df.to_markdown(index=False)
                except Exception:
                    pass
        
        # Add Exploratory Data Analysis results summary if available and relevant
        if "Exploratory Data Analysis" in st.session_state.all_results:
            eda_results = st.session_state.all_results["Exploratory Data Analysis"]
            # Correlation matrix
            corr_matrix = eda_results.get("display_corr")
            if corr_matrix is not None:
                context += "\n\n### Exploratory Data Analysis: Correlation Matrix\n"
                try:
                    context += corr_matrix.to_markdown(index=True)
                except Exception:
                    pass
            # Regression results
            if "regression_data" in eda_results:
                regression_data = eda_results["regression_data"]
                if isinstance(regression_data, list) and regression_data:
                    import pandas as pd
                    reg_df = pd.DataFrame(regression_data)
                    context += "\n\n### Input-Output Regression Summary\n"
                    context += reg_df.to_markdown(index=False)
            # AI insights
            ai_insights = eda_results.get("ai_insights")
            if ai_insights:
                context += f"\n#### AI Insights\n{ai_insights}\n"
            # Brief summary of available visualizations
            context += "\n- Cross cuts and regression plots are available for each input-output pair.\n- 2D cross cuts and contour plots visualize interactions between pairs of parameters.\n"
        
        # Add the user's question to the context
        context += f"\nPlease answer the user's question in the context of the full report. If necessary, refer to specific analyses by name.\n\nUser question: {prompt}"
        
        return context
    
    # Display existing chat messages in the sidebar
    for message in st.session_state.sidebar_global_chat_messages:
        with st.sidebar.chat_message(message["role"]):
            st.sidebar.write(message["content"])
    
    # Get user input in the sidebar
    sidebar_prompt = st.sidebar.chat_input("Ask a question about any analysis...", key="sidebar_chat_input")
    
    # Process user input
    if sidebar_prompt:
        # Add user message to chat history
        st.session_state.sidebar_global_chat_messages.append({"role": "user", "content": sidebar_prompt})
        
        # Generate context for the assistant
        context = sidebar_global_context_generator(sidebar_prompt)
        
        # Include previous conversation history
        chat_history = ""
        if len(st.session_state.sidebar_global_chat_messages) > 1:
            chat_history = "Previous conversation:\n"
            for i, msg in enumerate(st.session_state.sidebar_global_chat_messages[:-1]):
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n\n"
        
        # Create the final prompt
        chat_prompt = f"""
        {context}
        
        {chat_history}
        
        Current user question: {sidebar_prompt}
        
        Please provide a helpful, accurate response to this question.
        """
        
        # Call API with chat history
        with st.spinner("Thinking..."):
            try:
                response_text = call_groq_api(chat_prompt, model_name=selected_language_model)
            except Exception as e:
                st.sidebar.error(f"Error calling API: {str(e)}")
                response_text = "I'm sorry, I encountered an error while processing your question. Please try again."
        
        # Add assistant response to chat history
        st.session_state.sidebar_global_chat_messages.append({"role": "assistant", "content": response_text})
        
        # Rerun to display the new message immediately
        st.rerun()
else:
    st.sidebar.warning("Chat will be available after you run the Main Analysis.")

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
                    
                    # Add individual analysis buttons
                    st.subheader("Run Individual Analyses")
                    st.write("Select specific analyses to run individually based on your needs.")
                    
                    # Create helper functions for individual analyses
                    def run_model_understanding_analysis():
                        try:
                            with st.spinner("Running Model Understanding Analysis..."):
                                result = model_understanding(
                                    model, problem, current_code,
                                    language_model=selected_language_model, display_results=False
                                )
                                st.session_state.all_results["Model Understanding"] = result
                                st.session_state.analyses_ran = True
                                st.success("Model Understanding Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in Model Understanding Analysis: {str(e)}")
                    
                    def run_expectation_convergence_analysis():
                        try:
                            with st.spinner("Running Expectation Convergence Analysis..."):
                                from modules.expectation_convergence_analysis import (
                                    compute_expectation_convergence_analysis, generate_ai_insights
                                )
                                
                                # Compute the expectation convergence analysis
                                expectation_results = compute_expectation_convergence_analysis(
                                    model, problem, current_code, N_samples=8000
                                )
                                
                                # Generate AI insights
                                with st.spinner("Generating AI insights for Expectation Convergence Analysis..."):
                                    insights = generate_ai_insights(expectation_results, language_model=selected_language_model)
                                    expectation_results['ai_insights'] = insights
                                
                                # Ensure fit_df formatting if needed
                                if 'fit_df' in expectation_results and not expectation_results['fit_df'].empty:
                                    if not isinstance(expectation_results['fit_df'], pd.DataFrame):
                                        expectation_results['fit_df'] = pd.DataFrame(expectation_results['fit_df'])
                                    if 'OT_Distribution' in expectation_results['fit_df'].columns:
                                        expectation_results['fit_df'] = expectation_results['fit_df'].drop(columns=['OT_Distribution'])
                                
                                st.session_state.all_results["Expectation Convergence Analysis"] = expectation_results
                                st.session_state.analyses_ran = True
                                st.success("Expectation Convergence Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in Expectation Convergence Analysis: {str(e)}")
                    
                    def run_exploratory_data_analysis():
                        try:
                            with st.spinner("Running Exploratory Data Analysis..."):
                                # Ensure simulation data is available and convert it if needed
                                if 'simulation_data' in st.session_state and st.session_state.simulation_data is not None:
                                    df = create_monte_carlo_dataframe(st.session_state.simulation_data)
                                    result = exploratory_data_analysis(
                                        df, 1000, model, problem, current_code,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["Exploratory Data Analysis"] = result
                                    st.session_state.analyses_ran = True
                                    st.success("Exploratory Data Analysis completed!")
                                else:
                                    st.error("Simulation data not available. Please run the Monte Carlo simulation first.")
                        except Exception as e:
                            st.error(f"Error in Exploratory Data Analysis: {str(e)}")
                    
                    def run_sobol_analysis():
                        try:
                            with st.spinner("Running Sobol Sensitivity Analysis..."):
                                result = sobol_sensitivity_analysis(
                                    1024, model, problem, current_code,
                                    language_model=selected_language_model, display_results=False
                                )
                                st.session_state.all_results["Sobol Analysis"] = result
                                st.session_state.analyses_ran = True
                                st.success("Sobol Sensitivity Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in Sobol Sensitivity Analysis: {str(e)}")
                    
                    def run_fast_analysis():
                        try:
                            with st.spinner("Running FAST Sensitivity Analysis..."):
                                result = fast_sensitivity_analysis(
                                    size=1024, model=model, problem=problem, model_code_str=current_code,
                                    language_model=selected_language_model, display_results=False
                                )
                                st.session_state.all_results["FAST Analysis"] = result
                                st.session_state.analyses_ran = True
                                st.success("FAST Sensitivity Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in FAST Sensitivity Analysis: {str(e)}")
                    
                    def run_ancova_analysis():
                        try:
                            with st.spinner("Running ANCOVA Analysis..."):
                                result = ancova_analysis(
                                    1024, model, problem, current_code,
                                    language_model=selected_language_model, display_results=False
                                )
                                st.session_state.all_results["ANCOVA Analysis"] = result
                                st.session_state.analyses_ran = True
                                st.success("ANCOVA Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in ANCOVA Analysis: {str(e)}")
                    
                    def run_taylor_analysis():
                        try:
                            with st.spinner("Running Taylor Analysis..."):
                                result = taylor_analysis(
                                    model, problem, current_code,
                                    language_model=selected_language_model, display_results=False
                                )
                                st.session_state.all_results["Taylor Analysis"] = result
                                st.session_state.analyses_ran = True
                                st.success("Taylor Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in Taylor Analysis: {str(e)}")
                    
                    def run_correlation_analysis():
                        try:
                            with st.spinner("Running Correlation Analysis..."):
                                result = correlation_analysis(
                                    model, problem, current_code,
                                    size=1024, language_model=selected_language_model, display_results=False
                                )
                                st.session_state.all_results["Correlation Analysis"] = result
                                st.session_state.analyses_ran = True
                                st.success("Correlation Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in Correlation Analysis: {str(e)}")
                    
                    def run_hsic_analysis():
                        try:
                            with st.spinner("Running HSIC Analysis..."):
                                result = compute_hsic_analysis(
                                    hsic_size=200, model=model, problem=problem, model_code_str=current_code,
                                    language_model=selected_language_model
                                )
                                st.session_state.all_results["HSIC Analysis"] = result
                                st.session_state.analyses_ran = True
                                st.success("HSIC Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in HSIC Analysis: {str(e)}")
                    
                    def run_ml_analysis():
                        try:
                            with st.spinner("Running ML Analysis..."):
                                result = ml_analysis(
                                    model, problem, size=1000, model_code_str=current_code,
                                    language_model=selected_language_model, display_results=False
                                )
                                st.session_state.all_results["ML Analysis"] = result
                                st.session_state.analyses_ran = True
                                st.success("ML Analysis completed!")
                        except Exception as e:
                            st.error(f"Error in ML Analysis: {str(e)}")
                    
                    # Create a reset function
                    def reset_analyses():
                        if 'all_results' in st.session_state:
                            st.session_state.all_results = {}
                        st.session_state.analyses_ran = False
                        st.success("All analyses have been reset. You can now run them again.")
                    
                    # Create a grid of buttons for individual analyses
                    col1, col2, col3 = st.columns(3)
                    
                    if col1.button("🧠 Model Understanding", key="run_model_understanding"):
                        run_model_understanding_analysis()
                    
                    if col2.button("📊 Expectation Convergence", key="run_expectation_convergence"):
                        run_expectation_convergence_analysis()
                    
                    if col3.button("🔍 Exploratory Data Analysis", key="run_exploratory_data"):
                        run_exploratory_data_analysis()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if col1.button("🎯 Sobol Analysis", key="run_sobol"):
                        run_sobol_analysis()
                    
                    if col2.button("⚡ FAST Analysis", key="run_fast"):
                        run_fast_analysis()
                    
                    if col3.button("📈 ANCOVA Analysis", key="run_ancova"):
                        run_ancova_analysis()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if col1.button("📏 Taylor Analysis", key="run_taylor"):
                        run_taylor_analysis()
                    
                    if col2.button("🔗 Correlation Analysis", key="run_correlation"):
                        run_correlation_analysis()
                    
                    if col3.button("🧩 HSIC Analysis", key="run_hsic"):
                        run_hsic_analysis()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if col1.button("🤖 ML Analysis", key="run_ml"):
                        run_ml_analysis()
                    
                    if col3.button("🔄 Reset All Analyses", key="reset_analyses"):
                        reset_analyses()
                    
                    # Display the comprehensive report if analyses have been run
                    if st.session_state.analyses_ran and 'all_results' in st.session_state:
                        st.header("Comprehensive Analysis Report")
                        st.caption("This report contains the results of all analyses run on your model. Click on each section to expand and view detailed results.")
                        
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
                                                # Input distributions - REMOVED AS REQUESTED
                                                # if 'inputs_df' in results and not results['inputs_df'].empty:
                                                #     st.write("##### Input Distributions")
                                                #     st.dataframe(results['inputs_df'], hide_index=False, use_container_width=True)
                                                
                                                # Display the OpenTURNS markdown representation if available
                                                if 'problem' in results and hasattr(results['problem'], '__repr_markdown__'):
                                                    try:
                                                        markdown_repr = results['problem'].__repr_markdown__()
                                                        st.markdown(markdown_repr, unsafe_allow_html=True)
                                                    except:
                                                        pass
                                            else:
                                                st.info("No detailed results available for this analysis.")
                                    
                                    elif module_name == "Expectation Convergence Analysis":
                                        with st.container():
                                            if isinstance(results, dict):
                                                # Mean convergence visualization
                                                if 'mean_convergence_fig' in results:
                                                    st.write("##### Mean Convergence Analysis")
                                                    st.plotly_chart(results['mean_convergence_fig'], use_container_width=True)
                                                
                                                # Standard deviation convergence visualization
                                                if 'std_convergence_fig' in results:
                                                    st.write("##### Standard Deviation Convergence Analysis")
                                                    st.plotly_chart(results['std_convergence_fig'], use_container_width=True)
                                                
                                                # Distribution visualization
                                                if 'distribution_fig' in results:
                                                    st.write("##### Output Distribution Analysis")
                                                    
                                                    st.plotly_chart(results['distribution_fig'], use_container_width=True)
                                                
                                                # Distribution fitting results
                                                if 'fit_df' in results and not results['fit_df'].empty:
                                                    st.write("##### Distribution Fitting Results")
                                                    
                                                    # Create a display dataframe with just the key metrics
                                                    try:
                                                        display_df = results['fit_df'][['Distribution', 'AIC', 'BIC', 'KS_Statistic', 'KS_pvalue']].copy()
                                                        
                                                        # Format the numeric columns
                                                        display_df['AIC'] = display_df['AIC'].map('{:.2f}'.format)
                                                        display_df['BIC'] = display_df['BIC'].map('{:.2f}'.format)
                                                        display_df['KS_Statistic'] = display_df['KS_Statistic'].map('{:.4f}'.format)
                                                        display_df['KS_pvalue'] = display_df['KS_pvalue'].map('{:.4f}'.format)
                                                        
                                                        # Rename columns for display
                                                        display_df = display_df.rename(columns={
                                                            'KS_Statistic': 'KS Statistic',
                                                            'KS_pvalue': 'KS p-value'
                                                        })
                                                        
                                                        # Sort by AIC (best fit first)
                                                        display_df = display_df.sort_values('AIC')
                                                        
                                                        # Only show top 1 distribution
                                                        display_df = display_df.head(1)
                                                        
                                                        # Display the detailed distribution name if available
                                                        if 'display_name' in results and results['display_name']:
                                                            st.markdown(f"**Best Fit: {results['display_name']}**")
                                                        
                                                        # Display the dataframe
                                                        st.dataframe(display_df, hide_index=True, use_container_width=True)
                                                        
                                                        # Highlight the best distribution
                                                        if 'best_distribution_name' in results and results['best_distribution_name'] != "None":
                                                            # Get the OpenTURNS distribution type if available
                                                            ot_dist_type = results.get('ot_distribution_type', results['best_distribution_name'])
                                                                                                                    
                                                            # If we have parameters, display them
                                                            if 'best_params' in results and results['best_params']:
                                                                param_str = ", ".join([f"{p:.4f}" for p in results['best_params']])
                                                    except Exception as e:
                                                        # If there's an error formatting the dataframe, just display the original
                                                        st.warning(f"Error formatting distribution fitting results: {str(e)}")
                                                        st.dataframe(results['fit_df'], use_container_width=True)
                                                else:
                                                    st.info("No distribution fitting results available. This may happen if none of the standard distributions provided a good fit to the data.")
                                                
                                                # Input distributions - REMOVED AS REQUESTED
                                                # if 'inputs_df' in results and not results['inputs_df'].empty:
                                                #     st.write("##### Input Distributions")
                                                #     st.dataframe(results['inputs_df'], hide_index=True, use_container_width=True)
                                                
                                                # Create two columns for statistics tables
                                                col1, col2 = st.columns(2)
                                                
                                                # Convergence statistics
                                                if 'summary_df' in results:
                                                    with col1:
                                                        st.write("##### Convergence Statistics")
                                                        st.dataframe(results['summary_df'], hide_index=True, use_container_width=True)
                                                
                                                # Distribution statistics
                                                if 'dist_stats_df' in results:
                                                    with col2:
                                                        st.write("##### Distribution Statistics")
                                                        st.dataframe(results['dist_stats_df'], hide_index=True, use_container_width=True)
                                                
                                                # Quantiles
                                                if 'quantiles_df' in results:
                                                    st.write("##### Output Quantiles")
                                                    st.dataframe(results['quantiles_df'], hide_index=True, use_container_width=True)
                                                
                                                # AI insights
                                                if 'ai_insights' in results:
                                                    st.write("##### AI-Generated Insights")
                                                    st.markdown(results['ai_insights'])
                                                
                                                # Display any other available results
                                                for key, value in results.items():
                                                    if key not in ['mean_convergence_fig', 'std_convergence_fig', 'distribution_fig', 
                                                                  'summary_df', 'dist_stats_df', 'quantiles_df', 'fit_df', 'ai_insights',
                                                                  'inputs_df', 'Y_values', 'sample_sizes', 'mean_estimates', 'lower_bounds', 
                                                                  'upper_bounds', 'std_dev_estimates', 'final_std_dev', 'convergence_sample_size', 
                                                                  'mean_Y', 'std_Y', 'conf_int', 'skewness', 'kurtosis', 'q1', 'q3', 'iqr', 
                                                                  'best_distribution', 'best_params', 'best_distribution_name', 'input_parameters',
                                                                  'ot_distribution_type', 'display_name']:
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
                                                # Correlation matrix
                                                if 'fig_corr' in results:
                                                    st.write("##### Correlation Analysis")
                                                    st.plotly_chart(results['fig_corr'], use_container_width=True)
                                                
                                                # Combined cross cuts and regression plots
                                                if 'combined_plots' in results and results['combined_plots']:
                                                    st.write("##### Input-Output Relationships & Cross Cuts")
                                                    st.markdown("""
                                                    These plots show how each input parameter affects the output in two ways:
                                                    1. **Cross Cut** (left): Shows how the output changes when varying one input parameter while keeping all others at their nominal values
                                                    2. **Regression** (right): Shows the relationship between the input and output values from the Monte Carlo simulation
                                                    """)
                                                    
                                                    # Create output tabs if multiple outputs
                                                    if len(results['output_columns']) > 1:
                                                        output_tabs = st.tabs([results['display_names'][col] for col in results['output_columns']])
                                                        
                                                        for output_col, tab in zip(results['output_columns'], output_tabs):
                                                            with tab:
                                                                # Create tabs for each input variable
                                                                input_tabs = st.tabs(list(results['combined_plots'][output_col].keys()))
                                                                for i, (input_name, tab) in enumerate(zip(results['combined_plots'][output_col].keys(), input_tabs)):
                                                                    with tab:
                                                                        st.plotly_chart(results['combined_plots'][output_col][input_name], use_container_width=True)
                                                    else:
                                                        # If there's only one output, no need for output tabs
                                                        output_col = results['output_columns'][0]
                                                        
                                                        # Create tabs for each input variable
                                                        input_tabs = st.tabs(list(results['combined_plots'][output_col].keys()))
                                                        for i, (input_name, tab) in enumerate(zip(results['combined_plots'][output_col].keys(), input_tabs)):
                                                            with tab:
                                                                st.plotly_chart(results['combined_plots'][output_col][input_name], use_container_width=True)
                                                
                                                # 2D Cross cuts of the function
                                                if 'cross_cuts_2d' in results and results['cross_cuts_2d']:
                                                    st.write("##### 2D Cross Cuts of the Function")
                                                    st.markdown("""
                                                    These contour plots show how the output changes when varying two input parameters at a time,
                                                    while keeping all other parameters at their nominal values. This helps identify
                                                    interaction effects between pairs of parameters.
                                                    """)
                                                    
                                                    # Create tabs for each 2D cross cut
                                                    cross_cut_2d_tabs = st.tabs(list(results['cross_cuts_2d'].keys()))
                                                    for i, (input_pair, tab) in enumerate(zip(results['cross_cuts_2d'].keys(), cross_cut_2d_tabs)):
                                                        with tab:
                                                            st.plotly_chart(results['cross_cuts_2d'][input_pair], use_container_width=True)
                                                
                                                # AI insights
                                                if 'ai_insights' in results:
                                                    st.write("##### AI Insights")
                                                    st.markdown(results['ai_insights'])
                                            else:
                                                st.info("No detailed results available for this analysis.")
                                    
                                    elif module_name == "Sobol Analysis":
                                        with st.container():
                                            if isinstance(results, dict):
                                                # Display dimension information
                                                if 'dimension' in results:
                                                    st.metric("Number of Input Variables", results['dimension'])
                                                
                                                # Display variance decomposition metrics
                                                col1, col2, col3 = st.columns(3)
                                                if 'sum_first_order' in results:
                                                    with col1:
                                                        st.metric("Sum of First-Order Indices", f"{results['sum_first_order']:.4f}")
                                                if 'sum_total_order' in results:
                                                    with col2:
                                                        st.metric("Sum of Total-Order Indices", f"{results['sum_total_order']:.4f}")
                                                if 'interaction_effect' in results:
                                                    with col3:
                                                        st.metric("Interaction Effect", f"{results['interaction_effect']:.4f}")
                                                
                                                # Bar chart of sensitivity indices
                                                if 'fig_bar' in results:
                                                    st.write("##### Sobol Sensitivity Indices")
                                                    st.plotly_chart(results['fig_bar'], use_container_width=True)
                                                
                                                # Interaction chart
                                                if 'fig_interaction' in results:
                                                    st.write("##### Interaction Effects")
                                                    st.plotly_chart(results['fig_interaction'], use_container_width=True)
                                                
                                                # Heatmap for second-order indices
                                                if 'fig_heatmap' in results and results['fig_heatmap'] is not None:
                                                    st.write("##### Second-Order Interactions")
                                                    st.plotly_chart(results['fig_heatmap'], use_container_width=True)
                                                
                                                # Indices dataframe
                                                if 'indices_df' in results:
                                                    st.write("##### Sensitivity Indices Table")
                                                    st.dataframe(results['indices_df'], use_container_width=True)
                                                
                                                # Top interactions table
                                                if 'interactions_df' in results and results['interactions_df'] is not None:
                                                    st.write("##### Top Interaction Pairs")
                                                    st.dataframe(results['interactions_df'], use_container_width=True)
                                                
                                                # AI Insights
                                                if 'ai_insights' in results and results['ai_insights']:
                                                    st.write("##### AI-Generated Expert Analysis")
                                                    st.markdown(results['ai_insights'])
                                            else:
                                                st.info("No detailed results available for this analysis.")
                                    
                                    elif module_name == "Taylor Analysis":
                                        with st.container():
                                            if isinstance(results, dict):
                                                # Bar chart
                                                if 'bar_chart' in results:
                                                    st.write("##### Taylor Sensitivity Indices")
                                                    st.markdown("""
                                                    This bar chart shows the Taylor-based sensitivity indices for each input variable.
                                                    Higher values indicate variables with greater influence on the model output.
                                                    """)
                                                    st.plotly_chart(results['bar_chart'], use_container_width=True)
                                                
                                                # Validation plot
                                                if 'validation_plot' in results:
                                                    st.write("##### Taylor Approximation Validation")
                                                    st.markdown("""
                                                    This scatter plot compares the original model outputs with the Taylor approximation.
                                                    Points close to the diagonal line indicate good agreement between the model and its linear approximation.
                                                    """)
                                                    st.plotly_chart(results['validation_plot'], use_container_width=True)
                                                
                                                # Gradient plot
                                                if 'gradient_plot' in results:
                                                    st.write("##### Model Gradients at Nominal Point")
                                                    st.markdown("""
                                                    This bar chart shows the partial derivatives (gradients) of the model with respect to each input variable,
                                                    evaluated at the nominal point. The sign indicates whether increasing the variable increases (positive) or
                                                    decreases (negative) the output.
                                                    """)
                                                    st.plotly_chart(results['gradient_plot'], use_container_width=True)
                                                
                                                # Validation metrics
                                                if 'validation_metrics' in results:
                                                    st.write("##### Taylor Approximation Quality Metrics")
                                                    st.dataframe(results['validation_metrics'], use_container_width=True)
                                                
                                                # Taylor dataframe
                                                if 'taylor_df' in results:
                                                    st.write("##### Taylor Sensitivity Analysis Results")
                                                    st.dataframe(results['taylor_df'], use_container_width=True)
                                                
                                                # AI Insights
                                                if 'ai_insights' in results:
                                                    st.write("##### AI-Generated Expert Analysis")
                                                    st.markdown(results['ai_insights'])
                                                
                                                # Display any other available results
                                                for key, value in results.items():
                                                    if key not in ['bar_chart', 'validation_plot', 'gradient_plot', 'taylor_df', 'validation_metrics', 'ai_insights', 'input_names', 'nominal_point', 'nominal_value', 'gradients', 'variances', 'sensitivity_indices']:
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
                                                    st.markdown("""
                                                    This bar chart compares the First Order and Total Order sensitivity indices for each variable:
                                                    - **First Order Indices**: Measure the direct contribution of each variable to the output variance
                                                    - **Total Order Indices**: Measure the total contribution including interactions with other variables
                                                    """)
                                                    st.plotly_chart(results['fig_bar'], use_container_width=True)
                                                
                                                # Breakdown chart
                                                if 'fig_breakdown' in results:
                                                    st.write("##### FAST Sensitivity Indices Breakdown")
                                                    st.markdown("""
                                                    This stacked bar chart shows the breakdown of each variable's influence:
                                                    - **Direct Effect**: The first-order contribution (blue)
                                                    - **Interaction Effect**: The contribution due to interactions with other variables (red)
                                                    
                                                    The sum of both components equals the Total Order index for each variable.
                                                    """)
                                                    st.plotly_chart(results['fig_breakdown'], use_container_width=True)
                                                
                                                # Indices dataframe
                                                if 'indices_df' in results:
                                                    st.write("##### Sensitivity Indices Table")
                                                    st.dataframe(results['indices_df'], use_container_width=True)
                                                
                                                # Explanation
                                                if 'explanation' in results:
                                                    st.write("##### FAST Analysis Explanation")
                                                    st.markdown(results['explanation'])
                                                
                                                # AI Insights
                                                if 'llm_insights' in results:
                                                    st.write("##### AI Insights")
                                                    st.markdown(results['llm_insights'])
                                            else:
                                                st.info("No detailed results available for this analysis.")
                                    
                                    elif module_name == "ANCOVA Analysis":
                                        with st.container():
                                            if isinstance(results, dict):
                                                # Combined visualization
                                                if 'fig_combined' in results:
                                                    st.write("##### ANCOVA Sensitivity Analysis")
                                                    st.markdown("""
                                                    This grouped bar chart shows both the total ANCOVA sensitivity index ($S_i$) and its decomposition into 
                                                    uncorrelated ($S_i^U$) and correlated ($S_i^C$) parts for each variable.
                                                    """)
                                                    st.plotly_chart(results['fig_combined'], use_container_width=True)
                                                
                                                # Pie chart
                                                if 'fig_pie' in results:
                                                    st.write("##### ANCOVA Indices Distribution")
                                                    st.plotly_chart(results['fig_pie'], use_container_width=True)
                                                
                                                # Heatmap
                                                if 'fig_heatmap' in results:
                                                    st.write("##### Input Correlation Structure")
                                                    st.plotly_chart(results['fig_heatmap'], use_container_width=True)
                                                
                                                # Indices dataframe
                                                if 'indices_df' in results:
                                                    st.write("##### Sensitivity Indices Table")
                                                    st.dataframe(results['indices_df'], use_container_width=True)
                                                
                                                # Explanation
                                                if 'explanation' in results:
                                                    st.write("##### ANCOVA Explanation")
                                                    st.markdown(results['explanation'])
                                                
                                                # LLM Insights
                                                if 'llm_insights' in results:
                                                    st.write("##### AI Insights")
                                                    st.markdown(results['llm_insights'])
                                                
                                                # Display any other available results
                                                for key, value in results.items():
                                                    if key not in ['fig_combined', 'fig_pie', 'fig_heatmap', 'indices_df', 'explanation', 'llm_insights', 'correlation_matrix', 'variable_names', 'has_copula']:
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
                                                # Combined bar chart visualization
                                                if 'fig_bar' in results:
                                                    st.write("##### Correlation Coefficients")
                                                    st.markdown("""
                                                    This grouped bar chart shows different correlation coefficients between input variables and the output.
                                                    Both linear (Pearson, PCC, SRC) and rank-based (Spearman, PRCC, SRRC) correlation measures are displayed.
                                                    """)
                                                    st.plotly_chart(results['fig_bar'], use_container_width=True)
                                                
                                                # Correlation heatmap
                                                if 'fig_heatmap' in results:
                                                    st.write("##### Correlation Heatmap")
                                                    st.markdown("""
                                                    This heatmap shows the correlation structure between all variables (inputs and output).
                                                    Strong positive correlations appear in dark blue, while strong negative correlations appear in dark red.
                                                    """)
                                                    st.plotly_chart(results['fig_heatmap'], use_container_width=True)
                                                
                                                # Linear vs Rank correlation comparison
                                                if 'fig_comparison' in results:
                                                    st.write("##### Linear vs. Rank Correlation")
                                                    st.markdown("""
                                                    This scatter plot compares linear (Pearson) and rank-based (Spearman) correlation coefficients.
                                                    Points along the diagonal indicate variables where both measures agree, while deviations suggest non-linear relationships.
                                                    """)
                                                    st.plotly_chart(results['fig_comparison'], use_container_width=True)
                                                
                                                # Correlation dataframe
                                                if 'correlation_df' in results:
                                                    st.write("##### Correlation Coefficients Table")
                                                    st.dataframe(results['correlation_df'], use_container_width=True)
                                                
                                                # AI Insights
                                                if 'ai_insights' in results:
                                                    st.write("##### AI-Generated Expert Analysis")
                                                    st.markdown(results['ai_insights'])
                                                # For backward compatibility
                                                elif 'llm_insights' in results:
                                                    st.write("##### AI Insights")
                                                    st.markdown(results['llm_insights'])
                                                
                                                # Display any other available results
                                                for key, value in results.items():
                                                    if key not in ['fig_bar', 'fig_heatmap', 'fig_comparison', 'correlation_df', 'ai_insights', 'llm_insights']:
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
                                                # Model performance metrics
                                                col1, col2, col3 = st.columns(3)
                                                if 'top_feature' in results:
                                                    with col1:
                                                        st.metric("Most Influential Variable", results['top_feature'])
                                                if 'significant_features_count' in results:
                                                    with col2:
                                                        st.metric("Significant Variables", f"{results['significant_features_count']}")
                                                if 'performance_metrics' in results and 'r2' in results['performance_metrics']:
                                                    with col3:
                                                        st.metric("Model Accuracy", f"{results['performance_metrics']['r2']:.4f} R²")
                                                
                                                # SHAP bar plot (Feature Importance)
                                                if 'shap_bar_plot' in results:
                                                    st.write("##### Feature Importance")
                                                    st.plotly_chart(results['shap_bar_plot'], use_container_width=True)
                                                
                                                # SHAP summary plot (Feature Dependence)
                                                if 'shap_summary_plot' in results:
                                                    st.write("##### Feature Dependence")
                                                    st.plotly_chart(results['shap_summary_plot'], use_container_width=True)
                                                
                                                # Validation plot
                                                if 'validation_plot' in results:
                                                    st.write("##### Model Validation")
                                                    st.plotly_chart(results['validation_plot'], use_container_width=True)
                                                
                                                # Model metrics table (only if we don't have the individual metrics)
                                                if 'model_metrics' in results and not ('performance_metrics' in results and 'r2' in results['performance_metrics']):
                                                    st.write("##### Model Performance Metrics")
                                                    st.dataframe(results['model_metrics'], use_container_width=True)
                                                
                                                # AI Insights
                                                if 'ai_insights' in results and results['ai_insights']:
                                                    st.write("##### AI-Generated Expert Analysis")
                                                    st.markdown(results['ai_insights'])
                                            else:
                                                st.info("No detailed results available for this analysis.")
                                    
                                    # Add a separator between modules
                                    st.write("---")
                        
                        # No need for the chat interface in the main page as it's now in the sidebar
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
