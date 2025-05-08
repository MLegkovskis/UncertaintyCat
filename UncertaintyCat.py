import os
import streamlit as st
st.set_page_config(
    page_title="UncertaintyCat | UQ Made Easy",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)
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
from modules.pce_least_squares import pce_least_squares_analysis, display_pce_results

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

        # Shapley Analysis
        update_progress("Shapley Analysis")
        st.session_state.all_results["Shapley Analysis"] = ml_analysis(
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


###############################################################################
# 3) STREAMLIT APP START
###############################################################################

from utils.streamlit_app_start import streamlit_app_start

# Run app-wide Streamlit configuration and sidebar setup
current_code, selected_language_model, selected_page, dropdown_items = streamlit_app_start()

from utils.side_panel import render_sidebar_chat

# Add the global chat interface to the sidebar (now modularized)
render_sidebar_chat(current_code, selected_language_model)

from utils.code_editor import render_code_editor
from utils.model_loader import load_model_code

# Render the code editor and update current_code accordingly
current_code = render_code_editor(current_code)


###############################################################################
# 7) PAGE-SPECIFIC CONTENT
###############################################################################
if "📊 UQ Dashboard" in selected_page:
    
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
                    

                    if st.button("🚀 Run UQ", key="run_all_analyses", help="Runs all UQ analyses and generates a comprehensive report."):
                        run_all_analyses(model, problem, current_code, selected_language_model)
                    
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
                                        from modules.model_understanding import display_model_understanding
                                        display_model_understanding(results)
                                    
                                    elif module_name == "Expectation Convergence Analysis":
                                        from modules.expectation_convergence_analysis import display_expectation_convergence_results
                                        display_expectation_convergence_results(results)

                                    
                                    elif module_name == "Exploratory Data Analysis":
                                        from modules.exploratory_data_analysis import display_exploratory_data_analysis_results
                                        display_exploratory_data_analysis_results(results)

                                    
                                    elif module_name == "Sobol Analysis":
                                        from modules.sobol_sensitivity_analysis import display_sobol_results
                                        display_sobol_results(results, None)

                                    
                                    elif module_name == "Taylor Analysis":
                                        from modules.taylor_analysis import display_taylor_results
                                        display_taylor_results(results)

                                    
                                    elif module_name == "FAST Analysis":
                                        from modules.fast_analysis import display_fast_results
                                        display_fast_results(results)

                                    
                                    elif module_name == "ANCOVA Analysis":
                                        from modules.ancova_analysis import display_ancova_results
                                        display_ancova_results(results)

                                    
                                    elif module_name == "Correlation Analysis":
                                        from modules.correlation_analysis import display_correlation_results
                                        display_correlation_results(results)

                                    
                                    elif module_name == "HSIC Analysis":
                                        from modules.hsic_analysis import display_hsic_results
                                        display_hsic_results(results)

                                    
                                    elif module_name == "Shapley Analysis":
                                        from modules.ml_analysis import display_ml_results
                                        display_ml_results(results)
                        
                        # No need for the chat interface in the main page as it's now in the sidebar
                    else:
                        st.info("Please run UQ first by clicking the '🚀 Run UQ' button.")
                        
            except Exception as e:
                st.error(f"Error evaluating model code: {e}")

elif "📉 Dimensionality Reduction" in selected_page:
    # Use the modular dimensionality reduction page function from morris_analysis module
    # First check if the code is safe to execute
    is_safe, safety_message = check_code_safety(current_code)
    
    if not is_safe:
        st.error(f"Security Error: {safety_message}")
    else:
        # Try to execute the code to get model and problem
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
                dimensionality_reduction_page(current_code, model, problem, selected_language_model)
        except Exception as e:
            st.error(f"Error evaluating model code: {e}")

elif "📈 Distribution Fitting" in selected_page:
    # Use the distribution fitting page function
    distribution_fitting_page()

# elif "📐 PCE Least-Squares" in selected_page:
#     # ------------------------------------------------------------------ #
#     #  PCE surrogate-model generation & inspection                       #
#     # ------------------------------------------------------------------ #
#     is_safe, safety_message = check_code_safety(current_code)

#     if not is_safe:
#         st.error(f"Security Error: {safety_message}")
#     else:
#         try:
#             # Execute the user-supplied model code
#             eval_globals = {}
#             exec(current_code, eval_globals)
#             model   = eval_globals.get("model")
#             problem = eval_globals.get("problem")

#             if not model or not problem:
#                 st.error("Model code must define 'model' and 'problem' variables.")
#             else:
#                 # UI parameters ------------------------------------------------
#                 st.subheader("PCE configuration")
#                 N_train           = st.number_input("Training sample size",    200,  10000, 1000, 200)
#                 N_validate        = st.number_input("Validation sample size",  200,  10000, 1000, 200)
#                 basis_size_factor = st.slider("Basis size factor", 0.05, 1.0, 0.5, 0.05)
#                 use_model_sel     = st.checkbox("Use LARS model-selection", False)


#                 if st.button("🚀 Build PCE surrogate"):
#                     results = pce_least_squares_analysis(
#                         model,
#                         problem,
#                         current_code,
#                         N_train           = N_train,
#                         N_validate        = N_validate,
#                         basis_size_factor = basis_size_factor,
#                         use_model_selection = use_model_sel,
#                         language_model    = selected_language_model,
#                         display_results   = False,          # <── change here
#                     )
#                     st.session_state.all_results["PCE Least-Squares"] = results

#                 # Show previously-computed results (only one set of charts)
#                 if "PCE Least-Squares" in st.session_state.all_results:
#                     st.markdown("---")
#                     display_pce_results(st.session_state.all_results["PCE Least-Squares"])

#         except Exception as e:
#             st.error(f"Error evaluating model code: {e}")
