import os
import streamlit as st
# Ensure set_page_config is the first Streamlit command
if 'page_config_set' not in st.session_state:
    st.set_page_config(
        page_title="UncertaintyCat | UQ Made Easy",
        page_icon="🐱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state.page_config_set = True

import numpy as np
import openturns as ot
import pandas as pd
import plotly.express as px
import re
import traceback
# import os # os is imported twice, remove one

# Import modules
from modules.monte_carlo import monte_carlo_simulation, create_monte_carlo_dataframe
from modules.model_understanding import model_understanding, display_model_understanding
from modules.exploratory_data_analysis import exploratory_data_analysis, display_exploratory_data_analysis_results
from modules.expectation_convergence_analysis import expectation_convergence_analysis_joint, display_expectation_convergence_results, compute_expectation_convergence_analysis, generate_ai_insights # Added compute and generate
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis, display_sobol_results
from modules.taylor_analysis import taylor_analysis, display_taylor_results
from modules.correlation_analysis import correlation_analysis, display_correlation_results
from modules.hsic_analysis import compute_hsic_analysis, display_hsic_results
from modules.ml_analysis import ml_analysis, display_ml_results
from modules.morris_analysis import morris_analysis, dimensionality_reduction_page
from modules.fast_analysis import fast_analysis, fast_sensitivity_analysis, display_fast_results # fast_sensitivity_analysis might be an alternative entry or part of fast_analysis
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
    total_analyses = 10 # Ensure this matches the number of analyses below
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
        # Removed st.empty() for pause_text as it was brief and might not be necessary
        time.sleep(1)  # 1 second pause

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
        try:
            expectation_results = compute_expectation_convergence_analysis(
                model, problem, current_code, N_samples=8000
            )
            with st.spinner("Generating AI insights for Expectation Convergence Analysis..."):
                insights = generate_ai_insights(expectation_results, language_model=selected_language_model)
                expectation_results['ai_insights'] = insights
            
            if 'fit_df' in expectation_results and not expectation_results['fit_df'].empty:
                if not isinstance(expectation_results['fit_df'], pd.DataFrame):
                    expectation_results['fit_df'] = pd.DataFrame(expectation_results['fit_df'])
                if 'OT_Distribution' in expectation_results['fit_df'].columns:
                    expectation_results['fit_df'] = expectation_results['fit_df'].drop(columns=['OT_Distribution'])
            st.session_state.all_results["Expectation Convergence Analysis"] = expectation_results
        except Exception as e:
            st.error(f"Error in Expectation Convergence Analysis: {str(e)}")
            st.session_state.all_results["Expectation Convergence Analysis"] = {"error": str(e)}
        pause_between_analyses()

        # Exploratory Data Analysis
        update_progress("Exploratory Data Analysis")
        if 'simulation_data' in st.session_state and st.session_state.simulation_data is not None:
            df = create_monte_carlo_dataframe(st.session_state.simulation_data)
            st.session_state.all_results["Exploratory Data Analysis"] = exploratory_data_analysis(
                df, 1000, model, problem, current_code,
                language_model=selected_language_model, display_results=False
            )
        else:
            st.session_state.all_results["Exploratory Data Analysis"] = {"error": "Monte Carlo simulation data not available for EDA."}
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
            language_model=selected_language_model # Assuming display_results is handled internally or not applicable
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
        st.session_state.analyses_ran = False # Indicate failure
    
    # Final update
    progress_bar.progress(1.0)
    if st.session_state.get('analyses_ran', False):
        status_text.markdown('<div style="background-color: #d4edda; border-left: 5px solid #28a745; padding: 10px; border-radius: 4px;">All analyses completed! You can now view the comprehensive report below.</div>', unsafe_allow_html=True)
    else:
        status_text.error("One or more analyses failed. Please check the errors above or in the report section.")

    # Force a rerun to update the sidebar chat interface and main page report
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
# 2) LOAD MODEL CODE FROM EXAMPLES (Assumed to be handled by streamlit_app_start or elsewhere)
###############################################################################


###############################################################################
# 3) STREAMLIT APP START
###############################################################################

from utils.streamlit_app_start import streamlit_app_start

# Run app-wide Streamlit configuration and sidebar setup
current_code, selected_language_model, selected_page, dropdown_items = streamlit_app_start()

# Load CSS
load_css() # Assuming load_css() is defined in utils.css_styles and handles "running-indicator"

from utils.side_panel import render_sidebar_chat

# Add the global chat interface to the sidebar (now modularized)
render_sidebar_chat(current_code, selected_language_model)

from utils.code_editor import render_code_editor
# from utils.model_loader import load_model_code # This import was present but function not directly used in snippet

# Render the code editor and update current_code accordingly
current_code = render_code_editor(current_code)


###############################################################################
# 7) PAGE-SPECIFIC CONTENT
###############################################################################
if "📊 UQ Dashboard" in selected_page:
    
    if not current_code:
        st.info("Please select or upload a model file to begin analysis.")
    else:
        is_safe, safety_message = check_code_safety(current_code)
        
        if not is_safe:
            st.error(f"Security Error: {safety_message}")
        else:
            try:
                eval_globals = {}
                exec(current_code, eval_globals)
                model = eval_globals.get('model')
                problem = eval_globals.get('problem')
                
                if not model or not problem:
                    st.error("Model code must define 'model' and 'problem' variables.")
                else:
                    # Run Monte Carlo simulation if data doesn't exist (prerequisite for EDA)
                    if st.session_state.simulation_data is None:
                        with st.spinner("Running Initial Monte Carlo Simulation for EDA..."):
                            N_mc_initial = 1000 
                            data_mc = monte_carlo_simulation(model, problem, N_mc_initial)
                            st.session_state.simulation_data = data_mc
                            # st.success("Initial Monte Carlo Simulation completed.") # Optional: can be noisy

                    st.markdown("---")
                    st.header("🚀 Run All Analyses")
                    if st.button("Run Full UQ Suite", key="run_all_analyses_button", help="Runs all UQ analyses and generates a comprehensive report below."):
                        st.session_state.analyses_ran = False # Reset flag before starting full run
                        # st.session_state.all_results = {} # Optional: Clear previous results if a full run should always be fresh
                        run_all_analyses(model, problem, current_code, selected_language_model)
                        # run_all_analyses handles st.rerun()

                    st.markdown("---")
                    st.header("🔬 Run Individual Analysis Modules")
                    st.caption("Click a button to run a specific analysis. Results will be stored and displayed in the report section below.")

                    # --- Individual Analysis Buttons ---
                    cols1 = st.columns(3)
                    with cols1[0]:
                        if st.button("Model Understanding", key="run_model_understanding"):
                            with st.spinner("Running Model Understanding..."):
                                try:
                                    results = model_understanding(
                                        model, problem, current_code,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["Model Understanding"] = results
                                    st.success("Model Understanding completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["Model Understanding"] = {"error": str(e)}
                                st.rerun()
                        
                        if st.button("Exploratory Data Analysis", key="run_eda"):
                            if st.session_state.simulation_data is not None:
                                with st.spinner("Running Exploratory Data Analysis..."):
                                    try:
                                        df = create_monte_carlo_dataframe(st.session_state.simulation_data)
                                        results = exploratory_data_analysis(
                                            df, 1000, model, problem, current_code,
                                            language_model=selected_language_model, display_results=False
                                        )
                                        st.session_state.all_results["Exploratory Data Analysis"] = results
                                        st.success("Exploratory Data Analysis completed.")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")
                                        st.session_state.all_results["Exploratory Data Analysis"] = {"error": str(e)}
                                    st.rerun()
                            else:
                                st.warning("Monte Carlo data needed for EDA. It should run automatically if model is loaded.")
                                st.rerun()

                    with cols1[1]:
                        if st.button("Expectation Convergence", key="run_exp_conv"):
                            with st.spinner("Running Expectation Convergence Analysis..."):
                                try:
                                    exp_results = compute_expectation_convergence_analysis(
                                        model, problem, current_code, N_samples=8000
                                    )
                                    with st.spinner("Generating AI insights..."):
                                        insights = generate_ai_insights(exp_results, language_model=selected_language_model)
                                        exp_results['ai_insights'] = insights
                                    if 'fit_df' in exp_results and not exp_results['fit_df'].empty:
                                        if not isinstance(exp_results['fit_df'], pd.DataFrame):
                                            exp_results['fit_df'] = pd.DataFrame(exp_results['fit_df'])
                                        if 'OT_Distribution' in exp_results['fit_df'].columns:
                                            exp_results['fit_df'] = exp_results['fit_df'].drop(columns=['OT_Distribution'])
                                    st.session_state.all_results["Expectation Convergence Analysis"] = exp_results
                                    st.success("Expectation Convergence Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["Expectation Convergence Analysis"] = {"error": str(e)}
                                st.rerun()
                        
                        if st.button("Correlation Analysis", key="run_correlation"):
                            with st.spinner("Running Correlation Analysis..."):
                                try:
                                    results = correlation_analysis(
                                        model, problem, current_code, size=1024,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["Correlation Analysis"] = results
                                    st.success("Correlation Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["Correlation Analysis"] = {"error": str(e)}
                                st.rerun()

                    with cols1[2]:
                        if st.button("Taylor Analysis (Local)", key="run_taylor"):
                            with st.spinner("Running Taylor Analysis..."):
                                try:
                                    results = taylor_analysis(
                                        model, problem, current_code,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["Taylor Analysis"] = results
                                    st.success("Taylor Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["Taylor Analysis"] = {"error": str(e)}
                                st.rerun()

                        if st.button("HSIC Analysis", key="run_hsic"):
                             with st.spinner("Running HSIC Analysis..."):
                                try:
                                    results = compute_hsic_analysis(
                                        hsic_size=200, model=model, problem=problem, model_code_str=current_code,
                                        language_model=selected_language_model
                                    )
                                    st.session_state.all_results["HSIC Analysis"] = results
                                    st.success("HSIC Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["HSIC Analysis"] = {"error": str(e)}
                                st.rerun()
                    
                    st.markdown("---") # Separator before next row of buttons
                    cols2 = st.columns(3)
                    with cols2[0]:
                        if st.button("Sobol Analysis (Variance)", key="run_sobol"):
                            with st.spinner("Running Sobol Sensitivity Analysis..."):
                                try:
                                    results = sobol_sensitivity_analysis(
                                        1024, model, problem, current_code,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["Sobol Analysis"] = results
                                    st.success("Sobol Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["Sobol Analysis"] = {"error": str(e)}
                                st.rerun()
                    
                    with cols2[1]:
                        if st.button("FAST Analysis (Variance)", key="run_fast"):
                            with st.spinner("Running FAST Sensitivity Analysis..."):
                                try:
                                    results = fast_analysis(
                                        model, problem, model_code_str=current_code,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["FAST Analysis"] = results
                                    st.success("FAST Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["FAST Analysis"] = {"error": str(e)}
                                st.rerun()

                    with cols2[2]:
                        if st.button("ANCOVA Analysis", key="run_ancova"):
                            with st.spinner("Running ANCOVA Sensitivity Analysis..."):
                                try:
                                    results = ancova_analysis(
                                        model, problem, model_code_str=current_code,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["ANCOVA Analysis"] = results
                                    st.success("ANCOVA Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["ANCOVA Analysis"] = {"error": str(e)}
                                st.rerun()

                    st.markdown("---") # Separator before next row of buttons
                    cols3 = st.columns(3)
                    with cols3[0]:
                        if st.button("Shapley Analysis (ML)", key="run_shapley"):
                            with st.spinner("Running Shapley Analysis..."):
                                try:
                                    results = ml_analysis(
                                        model, problem, size=1000, model_code_str=current_code,
                                        language_model=selected_language_model, display_results=False
                                    )
                                    st.session_state.all_results["Shapley Analysis"] = results
                                    st.success("Shapley Analysis completed.")
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                                    st.session_state.all_results["Shapley Analysis"] = {"error": str(e)}
                                st.rerun()
                    
                    st.markdown("---")
                    # Display the comprehensive/stored results report
                    if 'all_results' in st.session_state and st.session_state.all_results:
                        is_full_run_complete = st.session_state.get('analyses_ran', False)
                        report_title = "Comprehensive Analysis Report" if is_full_run_complete else "Stored Analysis Results"
                        
                        report_caption = ("This report contains the results of all UQ analyses. Click on each section to expand."
                                          if is_full_run_complete 
                                          else "This section shows results from individually run analyses or a previous full UQ run. Click 'Run Full UQ Suite' for a fresh comprehensive report. Expand sections for details.")
                        
                        st.header(f"📊 {report_title}")
                        st.caption(report_caption)
                        
                        sorted_module_names = list(st.session_state.all_results.keys()) # Or define a preferred order
                        
                        for module_name in sorted_module_names: # Iterate in a defined or sorted order if desired
                            results_data = st.session_state.all_results[module_name]
                            if results_data:  # Only display if results exist for this module
                                is_error = "error" in results_data
                                expander_title = f"⚠️ {module_name} (Error)" if is_error else f"📄 {module_name}"
                                # Expand errors by default, or if it was the last run item (optional)
                                expanded_default = is_error 

                                with st.expander(expander_title, expanded=expanded_default):
                                    if is_error:
                                        st.error(f"An error occurred during the '{module_name}' analysis: {results_data['error']}")
                                    else:
                                        st.subheader(f"{module_name} Results")
                                        # Dynamically call display functions
                                        if module_name == "Model Understanding":
                                            display_model_understanding(results_data)
                                        elif module_name == "Expectation Convergence Analysis":
                                            display_expectation_convergence_results(results_data)
                                        elif module_name == "Exploratory Data Analysis":
                                            display_exploratory_data_analysis_results(results_data)
                                        elif module_name == "Sobol Analysis":
                                            display_sobol_results(results_data, None) 
                                        elif module_name == "Taylor Analysis":
                                            display_taylor_results(results_data)
                                        elif module_name == "FAST Analysis":
                                            display_fast_results(results_data)
                                        elif module_name == "ANCOVA Analysis":
                                            display_ancova_results(results_data)
                                        elif module_name == "Correlation Analysis":
                                            display_correlation_results(results_data)
                                        elif module_name == "HSIC Analysis":
                                            display_hsic_results(results_data)
                                        elif module_name == "Shapley Analysis":
                                            display_ml_results(results_data)
                                        # Add other modules if any were missed
                    elif current_code and model and problem: # Model loaded, but no results yet
                         st.info("Run the 'Full UQ Suite' or individual analysis modules to generate and view reports.")
            
            except Exception as e:
                st.error(f"Error evaluating model code: {e}")
                # traceback.print_exc() # For debugging, if needed

elif "📉 Dimensionality Reduction" in selected_page:
    # Use the modular dimensionality reduction page function from morris_analysis module
    if not current_code:
        st.info("Please select or upload a model file to perform dimensionality reduction.")
    else:
        is_safe, safety_message = check_code_safety(current_code)
        if not is_safe:
            st.error(f"Security Error: {safety_message}")
        else:
            try:
                eval_globals = {}
                exec(current_code, eval_globals)
                model = eval_globals.get('model')
                problem = eval_globals.get('problem')
                
                if not model or not problem:
                    st.error("Model code must define 'model' and 'problem' variables.")
                else:
                    dimensionality_reduction_page(current_code, model, problem, selected_language_model)
            except Exception as e:
                st.error(f"Error evaluating model code for Dimensionality Reduction: {e}")

elif "📈 Distribution Fitting" in selected_page:
    # Use the distribution fitting page function
    distribution_fitting_page()

# elif "📐 PCE Least-Squares" in selected_page:
#  ... (Your existing PCE code, no changes requested for it here) ...
#  Remember to add a button for it in the UQ Dashboard if you want individual PCE runs there too.
#  Example:
#  if st.button("PCE Least-Squares Analysis", key="run_pce"):
#      with st.spinner("Running PCE Least-Squares Analysis..."):
#          try:
#              # Assuming you'd define N_train, N_validate etc. here or retrieve from UI elements
#              # For simplicity, using defaults or fixed values for an individual button:
#              pce_N_train = st.session_state.get('pce_N_train', 1000) # Example: get from session or use default
#              pce_N_validate = st.session_state.get('pce_N_validate', 1000)
#              pce_basis_factor = st.session_state.get('pce_basis_factor', 0.5)
#              pce_use_model_sel = st.session_state.get('pce_use_model_sel', False)

#              results = pce_least_squares_analysis(
#                  model, problem, current_code,
#                  N_train=pce_N_train, N_validate=pce_N_validate,
#                  basis_size_factor=pce_basis_factor, use_model_selection=pce_use_model_sel,
#                  language_model=selected_language_model, display_results=False
#              )
#              st.session_state.all_results["PCE Least-Squares"] = results
#              st.success("PCE Least-Squares Analysis completed.")
#          except Exception as e:
#              st.error(f"Error in PCE Analysis: {str(e)}")
#              st.session_state.all_results["PCE Least-Squares"] = {"error": str(e)}
#          st.rerun()
#
# And ensure "PCE Least-Squares" is handled in the display loop:
# elif module_name == "PCE Least-Squares":
# from modules.pce_least_squares import display_pce_results # Ensure this import is available
# display_pce_results(results_data)