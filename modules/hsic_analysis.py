import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px # Included for consistency, though not directly used in plots
from plotly.subplots import make_subplots
import traceback
from utils.core_utils import call_groq_api
from utils.constants import RETURN_INSTRUCTION

def compute_hsic_indices(model, problem, N=200, seed=42):
    """
    Compute HSIC-based sensitivity indices using OpenTURNS built-in tools.
    (Content improved based on previous refinements)
    """
    if not callable(model):
        raise TypeError("The 'model' argument must be a callable function.")
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution.")

    dimension = problem.getDimension()
    input_names = [problem.getMarginal(i).getDescription()[0] if problem.getMarginal(i).getDescription() and problem.getMarginal(i).getDescription()[0] else f"X{i+1}" for i in range(dimension)]

    ot.RandomGenerator.SetSeed(seed)
    X_sample = problem.getSample(N)

    ot_model = model
    if not isinstance(model, ot.Function):
        ot_model = ot.PythonFunction(dimension, 1, model)

    Y_sample = ot_model(X_sample)
    if Y_sample.getDimension() != 1:
        # Prefer st.sidebar.warning if this is called from a UI context,
        # but since this function is core, a print or direct st.warning is okay.
        # For a library function, raising an error or allowing multi-output HSIC might be options.
        # For now, use first output as per OpenTURNS examples for global SA.
        print(f"Warning: HSIC analysis expects a scalar output, but model output dimension is {Y_sample.getDimension()}. Using the first component.")
        Y_sample = Y_sample.getMarginal(0) # Ensure Y_sample is 1D for HSICEstimatorGlobalSensitivity

    covarianceModelCollection = []
    for i in range(dimension):
        Xi_marginal_sample = X_sample.getMarginal(i)
        # Using SquaredExponential kernel with a single length-scale parameter.
        # OpenTURNS expects a list of length-scales, one for each input dimension of THIS kernel (which is 1D here).
        inputCovariance = ot.SquaredExponential([1.0]) # Default length-scale
        sd_xi = Xi_marginal_sample.computeStandardDeviation()
        if sd_xi[0] > 1e-9: # Avoid division by zero or tiny scales
            # The 'scale' in SquaredExponential is length-scale, not amplitude.
            # Setting length-scale based on std dev is a common heuristic.
            inputCovariance.setScale(sd_xi)
        covarianceModelCollection.append(inputCovariance)

    outputCovariance = ot.SquaredExponential([1.0]) # Default length-scale for output kernel
    sd_y = Y_sample.computeStandardDeviation()
    if sd_y[0] > 1e-9:
        outputCovariance.setScale(sd_y)
    covarianceModelCollection.append(outputCovariance)

    estimatorType = ot.HSICVStat()
    hsicEstimator = ot.HSICEstimatorGlobalSensitivity(
        covarianceModelCollection, X_sample, Y_sample, estimatorType
    )

    # The get... methods return ot.Point objects, convert to numpy arrays for consistency
    results = {
        'hsic_indices': np.array(hsicEstimator.getHSICIndices()),
        'normalized_indices': np.array(hsicEstimator.getR2HSICIndices()), # R2-HSIC
        'p_values_asymptotic': np.array(hsicEstimator.getPValuesAsymptotic()),
        'p_values_permutation': np.array(hsicEstimator.getPValuesPermutation()),
        'input_names': input_names,
        # 'hsic_estimator': hsicEstimator # Storing estimator is optional
    }
    # Debug prints from your original code - useful during development
    # print(f"HSIC indices shape: {results['hsic_indices'].shape}")
    # print(f"Normalized indices shape: {results['normalized_indices'].shape}")
    # print(f"P-values asymptotic shape: {results['p_values_asymptotic'].shape}")
    # print(f"P-values permutation shape: {results['p_values_permutation'].shape}")
    # print(f"P-values permutation: {results['p_values_permutation']}")
    return results

def create_hsic_dataframe(results):
    """
    Create a DataFrame with HSIC analysis results.
    (Column names improved for clarity)
    """
    if not all(k in results for k in ['input_names', 'hsic_indices', 'normalized_indices', 'p_values_asymptotic', 'p_values_permutation']):
        raise ValueError("HSIC results dictionary is missing required keys.")

    df = pd.DataFrame({
        'Variable': results['input_names'],
        'Raw_HSIC': results['hsic_indices'].flatten(), # Ensure 1D
        'R2_HSIC_Index': results['normalized_indices'].flatten(), # Ensure 1D, R2-HSIC is normalized
        'P_Value_Asymptotic': results['p_values_asymptotic'].flatten(), # Ensure 1D
        'P_Value_Permutation': results['p_values_permutation'].flatten() # Ensure 1D
    })
    df = df.sort_values('R2_HSIC_Index', ascending=False).reset_index(drop=True)
    return df

def create_hsic_plots(results):
    """
    Create interactive Plotly visualizations for HSIC results.
    (Content improved based on previous refinements)
    """
    # Use the dataframe for plotting as it's already sorted and structured
    df_for_plot = create_hsic_dataframe(results)
    if df_for_plot.empty:
        return go.Figure() # Return empty figure if no data

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "R²-HSIC Indices (Normalized)", "Raw HSIC Values",
            "Asymptotic p-values", "Permutation p-values"
        ),
        vertical_spacing=0.22, horizontal_spacing=0.15 # Increased spacing
    )

    # R2-HSIC (Normalized)
    fig.add_trace(go.Bar(
        x=df_for_plot['Variable'], y=df_for_plot['R2_HSIC_Index'], name="R²-HSIC Index",
        marker_color='rgb(0,102,204)', hovertemplate="<b>%{x}</b><br>R²-HSIC: %{y:.4f}<extra></extra>"
    ), row=1, col=1)

    # Raw HSIC
    fig.add_trace(go.Bar(
        x=df_for_plot['Variable'], y=df_for_plot['Raw_HSIC'], name="Raw HSIC Value",
        marker_color='rgb(26,118,188)', hovertemplate="<b>%{x}</b><br>Raw HSIC: %{y:.4e}<extra></extra>"
    ), row=1, col=2)

    # Asymptotic p-values
    fig.add_trace(go.Bar(
        x=df_for_plot['Variable'], y=df_for_plot['P_Value_Asymptotic'], name="Asymptotic p-value",
        marker_color='rgb(219,64,82)', hovertemplate="<b>%{x}</b><br>p-value (Asymp.): %{y:.4e}<extra></extra>"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df_for_plot['Variable'], y=[0.05] * len(df_for_plot), mode='lines',
        line=dict(color="black", width=1.5, dash="dash"), name="Significance (α=0.05)"
    ), row=2, col=1)

    # Permutation p-values
    fig.add_trace(go.Bar(
        x=df_for_plot['Variable'], y=df_for_plot['P_Value_Permutation'], name="Permutation p-value",
        marker_color='rgb(214,39,40)', hovertemplate="<b>%{x}</b><br>p-value (Perm.): %{y:.4e}<extra></extra>"
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=df_for_plot['Variable'], y=[0.05] * len(df_for_plot), mode='lines',
        line=dict(color="black", width=1.5, dash="dash"), name="Significance (α=0.05)", showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=750, width=None, # Allow width to be flexible with container
        title_text="<b>HSIC Sensitivity Analysis Dashboard</b>", title_x=0.5,
        title_font_size=20,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white", margin=dict(t=100, l=70, r=30, b=150) # Adjusted margins for tick labels
    )
    fig.update_yaxes(title_text="R²-HSIC Index", row=1, col=1, title_standoff=10)
    fig.update_yaxes(title_text="Raw HSIC Value", type="log", row=1, col=2, title_standoff=10) # Log scale often useful
    fig.update_yaxes(title_text="p-value", type="log", row=2, col=1, title_standoff=10)
    fig.update_yaxes(title_text="p-value", type="log", row=2, col=2, title_standoff=10)

    for r_idx in [1, 2]:
        for c_idx in [1, 2]:
            fig.update_xaxes(tickangle=-45, row=r_idx, col=c_idx, title_standoff=20, automargin=False)

    return fig


def compute_hsic_analysis(model, problem, hsic_size=200, model_code_str=None, language_model='groq'):
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion) analysis without UI components.
    This function calculates HSIC-based sensitivity indices and prepares AI insights.
    (Content improved and LLM prompt enhanced based on OpenTURNS HSIC docs)
    """
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution.")
    if not callable(model): # Added check for model callable
        raise TypeError("Model must be a callable function.")

    dimension = problem.getDimension()
    input_names_from_problem = [problem.getMarginal(i).getDescription()[0] if problem.getMarginal(i).getDescription() and problem.getMarginal(i).getDescription()[0] else f"X{i+1}" for i in range(dimension)]

    # Call the refactored internal computation
    hsic_results_data = compute_hsic_indices(model, problem, N=hsic_size, seed=42)
    hsic_df = create_hsic_dataframe(hsic_results_data) # Uses R2_HSIC_Index

    if hsic_df.empty: # Should not happen if compute_hsic_indices works
        return { # Return a structure indicating no results
            'hsic_results': hsic_results_data, 'hsic_df': hsic_df, 'fig': go.Figure(),
            'top_var': pd.Series(dtype='object'), 'significant_vars': [], 'ai_insights': "HSIC analysis yielded no data.",
            'input_names': input_names_from_problem, 'dimension': dimension
        }

    # Use R2_HSIC_Index for sorting and top variable
    top_var_series = hsic_df.sort_values('R2_HSIC_Index', ascending=False).iloc[0]
    significant_vars_list = hsic_df[hsic_df['P_Value_Asymptotic'] < 0.05]['Variable'].tolist()

    ai_insights_content = None
    if language_model and model_code_str:
        # Use relevant columns for the markdown table
        hsic_md_table = hsic_df[['Variable', 'R2_HSIC_Index', 'P_Value_Asymptotic']].to_markdown(
            index=False, floatfmt=(".s", ".4f", ".4e") # .s for string, .4f, .4e
        )

        inputs_description_parts = []
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = input_names_from_problem[i] # Use names derived at the start
            dist_class_name = marginal.__class__.__name__
            try:
                params = list(marginal.getParameter())
                params_str = ", ".join([f"{p:.3g}" for p in params]) # Format params
                inputs_description_parts.append(f"- **{name}**: `{dist_class_name}` (Parameters: `{params_str}`)")
            except Exception: # Fallback if getParameter fails or params are complex
                inputs_description_parts.append(f"- **{name}**: `{dist_class_name}`")
        inputs_description_md = "\n".join(inputs_description_parts)
        model_code_formatted = f"```python\n{model_code_str.strip()}\n```" if model_code_str else "Not provided."

        prompt = f"""
{RETURN_INSTRUCTION}

An HSIC (Hilbert-Schmidt Independence Criterion) sensitivity analysis was performed.

**Model Definition:**
{model_code_formatted}

**Input Variable Distributions:**
{inputs_description_md}

**HSIC Analysis Results (Sorted by R2-HSIC Index):**
{hsic_md_table}
*R2-HSIC is a normalized index (0 to 1). p-values assess statistical significance of dependence (lower is more significant against the null hypothesis of independence).*

Please provide a comprehensive expert analysis:

### 1. HSIC Methodology Insights
   - Explain what HSIC (and specifically R2-HSIC) measures in sensitivity analysis. Emphasize its capability to detect non-linear dependencies.
   - Discuss the advantages of HSIC over methods like Pearson correlation.
   - How should the asymptotic and permutation p-values be interpreted in practice, particularly concerning sample size?

### 2. Interpretation of Results
   - Identify the most influential variable(s) based on **R2-HSIC indices**. Quantify their relative importance if possible.
   - Which variables demonstrate statistically significant dependence with the output (e.g., p-value < 0.05)? Relate this to their R2-HSIC values.
   - Are there variables with high R2-HSIC but marginal/non-significant p-values, or vice-versa? What could such discrepancies imply (e.g., sample size issues, type of dependence)?
   - What insights do these dependence patterns provide about the underlying model behavior?

### 3. Implications & Recommendations
   - For variables with strong and significant dependence, what are the practical implications for understanding or controlling the system/model output?
   - Which input variables should be prioritized for further investigation, data collection, or uncertainty reduction efforts?
   - Based on these HSIC findings, what types of model behavior (e.g., strong non-linearities, interactions not captured by variance-based methods) might be present?
   - Suggest potential follow-up analyses (e.g., target HSIC, conditional HSIC as described in OpenTURNS documentation, or other methods for a comparative view).

### 4. Limitations of this Specific Analysis
   - Briefly comment on potential limitations related to the current analysis (e.g., chosen sample size N={hsic_size}, use of SquaredExponential kernel, V-statistic estimator).

Focus on providing clear, actionable insights grounded in the provided data and HSIC theory.
"""
        try:
            ai_insights_content = call_groq_api(prompt, model_name=language_model)
        except Exception as e:
            ai_insights_content = f"Error generating AI insights: {str(e)}\n\n{traceback.format_exc()}" # Include traceback in error

    # Create figure using the original hsic_results_data which contains all necessary components
    fig_hsic = create_hsic_plots(hsic_results_data)

    return {
        'hsic_results': hsic_results_data, # The raw dict from compute_hsic_indices
        'hsic_df': hsic_df,
        'fig': fig_hsic,
        'top_var': top_var_series.to_dict(), # Convert series to dict for easier access
        'significant_vars': significant_vars_list,
        'ai_insights': ai_insights_content,
        'input_names': hsic_results_data['input_names'], # Ensure this comes from results_data
        'dimension': dimension
    }

def display_hsic_results(analysis_results, model_code_str=None, language_model='groq'):
    """
    Display HSIC sensitivity analysis results using Streamlit.
    (Content improved, formatting, and significance stars)
    """
    try:
        if not analysis_results or 'hsic_df' not in analysis_results or analysis_results['hsic_df'].empty:
            st.warning("HSIC results are not available or are empty.")
            return

        hsic_df = analysis_results['hsic_df']
        # 'top_var' in analysis_results should be a dictionary after the change in compute_hsic_analysis
        top_var_dict = analysis_results.get('top_var', {})
        significant_vars_list = analysis_results.get('significant_vars', [])
        min_p_var_series = hsic_df.loc[hsic_df['P_Value_Asymptotic'].idxmin()] if not hsic_df.empty else pd.Series(dtype='object')


        st.subheader("📊 HSIC Analysis Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "🥇 Most Influential (R²-HSIC)",
                str(top_var_dict.get('Variable', 'N/A')),
                f"{top_var_dict.get('R2_HSIC_Index', 0.0):.4f}" # Use R2_HSIC_Index key
            )
        with col2:
            st.metric(
                "🔬 Statistically Significant Variables",
                f"{len(significant_vars_list)} of {len(hsic_df)}",
                f"({len(significant_vars_list)/len(hsic_df)*100:.1f}% if N>0 else 0.0%)", # Percentage
                help="Variables with asymptotic p-value < 0.05."
            )
        with col3:
            st.metric(
                "✨ Most Statistically Significant",
                str(min_p_var_series.get('Variable', 'N/A')),
                f"p ≈ {min_p_var_series.get('P_Value_Asymptotic', float('nan')):.2e}",
                help="Variable with the lowest asymptotic p-value."
            )

        st.markdown("""
        **HSIC (Hilbert-Schmidt Independence Criterion)** measures dependence (linear and non-linear) between model inputs and output.
        - **R²-HSIC Index**: Normalized measure (0 to 1) of dependence strength. Higher is stronger. Sum does not necessarily equal 1.
        - **Raw HSIC**: Unnormalized dependence measure.
        - **p-value**: Statistical significance of the dependence. Low p-value (e.g., < 0.05) suggests the observed dependence is unlikely due to chance.
        """)

        st.plotly_chart(analysis_results['fig'], use_container_width=True)

        st.subheader("🔢 Detailed HSIC Indices")
        display_df = hsic_df.copy()
        # Format numeric columns for display
        for col, fmt_str in [('Raw_HSIC', "{:.4e}"), ('R2_HSIC_Index', "{:.4f}"),
                             ('P_Value_Asymptotic', "{:.3e}"), ('P_Value_Permutation', "{:.3e}")]:
            if col in display_df:
                display_df[col] = display_df[col].apply(lambda x: fmt_str.format(x) if pd.notnull(x) else "N/A")

        def get_significance_display(p_value_numeric):
            if pd.isnull(p_value_numeric): return ""
            if p_value_numeric < 0.001: return "***"
            if p_value_numeric < 0.01: return "**"
            if p_value_numeric < 0.05: return "*"
            return "ns" # Not significant

        # Use original numeric p-values from hsic_df for significance calculation
        if 'P_Value_Asymptotic' in hsic_df:
            display_df['Significance (Asymp.)'] = hsic_df['P_Value_Asymptotic'].apply(get_significance_display)
            cols_to_show = ['Variable', 'R2_HSIC_Index', 'Raw_HSIC', 'P_Value_Asymptotic', 'P_Value_Permutation', 'Significance (Asymp.)']
        else:
            cols_to_show = ['Variable', 'R2_HSIC_Index', 'Raw_HSIC']


        st.dataframe(display_df[[c for c in cols_to_show if c in display_df.columns]], use_container_width=True)
        st.caption("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, ns p ≥ 0.05 (based on asymptotic p-value)")

        # Display interpretation (from original structure)
        st.subheader("Interpretation (Summary)")
        st.markdown(f"""
        - **Most influential variable (R²-HSIC)**: `{top_var_dict.get('Variable', 'N/A')}` (Normalized R²-HSIC: `{top_var_dict.get('R2_HSIC_Index', 0.0):.4f}`)
        - **Statistically significant variables (p < 0.05, Asymptotic)**: {len(significant_vars_list)} out of {len(hsic_df)}. Namely: `{', '.join(significant_vars_list) if significant_vars_list else 'None'}`.
        - HSIC captures both linear and non-linear dependencies. Lower p-values indicate stronger evidence against independence.
        """)

        ai_insights_text = analysis_results.get('ai_insights')
        if ai_insights_text:
            st.markdown("### 🤖 AI-Generated Expert Analysis")
            # Display AI insights directly without an inner expander
            st.markdown(ai_insights_text, unsafe_allow_html=True)
        elif model_code_str and language_model: # Original condition
             st.info("AI insights were not generated or an error occurred during generation.")

    except KeyError as ke:
        st.error(f"Error displaying HSIC results: A required key is missing from the analysis results: {str(ke)}")
        st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"An unexpected error occurred in HSIC results display: {str(e)}")
        st.code(traceback.format_exc())


def run_hsic_analysis(size=200, model=None, problem=None, model_code_str=None, language_model='groq', display_results=True):
    """
    Perform HSIC analysis with UI elements, calling compute and display functions.
    (Structure adapted to call the refactored compute_hsic_analysis)
    """
    analysis_output_dict = None # Define outside try for broader scope
    try:
        if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
            st.error("Error: Problem must be an OpenTURNS distribution.")
            return None
        if not callable(model):
            st.error("Error: Model must be a callable function.")
            return None

        hsic_intro_displayed = False
        if display_results:
            # Ensure this container and its content are not inside another expander in the calling page
            # to avoid the nested expander error.
            # Using st.container() for grouping, not st.expander() here unless it's top-level for this analysis.
            with st.container(): # Changed from st.expander to st.container if it's part of a larger page structure
                st.header("🚀 Hilbert-Schmidt Independence Criterion (HSIC) Analysis") # Changed to header
                st.markdown("""
                HSIC measures dependence (linear and non-linear) between inputs and output.
                - **R²-HSIC Index**: Normalized (0-1). Higher = stronger dependence. Not additive like Sobol'.
                - **p-values**: Statistical significance. Low p-value (e.g., < 0.05) suggests dependence is real.
                """)
                hsic_intro_displayed = True
                # Slider for sample size - make its key unique if run_hsic_analysis can be called multiple times
                # or ensure it's only created once per analysis type if it's a global control.
                # For now, assuming it's specific to this call context if display_results is True.
                current_hsic_size = st.slider(
                    "Number of Samples for HSIC",
                    min_value=100, max_value=2000, value=int(size), step=50,
                    key=f"hsic_sample_size_{str(model_code_str)[:20]}" # More unique key
                )
        else:
            current_hsic_size = int(size)

        # Compute HSIC analysis (core logic)
        # Spinner context:
        spinner_context = st.spinner("Computing HSIC indices and generating insights...") if display_results else nullcontext()

        with spinner_context:
            analysis_output_dict = compute_hsic_analysis( # This is your original function name
                model=model,
                problem=problem,
                hsic_size=current_hsic_size,
                model_code_str=model_code_str,
                language_model=language_model
            )

        # Display results if requested and computation was successful
        if display_results:
            if analysis_output_dict and not analysis_output_dict.get('hsic_df', pd.DataFrame()).empty:
                if not hsic_intro_displayed: # If intro wasn't shown (e.g. if above block was skipped)
                    st.subheader("HSIC Analysis Results") # Fallback title
                st.success("HSIC Analysis Complete!")
                display_hsic_results(analysis_output_dict, model_code_str, language_model)
            elif analysis_output_dict and analysis_output_dict.get('hsic_df', pd.DataFrame()).empty:
                st.warning("HSIC analysis completed but yielded no data to display.")
            else: # analysis_output_dict is None or indicates failure
                st.error("HSIC analysis failed to produce results.")

        # Store results in session state (as in your original structure)
        # This part should be outside the "if display_results" if you want to store results even when not displayed
        if analysis_output_dict:
            response_key_main = 'hsic_analysis_results' # Main results
            response_key_insights = 'hsic_analysis_response_markdown' # For AI insights specifically

            st.session_state.setdefault('all_results', {})[response_key_main] = analysis_output_dict
            if 'ai_insights' in analysis_output_dict and analysis_output_dict['ai_insights']:
                st.session_state[response_key_insights] = analysis_output_dict['ai_insights']
            elif response_key_insights in st.session_state: # Clear old insights if new ones are None
                del st.session_state[response_key_insights]


        return analysis_output_dict

    except Exception as e:
        error_message = f"Error in HSIC analysis workflow: {str(e)}"
        if display_results:
            st.error(error_message)
            # Avoid expander here if this function itself could be inside an expander
            st.text_area("Error Details:", traceback.format_exc(), height=200)
        else:
            print(error_message) # Log to console if not displaying in UI
            print(traceback.format_exc())
        return None # Indicate failure

# Example Usage (Illustrative)
if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="HSIC Module Test")
    st.sidebar.title("HSIC Test Controls")

    # Dummy model and problem for testing
    def simple_model_hsic(X_list_of_lists): # ot.PythonFunction expects list of lists for multiple points
        # This example model is simple and expects a single point [x1, x2]
        # but ot.PythonFunction will call it with a list of points if X is a Sample.
        # We need to handle scalar input from the list provided by ot.PythonFunction.
        x1, x2 = X_list_of_lists[0], X_list_of_lists[1] # Correct access for single point evaluation by ot.PythonFunction
        return [x1**2 * np.sin(x2) + x2 * np.cos(x1) + x1*x2] # Must return a list for ot.PythonFunction

    dist_x1_hsic = ot.Normal(0, 1); dist_x1_hsic.setDescription(["InputA_Normal"])
    dist_x2_hsic = ot.Uniform(-3, 3); dist_x2_hsic.setDescription(["InputB_Uniform"])
    problem_dist_hsic = ot.ComposedDistribution([dist_x1_hsic, dist_x2_hsic])
    model_code_str_hsic = """
def model(X): # X is a list [x1, x2]
    x1, x2 = X[0], X[1]
    return [x1**2 * np.sin(x2) + x2 * np.cos(x1) + x1*x2]
"""
    st.sidebar.markdown("### Demo Controls")
    sample_size_demo = st.sidebar.slider("HSIC Sample Size (Demo)", 100, 500, 150, 25, key="hsic_demo_size")
    lang_model_demo = st.sidebar.selectbox("Language Model (Demo)", [None, "groq", "other_model"], index=1, key="hsic_demo_lm")


    if st.sidebar.button("Run HSIC Analysis Demo", key="run_hsic_demo_button"):
        st.session_state.clear() # Clear previous states for clean demo run
        st.info(f"Running HSIC analysis with N={sample_size_demo}...")
        # This is the main entry point function from your original structure
        results = run_hsic_analysis(
            size=sample_size_demo,
            model=simple_model_hsic,
            problem=problem_dist_hsic,
            model_code_str=model_code_str_hsic,
            language_model=lang_model_demo,
            display_results=True
        )
        if results and not results.get('hsic_df', pd.DataFrame()).empty :
            st.sidebar.success("HSIC Demo Completed.")
        else:
            st.sidebar.error("HSIC Demo Failed or produced no results.")
            if results:
                st.sidebar.json(results) # Show what was returned if it failed partially
    else:
        st.info("Click the button in the sidebar to run the HSIC analysis demo.")