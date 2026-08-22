import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
# from plotly.subplots import make_subplots # Not used, can be removed if not planned
from utils.core_utils import call_groq_api # Assuming this is correctly defined elsewhere
from utils.constants import RETURN_INSTRUCTION # Assuming this is correctly defined elsewhere

def compute_correlation_analysis_univariate(model, problem, size=1000):
    """
    Perform correlation analysis calculations for a univariate model output.
    
    Calculates Pearson, Spearman, PCC, PRCC, SRC, and SRRC between
    input variables and the single model output.
    
    Parameters
    ----------
    model : callable
        The model function (e.g., ot.Function or Python callable returning a single output).
    problem : ot.Distribution
        OpenTURNS distribution for input uncertainty.
    size : int, optional
        Number of samples for analysis (default is 1000).
        
    Returns
    -------
    dict
        Contains the correlation DataFrame, input/output names, raw samples, and size.
    """
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution object.")
    
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        desc = marginal.getDescription()
        name = desc[0] if desc and desc[0] else f"X{i+1}"
        input_names.append(name)
    
    sample_X = problem.getSample(size)
    raw_output_values = model(sample_X)

    # Process and validate model output to ensure it's a univariate (N x 1) ot.Sample
    if isinstance(raw_output_values, ot.Sample):
        if raw_output_values.getDimension() != 1:
            raise ValueError(
                f"Model output (ot.Sample) must be univariate. "
                f"Detected dimension: {raw_output_values.getDimension()}"
            )
        sample_Y = raw_output_values
    else:  # Attempt to convert from Python list/numpy array
        try:
            np_Y = np.asarray(raw_output_values, dtype=float) # Ensure numeric
            if np_Y.ndim == 1:  # If it's a 1D array (N,)
                np_Y = np_Y.reshape(-1, 1)  # Reshape to (N,1)
            elif np_Y.ndim == 0:  # scalar for a single sample run (should not happen with size > 1)
                np_Y = np.array([[np_Y]]) # Reshape to (1,1)
            
            if np_Y.shape[0] != size:
                if np_Y.shape[0] == 1 and np_Y.shape[1] == size: # Check for (1, N)
                    np_Y = np_Y.T
                else:
                    raise ValueError(
                        f"Model output sample size ({np_Y.shape[0]}) "
                        f"does not match input sample size ({size})."
                    )
            if np_Y.shape[1] != 1:
                raise ValueError(
                    f"Model output must be univariate (single column/value per input sample). "
                    f"Detected columns: {np_Y.shape[1]}"
                )
            sample_Y = ot.Sample(np_Y)
        except Exception as e:
            raise TypeError(
                f"Failed to process model output into a univariate ot.Sample. "
                f"Original output type: {type(raw_output_values)}. Error: {e}"
            )

    # Determine the single output name
    output_desc = sample_Y.getDescription()
    output_name = output_desc[0] if output_desc and output_desc[0] else "Y"
        
    corr_analysis_engine = ot.CorrelationAnalysis(sample_X, sample_Y)
    
    methods_coeffs = {
        "Pearson": list(corr_analysis_engine.computeLinearCorrelation()),
        "Spearman": list(corr_analysis_engine.computeSpearmanCorrelation()),
        "PCC": list(corr_analysis_engine.computePCC()),
        "PRCC": list(corr_analysis_engine.computePRCC()),
        "SRC": list(corr_analysis_engine.computeSRC()),
        "SRRC": list(corr_analysis_engine.computeSRRC())
    }
    
    corr_df = pd.DataFrame(methods_coeffs, index=input_names)
    # Sort by absolute Pearson correlation for consistent display
    corr_df['abs_Pearson_Sort'] = corr_df['Pearson'].abs()
    corr_df = corr_df.sort_values('abs_Pearson_Sort', ascending=False).drop('abs_Pearson_Sort', axis=1)
    
    return {
        'correlation_df': corr_df,
        'input_names': input_names,
        'output_name': output_name, # Singular output name
        'sample_X': sample_X,
        'sample_Y': sample_Y,
        'size': size
    }

def create_correlation_visualizations_univariate(corr_df, output_name):
    """
    Create Plotly visualizations for correlation analysis results of the single output.
    The 'Correlation Heatmap' is removed. The 'Pearson vs. Spearman' scatter plot
    is prepared for full-width display by the calling Streamlit function.
    A minor adjustment is made to text annotations on the scatter plot.
    """
    # Determine strongest correlations from Pearson
    strongest_pos_val = corr_df['Pearson'].max()
    strongest_pos_var = corr_df['Pearson'].idxmax() if not pd.isna(strongest_pos_val) else "N/A"
    strongest_neg_val = corr_df['Pearson'].min()
    strongest_neg_var = corr_df['Pearson'].idxmin() if not pd.isna(strongest_neg_val) else "N/A"
    
    strongest_abs_val = corr_df['Pearson'].abs().max()
    strongest_abs_var = corr_df['Pearson'].abs().idxmax() if not pd.isna(strongest_abs_val) else "N/A"
    
    strongest_abs_sign = "neutral"
    if strongest_abs_var != "N/A":
        pearson_val_for_abs = corr_df.loc[strongest_abs_var, 'Pearson']
        if pearson_val_for_abs > 1e-6: # Threshold for positive
            strongest_abs_sign = "positive"
        elif pearson_val_for_abs < -1e-6: # Threshold for negative
            strongest_abs_sign = "negative"

    # Combined Bar Chart for all methods
    fig_combined = go.Figure()
    methods = ['Pearson', 'Spearman', 'PCC', 'PRCC', 'SRC', 'SRRC']
    # Using a distinct color sequence
    colors = px.colors.qualitative.Vivid 

    for i, method in enumerate(methods):
        fig_combined.add_trace(go.Bar(
            x=corr_df.index, y=corr_df[method], name=method,
            marker_color=colors[i % len(colors)],
            hovertemplate=f"<b>{method}</b><br>Input: %{{x}}<br>Coefficient: %{{y:.3f}}<extra></extra>"
        ))
    fig_combined.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))
    fig_combined.update_layout(
        title_text=f'Correlation Coefficients for Output: {output_name}',
        xaxis_title='Input Variables', yaxis_title='Correlation Coefficient Value',
        template='plotly_white', height=500, barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80) # Adjusted top margin for title
    )

    # Pearson vs. Spearman Scatter Plot
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(
        x=corr_df['Pearson'], y=corr_df['Spearman'], mode='markers+text',
        marker=dict(size=10, color=colors[0 % len(colors)]), 
        text=corr_df.index, 
        textposition="top right",
        textfont=dict(
            size=9  # Slightly reduced font size for annotations
        ),
        name='Variables', 
        hovertemplate="<b>%{text}</b><br>Pearson: %{x:.3f}<br>Spearman: %{y:.3f}<extra></extra>"
    ))
    max_val = 1.05  # Range for axes [-1.05, 1.05]
    fig_comparison.add_shape(type="line", x0=-max_val, y0=-max_val, x1=max_val, y1=max_val,
                            line=dict(color="grey", dash="dash", width=1))
    fig_comparison.add_hline(y=0, line=dict(color="black", width=0.5, dash="dot"))
    fig_comparison.add_vline(x=0, line=dict(color="black", width=0.5, dash="dot"))
    fig_comparison.update_layout(
        title_text=f'Pearson (Linear) vs. Spearman (Monotonic) for Output: {output_name}',
        xaxis_title='Pearson Coefficient', yaxis_title='Spearman Coefficient',
        template='plotly_white', height=500,  # You can adjust height as needed
        xaxis=dict(range=[-max_val, max_val], zeroline=False, gridcolor='rgba(200,200,200,0.3)'), 
        yaxis=dict(range=[-max_val, max_val], zeroline=False, gridcolor='rgba(200,200,200,0.3)'),
        margin=dict(t=80),
        # ── added to minimise text overlap ──
        uniformtext_minsize=9,
        uniformtext_mode='hide'
    )
    fig_comparison.add_annotation(x=0.5, y=0.97, xref="paper", yref="paper",
                                text="Points on diagonal: Linear relationship strength matches monotonic strength.",
                                showarrow=False, font_size=10)
    fig_comparison.add_annotation(x=0.5, y=0.03, xref="paper", yref="paper",
                                text="Points off diagonal: Indicate non-linear monotonic relationships or outliers.",
                                showarrow=False, font_size=10)

    return {
        'fig_combined_bars': fig_combined, 
        'fig_pearson_spearman_scatter': fig_comparison,
        'strongest_pos_val': strongest_pos_val, 'strongest_pos_var': strongest_pos_var,
        'strongest_neg_val': strongest_neg_val, 'strongest_neg_var': strongest_neg_var,
        'strongest_abs_val': strongest_abs_val, 'strongest_abs_var': strongest_abs_var,
        'strongest_abs_sign': strongest_abs_sign
    }


def display_correlation_results(analysis_results_dict):
    """
    Display correlation analysis results for the single univariate output.
    """
    st.header("📊 Correlation Analysis Results")

    # Extract data from the results dictionary
    corr_df = analysis_results_dict['correlation_df']
    output_name = analysis_results_dict['output_name']
    viz_bundle = analysis_results_dict['visualizations'] # This now holds the dict from create_correlation_visualizations

    # Display summary metrics
    st.subheader(f"Key Correlation Metrics for Output: {output_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Most Correlated (Abs Pearson)", 
        f"{viz_bundle['strongest_abs_var']}",
        f"{viz_bundle['strongest_abs_val']:.3f} ({viz_bundle['strongest_abs_sign']})"
    )
    col2.metric(
        "Strongest Positive Pearson", 
        f"{viz_bundle['strongest_pos_var']}",
        f"{viz_bundle['strongest_pos_val']:.3f}"
    )
    col3.metric(
        "Strongest Negative Pearson", 
        f"{viz_bundle['strongest_neg_var']}",
        f"{viz_bundle['strongest_neg_val']:.3f}"
    )
    
    # Display plots
    st.subheader("Visualizations")
    st.markdown("##### All Correlation Coefficients (Bar Chart)")
    st.markdown("Compares different correlation coefficients for each input variable. Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation). A value near 0 indicates a weak correlation of that type.")
    st.plotly_chart(viz_bundle['fig_combined_bars'], width='stretch')
    
    # Display correlation table
    st.subheader("Correlation Coefficients Table")
    st.dataframe(corr_df.style.format("{:.4f}").background_gradient(cmap='RdBu_r', axis=None, vmin=-1, vmax=1), width='stretch')

    st.subheader("Pearson vs. Spearman Coefficients") # Using subheader
    st.markdown("""
    This scatter plot compares Pearson (linear) and Spearman (monotonic) correlation coefficients for each input variable against a specific output or between pairs.
    
    * **Diagonal Line (y=x)**: Points lying close to this line suggest the relationship is well-captured by a linear model.
    * **Deviations**:
        * Points significantly off the diagonal but where both coefficients are strong indicate a non-linear monotonic relationship.
        * Points where Spearman is strong but Pearson is weak are classic indicators of such non-linear monotonic trends.
    """)
    st.plotly_chart(viz_bundle['fig_pearson_spearman_scatter'], width='stretch')

    # Enhanced Interpretation Section
    st.subheader("Interpreting Correlation Coefficients")
    st.markdown(f"""
    Correlation analysis helps quantify the strength and direction of association between each input variable and the model output ('{output_name}').
    
    - **Pearson Coefficient ($R$ or $\\rho$):** Measures **linear** relationship.
        - Range: -1 to +1.
        - Sensitive to outliers.
        - $\\rho = 0$ implies no linear correlation, but a non-linear relationship might still exist.
    
    - **Spearman Rank Coefficient ($\\rho_S$):** Measures **monotonic** relationship (how well the relationship can be described by a monotonic function).
        - Calculated on the ranks of the data, making it robust to outliers and non-normally distributed data.
        - Range: -1 to +1.
        - $\\rho_S = 1$ means as one variable increases, the other always increases (not necessarily linearly).
        - If $|\\rho_S| > |R|$, this often suggests a monotonic, non-linear relationship.

    - **Partial Correlation Coefficient (PCC):** Measures the **linear** correlation between an input $X_i$ and the output $Y$, after removing the linear effects of all *other* input variables $X_j (j \\neq i)$ from both $X_i$ and $Y$.
        - Useful when inputs are correlated among themselves, as it isolates the direct linear association.
        - Example: If $X_1$ correlates with $Y$ only because $X_1$ correlates with $X_2$, and $X_2$ truly drives $Y$, PCC($X_1, Y | X_2$) would be low.

    - **Partial Rank Correlation Coefficient (PRCC):** Similar to PCC, but calculated on the **ranks** of the data.
        - Measures monotonic relationship between $X_i$ and $Y$, accounting for the monotonic effects of other inputs.
        - Robust to non-linear monotonic relationships and outliers when assessing partial correlations.

    - **Standard Regression Coefficient (SRC):** Derived from a linear regression model $Y = \\beta_0 + \\sum \\beta_i X_i + \\epsilon$.
        - $SRC_i = \\beta_i \\frac{{\sigma_{{X_i}}}}{{\sigma_Y}}$.
        - Represents the change in $Y$ (in standard deviations) for a one standard deviation change in $X_i$, assuming a linear model and *independent inputs*.
        - $SRC_i^2$ can be interpreted as the proportion of $Y$'s variance explained by $X_i$ if the model is linear and inputs are uncorrelated (equivalent to first-order Sobol' index).
        - **Caution:** If inputs are correlated or the model is non-linear, SRCs can be misleading.

    - **Standard Rank Regression Coefficient (SRRC):** SRCs calculated on the **ranks** of the inputs and output.
        - Linearizes monotonic relationships before applying regression.
        - More reliable than SRCs if the underlying relationship is monotonic but non-linear. Still assumes ranked inputs are somewhat linearly related to ranked output.
    
    **General Guidance:**
    - Compare Pearson/PCC with Spearman/PRCC. Large differences suggest non-linearity.
    - If inputs are known to be significantly correlated, PCC/PRCC are generally more reliable than Pearson/Spearman for assessing direct influence.
    - SRC/SRRC are useful for linear/monotonic sensitivity but their sum of squares does not necessarily sum to 1 if inputs are correlated.
    """)

    # Display AI insights if available
    if 'llm_insights' in analysis_results_dict and analysis_results_dict['llm_insights']:
        st.subheader("🤖 AI-Powered Insights & Interpretation")
        st.markdown(analysis_results_dict['llm_insights'])

def correlation_analysis(model, problem, model_code_str=None, size=1000, language_model="groq", display_results=True):
    """
    Main function to perform and display correlation analysis for a univariate model.
    """
    results_placeholder = None
    if display_results:
        results_placeholder = st.empty()
        results_placeholder.info("🚀 Starting Correlation Analysis...")

    try:
        # 1. Compute all correlation analysis results
        computed_data = compute_correlation_analysis_univariate(model, problem, size=size)
        
        # 2. Create visualizations (for the single output)
        viz_bundle = create_correlation_visualizations_univariate(
            computed_data['correlation_df'], 
            computed_data['output_name']
        )
        
        # 3. Consolidate results for display and LLM
        analysis_results_dict = {
            **computed_data, # Includes correlation_df, input_names, output_name, size
            'visualizations': viz_bundle, # Contains all figures and strongest_xx stats
            'llm_insights': None # Initialize
        }
        
        # 4. Generate AI insights if model and code are provided
        if language_model and model_code_str:
            correlation_md_table = computed_data['correlation_df'].to_markdown(index=True, floatfmt=".3f")
            
            dist_info_list = []
            for i, name in enumerate(computed_data['input_names']):
                marginal = problem.getMarginal(i)
                params_list = [f"{p:.2f}" for p in marginal.getParameter()] # Format numbers
                dist_info_list.append(f"- **{name}**: {marginal.getClassName()} (Params: {', '.join(params_list)}, Mean: {marginal.getMean()[0]:.2f}, StdDev: {marginal.getStandardDeviation()[0]:.2f})")
            dist_info_md = "\n".join(dist_info_list)

            model_code_formatted = f"```python\n{model_code_str.strip()}\n```" # Standard markdown code block

            prompt = f"""{RETURN_INSTRUCTION}
You are an expert in statistical analysis and uncertainty quantification. I have performed a correlation analysis for a computational model with a single output. Please provide a comprehensive interpretation.

**Model Under Analysis:**
{model_code_formatted}

**Input Variable Distributions:**
{dist_info_md}

**Correlation Coefficients (Input Variables vs. Output: '{computed_data['output_name']}')**
{correlation_md_table}

**Key Pearson Correlation Findings for Output '{computed_data['output_name']}':**
- Most Correlated Input (Absolute Pearson): {viz_bundle['strongest_abs_var']} (Value: {viz_bundle['strongest_abs_val']:.3f}, Direction: {viz_bundle['strongest_abs_sign']})
- Strongest Positive Pearson Correlation: {viz_bundle['strongest_pos_var']} (Value: {viz_bundle['strongest_pos_val']:.3f})
- Strongest Negative Pearson Correlation: {viz_bundle['strongest_neg_var']} (Value: {viz_bundle['strongest_neg_val']:.3f})

**Analysis Request:**
Please provide a detailed scientific analysis covering the following points:

1.  **Executive Summary:**
    * Briefly summarize the most critical findings. What are the key takeaways for someone who needs a quick understanding?
    * Which input variables appear most influential on the output '{computed_data['output_name']}' based on this correlation study?

2.  **Interpretation of Correlation Coefficients:**
    * Discuss the observed Pearson correlations. What do the strongest positive and negative values imply about the model's behavior with respect to '{computed_data['output_name']}'?
    * Compare Pearson ($R$) with Spearman ($\\rho_S$) coefficients for key variables. What do significant differences suggest about the nature of the relationships (e.g., non-linearity)?
    * Explain the insights gained from Partial Correlation Coefficients (PCC) and Partial Rank Correlation Coefficients (PRCC). How do they refine the understanding obtained from simple Pearson/Spearman, especially if input variables might be inter-correlated?
    * Discuss the Standard Regression Coefficients (SRC) and Standard Rank Regression Coefficients (SRRC). What do they indicate about sensitivity, and what are their limitations or underlying assumptions (e.g., linearity, input independence)?

3.  **Model Behavior Insights:**
    * Based on the full set of coefficients, what can be inferred about the dominant relationships (linear, monotonic, etc.) driving the output '{computed_data['output_name']}'?
    * Are there any variables that show consistently strong or weak correlations across multiple methods? What does this consistency (or lack thereof) imply?
    * Highlight any counter-intuitive findings or relationships that warrant further investigation.

4.  **Recommendations for Further Action:**
    * Which input variables should be prioritized for uncertainty reduction efforts to better control or predict '{computed_data['output_name']}'?
    * Are there specific types of model non-linearity or input interactions suggested by these results that might guide further, more advanced sensitivity analyses (e.g., Sobol indices for variance decomposition)?
    * What practical advice can be given to model users or developers based on this correlation analysis?

Please structure your response clearly. Use markdown for formatting, including headers and bullet points. Refer to specific input variables and their coefficient values where appropriate.
"""
            model_name_for_api = language_model
            if not language_model or language_model.lower() == 'groq':
                model_name_for_api = "llama3-70b-8192" # Example capable model for Groq

            max_retries = 3; retry_count = 0; ai_insights_text = None
            while retry_count < max_retries:
                try:
                    ai_insights_text = call_groq_api(prompt, model_name=model_name_for_api)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        ai_insights_text = f"Error generating insights after {max_retries} attempts: {str(e)}"
                        if display_results: st.error(ai_insights_text)
            analysis_results_dict['llm_insights'] = ai_insights_text
        
        if display_results and results_placeholder:
            results_placeholder.success("✅ Correlation Analysis Completed!")
        
        # 5. Display results if requested
        if display_results:
            display_correlation_results(analysis_results_dict)
            
        # 6. Clean up large raw data from the final returned dictionary
        # (sample_X and sample_Y were already not in analysis_results_dict at this point if not needed)
        # If they were added for some reason, this ensures cleanup.
        analysis_results_dict.pop('sample_X', None)
        analysis_results_dict.pop('sample_Y', None)
            
        return analysis_results_dict

    except Exception as e:
        error_message = f"Error during correlation analysis: {str(e)}"
        if display_results:
            if results_placeholder: results_placeholder.error(error_message)
            else: st.error(error_message)
        else:
            print(error_message) # Log to console if not in Streamlit display mode
        
        # Return a structured error dictionary
        return {
            'correlation_df': pd.DataFrame(), 'input_names': [], 'output_name': "Error", 'size': 0,
            'visualizations': {
                'fig_combined_bars': go.Figure().update_layout(title=error_message),
                'fig_pearson_spearman_scatter': go.Figure().update_layout(title=error_message),
            },
            'llm_insights': f"Could not generate insights due to error: {error_message}"
        }