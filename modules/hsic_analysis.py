import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model, sample_inputs

def compute_hsic_analysis(model, problem, hsic_size=200, model_code_str=None, language_model='groq'):
    """
    Compute HSIC (Hilbert-Schmidt Independence Criterion) analysis without UI components.
    
    This function calculates HSIC-based sensitivity indices to measure the dependence between
    input variables and model outputs, including both raw and normalized indices.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    hsic_size : int, optional
        Number of samples for HSIC analysis, by default 200
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis, by default "groq"
        
    Returns
    -------
    dict
        Dictionary containing HSIC analysis results
    """
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Get input names from distribution
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        input_names.append(marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}")
    
    # Compute HSIC indices
    hsic_results = compute_hsic_indices(model, problem, N=hsic_size, seed=42)
    
    # Create DataFrame from the results
    hsic_df = create_hsic_dataframe(hsic_results)
    
    # Find most influential variables
    top_var = hsic_df.sort_values('Normalized_Index', ascending=False).iloc[0]
    significant_vars = hsic_df[hsic_df['p_value_asymptotic'] < 0.05]
    
    # Create visualizations using Plotly
    fig = create_hsic_plots(hsic_results)
    
    # Generate AI insights if a language model is provided
    ai_insights = None
    if language_model and model_code_str:
        # Prepare the data for the API call
        hsic_md_table = hsic_df.to_markdown(index=False, floatfmt=".4e")
        
        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')]) if model_code_str else ""
        
        # Prepare the inputs description
        input_parameters = []
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = input_names[i]
            dist_type = marginal.__class__.__name__
            params = marginal.getParameter()
            input_parameters.append(f"- **{name}**: {dist_type} distribution with parameters {list(params)}")
        
        inputs_description = '\n'.join(input_parameters)
        
        # Prepare the prompt
        prompt = f"""
        {RETURN_INSTRUCTION}
        
        Given the following user-defined model defined in Python code:
        
        ```python
        {model_code_formatted}
        ```
        
        and the following uncertain input distributions:
        
        {inputs_description}
        
        The results of the HSIC analysis are given in the table below:
        
        {hsic_md_table}
        
        Please provide an expert analysis of the HSIC results:
        
        1. **Methodology Overview**
           - Explain the mathematical basis of the HSIC method in sensitivity analysis
           - Discuss the advantages of HSIC over traditional correlation-based methods
           - Explain how HSIC captures both linear and non-linear dependencies
        
        2. **Results Interpretation**
           - Identify which variables have the strongest dependencies with the output based on HSIC indices
           - Discuss the statistical significance of these dependencies based on p-values
           - Explain what these patterns suggest about the model behavior and input-output relationships
        
        3. **Comparison with Other Methods**
           - Discuss how HSIC results might differ from variance-based or correlation-based methods
           - Explain when HSIC is particularly valuable (e.g., for highly non-linear models)
           - Identify potential limitations of the HSIC approach
        
        4. **Recommendations**
           - Suggest which variables should be prioritized for uncertainty reduction based on HSIC indices
           - Recommend additional analyses that might be valuable given these HSIC patterns
           - Provide guidance on how these results can inform decision-making or model refinement
        
        Please keep your response concise, technical, and focused on actionable insights.
        """
        
        try:
            ai_insights = call_groq_api(prompt, model_name=language_model)
        except Exception as e:
            ai_insights = f"Unable to generate AI insights: {str(e)}"
    
    # Return all results in a dictionary
    return {
        'hsic_results': hsic_results,
        'hsic_df': hsic_df,
        'fig': fig,
        'top_var': top_var,
        'significant_vars': significant_vars,
        'ai_insights': ai_insights,
        'input_names': input_names,
        'dimension': dimension
    }

def display_hsic_results(analysis_results, model_code_str=None, language_model='groq'):
    """
    Display HSIC analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing HSIC analysis results
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis, by default "groq"
    """
    # Results Section
    with st.expander("Results", expanded=True):
        st.subheader("HSIC Analysis")
        st.markdown("""
        HSIC (Hilbert-Schmidt Independence Criterion) analysis is a powerful method for detecting non-linear dependencies 
        between input variables and model outputs. Unlike correlation-based methods, HSIC can capture complex, non-linear 
        relationships without making assumptions about the underlying model structure.
        
        The HSIC method provides several key metrics:
        
        - **Raw HSIC Indices**: Measure the absolute strength of dependence between each input and the output
        - **Normalized HSIC Indices**: Represent the relative importance of each input (values sum to 1)
        - **p-values**: Determine the statistical significance of the dependencies
        
        HSIC can detect both linear and non-linear dependencies and makes no assumptions about the distribution of variables,
        making it particularly valuable for complex models with unknown structures.
        """)
        
        # Extract results from the analysis_results dictionary
        hsic_df = analysis_results['hsic_df']
        top_var = analysis_results['top_var']
        significant_vars = analysis_results['significant_vars']
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Most Influential Variable", 
                top_var['Variable'],
                f"HSIC: {top_var['Normalized_Index']:.4f}"
            )
        with col2:
            st.metric(
                "Significant Variables", 
                f"{len(significant_vars)} of {len(hsic_df)}",
                f"{len(significant_vars)/len(hsic_df)*100:.1f}%"
            )
        with col3:
            # Find variable with lowest p-value
            min_p_var = hsic_df.sort_values('p_value_asymptotic').iloc[0]
            st.metric(
                "Most Significant Variable", 
                min_p_var['Variable'],
                f"p-value: {min_p_var['p_value_asymptotic']:.2e}"
            )
        
        # Create bar chart for normalized HSIC indices
        st.subheader("Normalized HSIC Indices")
        st.markdown("""
        These indices represent the relative importance of each input variable in determining the model output.
        Higher values indicate stronger dependence between the input and output.
        """)
        
        # Sort by normalized index for the chart
        sorted_df = hsic_df.sort_values('Normalized_Index', ascending=False)
        
        # Create the bar chart
        fig_bar = px.bar(
            sorted_df,
            x='Variable',
            y='Normalized_Index',
            title='Normalized HSIC Indices',
            text='Normalized_Index',
            color='Normalized_Index',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig_bar.update_layout(
            template='plotly_white',
            height=500,
            coloraxis_showscale=False,
            yaxis_title='Normalized HSIC Index',
            xaxis_title='Variable'
        )
        
        # Format text
        fig_bar.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside'
        )
        
        # Display the plot
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Create significance plot
        st.subheader("Statistical Significance")
        st.markdown("""
        This plot shows the p-values for each variable. Lower p-values indicate stronger evidence
        against the null hypothesis of independence between the input and output.
        
        - **p-value < 0.01**: Strong evidence of dependence (highly significant)
        - **p-value < 0.05**: Moderate evidence of dependence (significant)
        - **p-value < 0.10**: Weak evidence of dependence (marginally significant)
        - **p-value ≥ 0.10**: Insufficient evidence of dependence (not significant)
        """)
        
        # Sort by p-value for the chart
        sorted_p_df = hsic_df.sort_values('p_value_asymptotic', ascending=True)
        
        # Create the bar chart for p-values
        fig_p = px.bar(
            sorted_p_df,
            x='Variable',
            y='p_value_asymptotic',
            title='p-values (Asymptotic)',
            text='p_value_asymptotic',
            color='p_value_asymptotic',
            color_continuous_scale='Viridis_r'  # Reversed scale so lower p-values are darker
        )
        
        # Add significance thresholds
        fig_p.add_shape(
            type="line",
            x0=-0.5,
            y0=0.01,
            x1=len(hsic_df)-0.5,
            y1=0.01,
            line=dict(color="red", width=2, dash="dash"),
            name="p=0.01"
        )
        
        fig_p.add_shape(
            type="line",
            x0=-0.5,
            y0=0.05,
            x1=len(hsic_df)-0.5,
            y1=0.05,
            line=dict(color="orange", width=2, dash="dash"),
            name="p=0.05"
        )
        
        # Update layout
        fig_p.update_layout(
            template='plotly_white',
            height=500,
            coloraxis_showscale=False,
            yaxis_title='p-value',
            xaxis_title='Variable',
            yaxis_type="log"  # Log scale for better visualization
        )
        
        # Format text
        fig_p.update_traces(
            texttemplate='%{text:.4f}',
            textposition='outside'
        )
        
        # Display the plot
        st.plotly_chart(fig_p, use_container_width=True)
        
        # Display detailed results table
        st.subheader("Detailed Results")
        
        # Format the dataframe for display
        display_df = hsic_df.copy()
        display_df['Raw_HSIC'] = display_df['Raw_HSIC'].apply(lambda x: f"{x:.6f}")
        display_df['Normalized_Index'] = display_df['Normalized_Index'].apply(lambda x: f"{x:.4f}")
        display_df['p_value_asymptotic'] = display_df['p_value_asymptotic'].apply(lambda x: f"{x:.4f}")
        
        # Add significance indicators
        def get_significance(p_value):
            p = float(p_value)
            if p < 0.01:
                return "*** (p<0.01)"
            elif p < 0.05:
                return "** (p<0.05)"
            elif p < 0.10:
                return "* (p<0.10)"
            else:
                return "ns (not significant)"
        
        display_df['Significance'] = display_df['p_value_asymptotic'].apply(get_significance)
        
        # Display the dataframe
        st.dataframe(display_df, use_container_width=True)
        
        # Add interpretation guide
        st.subheader("Interpretation Guide")
        st.markdown("""
        **How to interpret HSIC results:**
        
        1. **Normalized HSIC Indices**:
           - Higher values indicate stronger dependence between the input and output
           - Values sum to 1, representing the relative contribution of each variable
           - Unlike Sobol indices, HSIC indices can capture non-linear dependencies
        
        2. **p-values**:
           - Measure the statistical significance of the dependence
           - Lower p-values provide stronger evidence against independence
           - Variables with p-values < 0.05 are typically considered significant
        
        3. **Comparison with other methods**:
           - HSIC can detect dependencies that correlation analysis might miss
           - Unlike Sobol indices, HSIC doesn't decompose variance but measures general dependence
           - HSIC is particularly valuable for highly non-linear models
        """)
    
    # AI Insights section (only if ai_insights is available)
    if 'ai_insights' in analysis_results and analysis_results['ai_insights']:
        with st.expander("AI Insights", expanded=True):
            st.markdown("### AI-Generated Expert Analysis")
            st.markdown(analysis_results['ai_insights'])

def hsic_analysis(model, problem, model_code_str=None, language_model='groq', display_results=True):
    """
    Perform HSIC (Hilbert-Schmidt Independence Criterion) analysis with enterprise-grade visualizations.
    
    This function calculates HSIC-based sensitivity indices to measure the dependence between
    input variables and model outputs, including both raw and normalized indices.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis, by default "groq"
    display_results : bool, optional
        Whether to display results in the UI (default is True)
        
    Returns
    -------
    dict
        Dictionary containing HSIC analysis results
    """
    try:
        # Add a slider for sample size if displaying results
        if display_results:
            hsic_size = st.slider("Number of HSIC Samples", min_value=100, max_value=1000, value=200, step=100)
        else:
            hsic_size = 200  # Default value when not displaying results
        
        # Compute HSIC analysis
        with st.spinner("Computing HSIC indices...") if display_results else nullcontext():
            analysis_results = compute_hsic_analysis(
                model, problem, hsic_size=hsic_size, 
                model_code_str=model_code_str, language_model=language_model
            )
        
        # Store results in session state for later access
        if 'hsic_analysis_results' not in st.session_state:
            st.session_state.hsic_analysis_results = {}
        
        # Use model's description as a key if available, otherwise use a default key
        model_key = model.getDescription()[0] if hasattr(model, 'getDescription') and model.getDescription()[0] != "" else "default_model"
        st.session_state.hsic_analysis_results[model_key] = analysis_results
        
        # Display results if requested
        if display_results:
            display_hsic_results(analysis_results, model_code_str, language_model)
        
        return analysis_results
    
    except Exception as e:
        if display_results:
            st.error(f"Error in HSIC analysis: {str(e)}")
            st.code(traceback.format_exc())
        raise e

# Define a nullcontext class to use when display_results is False
class nullcontext:
    def __init__(self, enter_result=None):
        self.enter_result = enter_result
    def __enter__(self):
        return self.enter_result
    def __exit__(self, *excinfo):
        pass

def compute_hsic_indices(model, problem, N=200, seed=42):
    """
    Compute HSIC-based sensitivity indices using OpenTURNS built-in tools.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    N : int, optional
        Number of samples, by default 200
    seed : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    dict
        Dictionary containing HSIC results
    """
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")

    # Get input names from distribution
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        input_names.append(marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}")

    # Set random seed
    ot.RandomGenerator.SetSeed(seed)

    # Generate samples
    X = problem.getSample(N)
    
    # Convert model to OpenTURNS function if it's not already
    if not isinstance(model, ot.Function):
        ot_model = ot.PythonFunction(dimension, 1, model)
    else:
        ot_model = model
    
    # Evaluate model
    Y = ot_model(X)
    
    # Create covariance model collection for HSIC
    covarianceModelCollection = []
    
    # Add covariance models for each input variable
    for i in range(dimension):
        Xi = X.getMarginal(i)
        inputCovariance = ot.SquaredExponential(1)
        inputCovariance.setScale(Xi.computeStandardDeviation())
        covarianceModelCollection.append(inputCovariance)
    
    # Add covariance model for output
    outputCovariance = ot.SquaredExponential(1)
    outputCovariance.setScale(Y.computeStandardDeviation())
    covarianceModelCollection.append(outputCovariance)
    
    # Choose an estimator (HSICVStat is biased but asymptotically unbiased)
    estimatorType = ot.HSICVStat()
    
    # Build the HSIC estimator
    hsicEstimator = ot.HSICEstimatorGlobalSensitivity(
        covarianceModelCollection, X, Y, estimatorType
    )
    
    # Get the HSIC indices
    hsic_indices = np.array(hsicEstimator.getHSICIndices())
    
    # Get the R2-HSIC indices (normalized)
    normalized_indices = np.array(hsicEstimator.getR2HSICIndices())
    
    # Get p-values
    p_values_asymptotic = np.array(hsicEstimator.getPValuesAsymptotic())
    p_values_permutation = np.array(hsicEstimator.getPValuesPermutation())
    
    # Create results dictionary
    results = {
        'hsic_indices': hsic_indices,
        'normalized_indices': normalized_indices,
        'p_values_asymptotic': p_values_asymptotic,
        'p_values_permutation': p_values_permutation,
        'input_names': input_names,
        'hsic_estimator': hsicEstimator  # Store the estimator for potential future use
    }

    return results

def create_hsic_dataframe(results):
    """
    Create a DataFrame with HSIC analysis results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing HSIC results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with HSIC results
    """
    df = pd.DataFrame({
        'Variable': results['input_names'],
        'HSIC_Index': results['hsic_indices'],
        'Normalized_Index': results['normalized_indices'],
        'p_value_asymptotic': results['p_values_asymptotic'],
        'p_value_permutation': results['p_values_permutation']
    })
    
    # Sort by normalized index
    df = df.sort_values('Normalized_Index', ascending=False).reset_index(drop=True)
    
    return df

def create_hsic_plots(results):
    """
    Create interactive Plotly visualizations for HSIC results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing HSIC results
        
    Returns
    -------
    go.Figure
        Plotly figure with HSIC visualizations
    """
    # Extract data
    normalized_indices = results['normalized_indices']
    hsic_indices = results['hsic_indices']
    p_values_asymptotic = results['p_values_asymptotic']
    p_values_permutation = results['p_values_permutation']
    input_names = results['input_names']
    
    # Sort by normalized index
    idx = np.argsort(normalized_indices)[::-1]
    normalized_indices = normalized_indices[idx]
    hsic_indices = hsic_indices[idx]
    p_values_asymptotic = p_values_asymptotic[idx]
    p_values_permutation = p_values_permutation[idx]
    input_names = [input_names[i] for i in idx]
    
    # Create a subplot with 3 plots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            "Normalized HSIC Indices", 
            "Raw HSIC Indices", 
            "Asymptotic p-values", 
            "Permutation p-values"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add normalized HSIC indices
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=normalized_indices,
            name="Normalized HSIC",
            marker_color='rgb(55, 83, 109)',
            hovertemplate="<b>%{x}</b><br>Normalized HSIC: %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add raw HSIC indices
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=hsic_indices,
            name="Raw HSIC",
            marker_color='rgb(26, 118, 255)',
            hovertemplate="<b>%{x}</b><br>HSIC: %{y:.6e}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add asymptotic p-values
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=p_values_asymptotic,
            name="Asymptotic p-value",
            marker_color='rgb(219, 64, 82)',
            hovertemplate="<b>%{x}</b><br>p-value: %{y:.4e}<extra></extra>"
        ),
        row=2, col=1
    )
    
    # Add significance threshold line
    fig.add_trace(
        go.Scatter(
            x=input_names,
            y=[0.05] * len(input_names),
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name="Significance (α=0.05)",
            hoverinfo='name'
        ),
        row=2, col=1
    )
    
    # Add permutation p-values
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=p_values_permutation,
            name="Permutation p-value",
            marker_color='rgb(214, 39, 40)',
            hovertemplate="<b>%{x}</b><br>p-value: %{y:.4e}<extra></extra>"
        ),
        row=2, col=2
    )
    
    # Add significance threshold line
    fig.add_trace(
        go.Scatter(
            x=input_names,
            y=[0.05] * len(input_names),
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name="Significance (α=0.05)",
            hoverinfo='name',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        width=1000,
        title_text="HSIC Sensitivity Analysis",
        title_font_size=20,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Normalized HSIC", row=1, col=1)
    fig.update_yaxes(title_text="Raw HSIC", row=1, col=2)
    fig.update_yaxes(title_text="p-value", type="log", row=2, col=1)
    fig.update_yaxes(title_text="p-value", type="log", row=2, col=2)
    
    # Update x-axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(tickangle=45, row=i, col=j)
    
    return fig
