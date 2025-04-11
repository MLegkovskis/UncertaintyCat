import numpy as np
import pandas as pd
import openturns as ot
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def compute_fast_sensitivity_analysis(model, problem, size=400, model_code_str=None, language_model=None):
    """Perform enterprise-grade FAST sensitivity analysis computation.
    
    This function handles all computational aspects of the FAST sensitivity analysis
    without any UI components. It computes sensitivity indices using the Fourier Amplitude 
    Sensitivity Test (FAST) method.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    size : int, optional
        Number of samples for FAST analysis (default is 400)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
        
    Returns
    -------
    dict
        Dictionary containing the results of the FAST analysis
    """
    # Verify input types
    if not isinstance(model, ot.Function):
        raise TypeError("Model must be an OpenTURNS Function")
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise TypeError("Problem must be an OpenTURNS Distribution")
        
    # Get dimension from the model's input dimension
    dimension = model.getInputDimension()
    
    # Get variable names
    variable_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        name = marginal.getDescription()[0]
        variable_names.append(name if name != "" else f"X{i+1}")
    
    # Create independent distribution for FAST (since FAST doesn't work with correlated inputs)
    # Extract marginals from the original problem
    marginals = [problem.getMarginal(i) for i in range(dimension)]
    independent_dist = ot.JointDistribution(marginals)
    
    # Create FAST analysis with independent distribution
    sensitivityAnalysis = ot.FAST(model, independent_dist, size)
    
    # Compute the first order indices
    firstOrderIndices = sensitivityAnalysis.getFirstOrderIndices()
    
    # Retrieve total order indices
    totalOrderIndices = sensitivityAnalysis.getTotalOrderIndices()
    
    # Create DataFrame for indices
    indices_data = []
    for i, name in enumerate(variable_names):
        indices_data.append({
            'Variable': name,
            'First Order': float(firstOrderIndices[i]),
            'Total Order': float(totalOrderIndices[i]),
            'Interaction': float(totalOrderIndices[i]) - float(firstOrderIndices[i]),
            'Interaction %': (float(totalOrderIndices[i]) - float(firstOrderIndices[i])) / float(totalOrderIndices[i]) * 100 if float(totalOrderIndices[i]) > 0 else 0
        })
    
    # Create DataFrame for display
    indices_df = pd.DataFrame(indices_data)
    
    # Sort by total order index for better visualization
    indices_df = indices_df.sort_values('Total Order', ascending=False)
    
    # Create Plotly bar chart for sensitivity indices
    fig_bar = go.Figure()
    
    # Add first order indices
    fig_bar.add_trace(go.Bar(
        x=indices_df['Variable'],
        y=indices_df['First Order'],
        name='First Order Indices',
        marker_color='rgba(31, 119, 180, 0.8)'
    ))
    
    # Add total order indices
    fig_bar.add_trace(go.Bar(
        x=indices_df['Variable'],
        y=indices_df['Total Order'],
        name='Total Order Indices',
        marker_color='rgba(214, 39, 40, 0.8)'
    ))
    
    # Update layout
    fig_bar.update_layout(
        title='FAST Sensitivity Indices',
        xaxis_title='Input Variables',
        yaxis_title='Sensitivity Index',
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # Create pie chart for first order indices
    fig_pie_first = px.pie(
        indices_df, 
        values='First Order', 
        names='Variable',
        title='First Order Indices Distribution',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_pie_first.update_traces(textposition='inside', textinfo='percent+label')
    
    # Create pie chart for total order indices
    fig_pie_total = px.pie(
        indices_df, 
        values='Total Order', 
        names='Variable',
        title='Total Order Indices Distribution',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_pie_total.update_traces(textposition='inside', textinfo='percent+label')
    
    # Create explanation text
    fast_explanation = """
    ## FAST Sensitivity Analysis Results
    
    The Fourier Amplitude Sensitivity Test (FAST) is a global sensitivity analysis method that quantifies how much each input variable contributes to the variance of the model output.
    
    - **First Order Indices**: Measure the direct effect of each input variable on the output variance, without considering interactions.
    - **Total Order Indices**: Measure the total contribution of each input variable to the output variance, including all interactions with other variables.
    - **Interaction**: The difference between Total Order and First Order indices, representing the contribution of interactions.
    
    The sum of First Order indices being close to 1 indicates that the model is primarily additive with minimal interactions between variables.
    """
    
    # Generate LLM insights if a language model is provided
    llm_insights = None
    if language_model and model_code_str:
        # Create a prompt for the language model
        prompt = f"""
        I have performed a FAST sensitivity analysis on the following model:
        
        ```python
        {model_code_str}
        ```
        
        Here are the results:
        First Order Indices: {firstOrderIndices}
        Total Order Indices: {totalOrderIndices}
        
        Variable names: {variable_names}
        
        Please provide insights about these sensitivity analysis results. 
        Explain what they mean for this specific model, which variables are most influential, 
        and whether there are significant interaction effects. 
        Keep your response concise and focused on actionable insights.
        
        {RETURN_INSTRUCTION}
        """
        
        try:
            llm_insights = call_groq_api(prompt, model_name=language_model)
        except Exception as e:
            llm_insights = f"Unable to generate insights: {str(e)}"
    
    # Return results as a dictionary
    return {
        'indices_df': indices_df,
        'fig_bar': fig_bar,
        'fig_pie_first': fig_pie_first,
        'fig_pie_total': fig_pie_total,
        'explanation': fast_explanation,
        'llm_insights': llm_insights
    }

def fast_sensitivity_analysis(model, problem, size=400, model_code_str=None, language_model=None, display_results=True):
    """Perform enterprise-grade FAST sensitivity analysis.
    
    This module provides comprehensive global sensitivity analysis using the Fourier Amplitude 
    Sensitivity Test (FAST) method, which is a relevant alternative to the classical simulation 
    approach for computing sensitivity indices. The FAST method decomposes the model response 
    using Fourier decomposition.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    size : int, optional
        Number of samples for FAST analysis (default is 400)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    display_results : bool, optional
        Whether to display results in the UI (default is True)
        
    Returns
    -------
    dict
        Dictionary containing the results of the FAST analysis
    """
    try:
        # Compute FAST sensitivity analysis results
        fast_results = compute_fast_sensitivity_analysis(
            model, problem, size=size, model_code_str=model_code_str,
            language_model=language_model
        )
        
        # Store results in session state for later access
        if 'fast_results' not in st.session_state:
            st.session_state.fast_results = {}
        
        # Use model's description as a key if available, otherwise use a default key
        model_key = model.getDescription()[0] if model.getDescription()[0] != "" else "default_model"
        st.session_state.fast_results[model_key] = fast_results
        
        # Display results if requested
        if display_results:
            display_fast_results(fast_results, language_model, model_code_str)
            
        return fast_results
    except Exception as e:
        if display_results:
            st.error(f"Error in FAST sensitivity analysis: {str(e)}")
        raise e

def display_fast_results(fast_results, language_model=None, model_code_str=None):
    """
    Display FAST sensitivity analysis results in the Streamlit interface.
    
    Parameters
    ----------
    fast_results : dict
        Dictionary containing the results of the FAST analysis
    language_model : str, optional
        Language model used for analysis, by default None
    model_code_str : str, optional
        String representation of the model code, by default None
    """
    # Results Section
    with st.expander("Results", expanded=True):
        # Display the explanation
        st.markdown(fast_results['explanation'])
        
        # Display the indices table
        st.subheader("Sensitivity Indices")
        
        # Get most influential variable
        most_influential = fast_results['indices_df'].iloc[0]['Variable']
        most_influential_index = fast_results['indices_df'].iloc[0]['Total Order']
        
        # Calculate sums
        sum_first_order = fast_results['indices_df']['First Order'].sum()
        sum_total_order = fast_results['indices_df']['Total Order'].sum()
        
        # Create summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Most Influential Variable", 
                most_influential,
                f"Total Order: {most_influential_index:.4f}"
            )
        with col2:
            st.metric("Sum of First Order Indices", f"{sum_first_order:.4f}")
        with col3:
            st.metric("Sum of Total Order Indices", f"{sum_total_order:.4f}")
        
        # Display the indices table
        st.subheader("Detailed Numerical Results")
        display_df = fast_results['indices_df'][['Variable', 'First Order', 'Total Order', 'Interaction', 'Interaction %']]
        display_df['Interaction %'] = display_df['Interaction %'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Sensitivity Visualizations")
        
        # Display the bar chart
        st.markdown("#### FAST Sensitivity Indices")
        st.markdown("""
        This bar chart compares the First Order and Total Order sensitivity indices for each variable:
        - **First Order Indices**: Measure the direct contribution of each variable to the output variance
        - **Total Order Indices**: Measure the total contribution including interactions with other variables
        """)
        st.plotly_chart(fast_results['fig_bar'], use_container_width=True)
        
        # Display pie charts in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### First Order Indices Distribution")
            st.markdown("""
            This pie chart shows the relative direct contribution of each variable to the output variance.
            """)
            st.plotly_chart(fast_results['fig_pie_first'], use_container_width=True)
        with col2:
            st.markdown("#### Total Order Indices Distribution")
            st.markdown("""
            This pie chart shows the relative total contribution (including interactions) of each variable.
            """)
            st.plotly_chart(fast_results['fig_pie_total'], use_container_width=True)
            
        # Add interpretation based on results
        if sum_first_order < 0.7:
            st.info("""
            **High interaction effects detected.** The sum of first order indices is significantly less than 1, 
            indicating that a substantial portion of the output variance is explained by interactions between variables.
            This suggests that the model behavior cannot be understood by studying each variable separately.
            """)
        elif sum_first_order > 0.9:
            st.success("""
            **Low interaction effects detected.** The sum of first order indices is close to 1, 
            indicating that the model output is primarily determined by the direct effects of individual variables,
            with minimal interaction effects.
            """)
        
    # AI Insights Section
    if fast_results['llm_insights'] and language_model:
        with st.expander("AI Insights", expanded=True):
            # Store the insights in session state for reuse
            if 'fast_analysis_response_markdown' not in st.session_state:
                st.session_state['fast_analysis_response_markdown'] = fast_results['llm_insights']
            
            st.markdown(fast_results['llm_insights'])

def fast_analysis(model, problem, size=400, model_code_str=None, language_model=None, display_results=True):
    """
    Perform and display FAST sensitivity analysis.
    
    This function serves as the main entry point for FAST analysis, handling both
    the calculation and visualization of results.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    size : int, optional
        Number of samples for FAST analysis (default is 400)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    display_results : bool, optional
        Whether to display results in the UI (default is True)
    """
    with st.spinner("Running FAST Sensitivity Analysis..."):
        fast_results = fast_sensitivity_analysis(
            model, problem, size=size, model_code_str=model_code_str,
            language_model=language_model, display_results=display_results
        )
        
        return fast_results
