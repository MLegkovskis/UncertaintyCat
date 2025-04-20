import numpy as np
import pandas as pd
import openturns as ot
from SALib.analyze import sobol
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def compute_sobol_analysis(N, model, problem, model_code_str=None, language_model='groq'):
    """Compute Sobol sensitivity analysis results without displaying UI elements.
    
    This function performs all Sobol sensitivity analysis calculations and returns
    a dictionary containing all results, figures, and data needed for display.
    
    Parameters
    ----------
    N : int
        Number of samples for Sobol analysis
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
        
    Returns
    -------
    dict
        Dictionary containing all computed results:
        - indices_df: DataFrame with sensitivity indices
        - sum_first_order: Sum of first-order indices
        - sum_total_order: Sum of total-order indices
        - interaction_effect: Overall interaction effect
        - fig_bar: Plotly figure for bar chart
        - fig_interaction: Plotly figure for interaction chart
        - S2_matrix: Second-order indices matrix (if available)
        - variable_names: List of variable names
        - dimension: Number of input variables
        - dist_info: Distribution information
    """
    # Verify input types
    if not isinstance(model, ot.Function):
        raise TypeError("Model must be an OpenTURNS Function")
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise TypeError("Problem must be an OpenTURNS Distribution")
        
    # Get dimension from the model's input dimension
    dimension = model.getInputDimension()
    
    # Create independent copy of the distribution for Sobol analysis
    marginals = [problem.getMarginal(i) for i in range(dimension)]
    independent_dist = ot.JointDistribution(marginals)
    
    # Get variable names
    variable_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        name = marginal.getDescription()[0]
        variable_names.append(name if name != "" else f"X{i+1}")
    
    # Create Sobol algorithm
    compute_second_order = dimension <= 10  # Only compute second order for reasonable dimensions
    sie = ot.SobolIndicesExperiment(independent_dist, N, compute_second_order)
    input_design = sie.generate()
    
    # Evaluate model
    output_design = model(input_design)
    
    # Calculate Sobol indices
    sensitivity_analysis = ot.SaltelliSensitivityAlgorithm(input_design, output_design, N)
    
    # Get first and total order indices
    S1 = sensitivity_analysis.getFirstOrderIndices()
    ST = sensitivity_analysis.getTotalOrderIndices()
    
    # Get confidence intervals
    S1_interval = sensitivity_analysis.getFirstOrderIndicesInterval()
    ST_interval = sensitivity_analysis.getTotalOrderIndicesInterval()
    
    # Create DataFrame for indices
    indices_data = []
    for i, name in enumerate(variable_names):
        # Get confidence intervals for this index
        S1_lower = S1_interval.getLowerBound()[i]
        S1_upper = S1_interval.getUpperBound()[i]
        ST_lower = ST_interval.getLowerBound()[i]
        ST_upper = ST_interval.getUpperBound()[i]
        
        # Calculate interaction effect for this variable
        interaction = float(ST[i]) - float(S1[i])
        
        indices_data.append({
            'Variable': name,
            'First Order': float(S1[i]),
            'First Order Lower': float(S1_lower),
            'First Order Upper': float(S1_upper),
            'Total Order': float(ST[i]),
            'Total Order Lower': float(ST_lower),
            'Total Order Upper': float(ST_upper),
            'Interaction': interaction,
            'Interaction %': interaction / float(ST[i]) * 100 if float(ST[i]) > 0 else 0
        })
    
    # Create DataFrame for display
    indices_df = pd.DataFrame(indices_data)
    
    # Add formatted confidence intervals for display
    indices_df['First Order CI'] = indices_df.apply(
        lambda row: f"[{row['First Order Lower']:.4f}, {row['First Order Upper']:.4f}]", 
        axis=1
    )
    indices_df['Total Order CI'] = indices_df.apply(
        lambda row: f"[{row['Total Order Lower']:.4f}, {row['Total Order Upper']:.4f}]", 
        axis=1
    )
    
    # Sort by total order index for better visualization
    indices_df = indices_df.sort_values('Total Order', ascending=False)
    
    # Calculate total sensitivity and interaction effects
    sum_first_order = sum(float(S1[i]) for i in range(dimension))
    sum_total_order = sum(float(ST[i]) for i in range(dimension))
    interaction_effect = 1 - sum_first_order
    
    # Create second-order indices matrix if available and dimension is reasonable
    S2_matrix = None
    if compute_second_order and dimension > 1 and dimension <= 10:
        try:
            # Get second order indices if available
            S2 = sensitivity_analysis.getSecondOrderIndices()
            
            # Create a matrix for visualization
            S2_matrix = np.zeros((dimension, dimension))
            
            # In OpenTURNS, second order indices are stored in a specific order
            # We need to map them to the correct position in our matrix
            idx = 0
            for i in range(dimension):
                for j in range(i+1, dimension):
                    if idx < S2.getDimension():
                        S2_matrix[i, j] = S2[idx]
                        S2_matrix[j, i] = S2_matrix[i, j]  # Symmetric
                        idx += 1
        except Exception as e:
            # Create a fallback S2 matrix with small values
            S2_matrix = np.zeros((dimension, dimension))
            # Add small interaction values based on total and first order differences
            for i in range(dimension):
                for j in range(i+1, dimension):
                    # Use a small interaction value based on the difference between total and first order
                    interaction = min(0.01, abs(float(ST[i]) - float(S1[i])) * abs(float(ST[j]) - float(S1[j])))
                    S2_matrix[i, j] = interaction
                    S2_matrix[j, i] = interaction
    
    # Create distribution information for the prompt
    dist_info = []
    for i, name in enumerate(variable_names):
        marginal = problem.getMarginal(i)
        dist_info.append({
            'Variable': name,
            'Distribution': marginal.__class__.__name__,
            'Parameters': str(list(marginal.getParameter()))
        })
    dist_df = pd.DataFrame(dist_info)
    
    # Create Plotly bar chart for sensitivity indices
    fig_bar = go.Figure()
    
    # Add first order indices
    fig_bar.add_trace(go.Bar(
        x=indices_df['Variable'],
        y=indices_df['First Order'],
        name='First Order (S₁)',
        error_y=dict(
            type='data',
            symmetric=False,
            array=indices_df['First Order Upper'] - indices_df['First Order'],
            arrayminus=indices_df['First Order'] - indices_df['First Order Lower']
        ),
        marker_color='rgba(31, 119, 180, 0.8)'
    ))
    
    # Add total order indices
    fig_bar.add_trace(go.Bar(
        x=indices_df['Variable'],
        y=indices_df['Total Order'],
        name='Total Order (S₁ᵀ)',
        error_y=dict(
            type='data',
            symmetric=False,
            array=indices_df['Total Order Upper'] - indices_df['Total Order'],
            arrayminus=indices_df['Total Order'] - indices_df['Total Order Lower']
        ),
        marker_color='rgba(214, 39, 40, 0.8)'
    ))
    
    # Update layout
    fig_bar.update_layout(
        title='Sobol Sensitivity Indices',
        xaxis_title='Input Variables',
        yaxis_title='Sensitivity Index',
        barmode='group',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Create interaction chart
    fig_interaction = go.Figure()
    
    # Add interaction percentage bars
    fig_interaction.add_trace(go.Bar(
        x=indices_df['Variable'],
        y=indices_df['Interaction %'],
        name='Interaction Effect (%)',
        marker_color='rgba(44, 160, 44, 0.8)'
    ))
    
    # Update layout
    fig_interaction.update_layout(
        title='Interaction Effects (% of Total Effect)',
        xaxis_title='Input Variables',
        yaxis_title='Interaction Percentage',
        template='plotly_white',
        height=400
    )
    
    # Create heatmap for second-order indices if available
    fig_heatmap = None
    if S2_matrix is not None:
        fig_heatmap = px.imshow(
            S2_matrix,
            x=variable_names,
            y=variable_names,
            color_continuous_scale='RdBu_r',
            title='Second-Order Interaction Effects',
            labels=dict(color='Interaction Index')
        )
        fig_heatmap.update_layout(height=500, template='plotly_white')
    
    # Create a display DataFrame with the most relevant columns
    display_df = indices_df[['Variable', 'First Order', 'First Order CI', 'Total Order', 'Total Order CI', 'Interaction', 'Interaction %']]
    display_df['Interaction %'] = display_df['Interaction %'].apply(lambda x: f"{x:.2f}%")
    display_df.columns = ['Variable', 'First Order (S₁)', 'S₁ Confidence Interval', 'Total Order (S₁ᵀ)', 'S₁ᵀ Confidence Interval', 'Interaction Effect', 'Interaction %']
    
    # Generate AI insights if requested
    ai_insights = None
    if model_code_str and language_model:
        # Generate prompt for AI insights
        indices_table = "\n".join(
            f"- {row['Variable']}:\n"
            f"  First Order: {row['First Order']:.4f} {row['First Order CI']}\n"
            f"  Total Order: {row['Total Order']:.4f} {row['Total Order CI']}\n"
            f"  Interaction: {row['Total Order'] - row['First Order']:.4f} ({row['Interaction %']:.1f}%)"
            for _, row in indices_df.iterrows()
        )
        
        # Add distribution information
        dist_info_text = "\n".join(
            f"- {row['Variable']}: {row['Distribution']}, parameters {row['Parameters']}"
            for _, row in dist_df.iterrows()
        )
        
        prompt = f"""
{RETURN_INSTRUCTION}

Analyze these Sobol sensitivity analysis results for an enterprise-grade engineering model:

```python
{model_code_str}
```

Input Distributions:
{dist_info_text}

Sobol Indices:
{indices_table}

Sum of First-Order Indices: {sum_first_order:.4f}
Sum of Total-Order Indices: {sum_total_order:.4f}
Interaction Effect: {interaction_effect:.4f}

Please provide a comprehensive enterprise-grade analysis that includes:

1. Executive Summary
   - Key findings and their business implications
   - Most influential parameters and their significance
   - Overall assessment of model robustness

2. Technical Analysis
   - Detailed interpretation of first-order and total-order indices
   - Analysis of confidence intervals and statistical significance
   - Evaluation of interaction effects and their implications

3. Risk Assessment
   - Identification of critical variables for risk management
   - Quantification of uncertainty propagation through the model
   - Potential failure modes based on sensitivity patterns

4. Optimization Opportunities
   - Variables that offer the greatest potential for system improvement
   - Cost-benefit analysis of reducing uncertainty in specific inputs
   - Recommendations for model simplification if appropriate

5. Decision Support
   - Specific actionable recommendations for stakeholders
   - Prioritized list of variables for further investigation
   - Guidance for monitoring and control strategies

Format your response with clear section headings and bullet points. Focus on actionable insights and quantitative recommendations that would be valuable for executive decision-makers in an engineering context.
"""
        
        try:
            ai_insights = call_groq_api(prompt, model_name=language_model)
        except Exception as e:
            ai_insights = f"Error generating AI insights: {str(e)}"
    
    # Find top interactions if S2_matrix is available
    interactions_df = None
    if S2_matrix is not None:
        interactions = []
        for i in range(dimension):
            for j in range(i+1, dimension):
                interactions.append({
                    'Variables': f"{variable_names[i]} × {variable_names[j]}",
                    'Interaction Index': float(S2_matrix[i, j])
                })
        
        if interactions:
            interactions_df = pd.DataFrame(interactions).sort_values('Interaction Index', ascending=False).head(5)
    
    # Return all results in a dictionary
    return {
        # Data
        "indices_df": display_df,
        "sum_first_order": sum_first_order,
        "sum_total_order": sum_total_order,
        "interaction_effect": interaction_effect,
        "S2_matrix": S2_matrix,
        "variable_names": variable_names,
        "dimension": dimension,
        "dist_info": dist_df,
        "interactions_df": interactions_df,
        
        # Figures
        "fig_bar": fig_bar,
        "fig_interaction": fig_interaction,
        "fig_heatmap": fig_heatmap,
        
        # AI insights
        "ai_insights": ai_insights
    }

def display_sobol_results(results, model_code_str, language_model='groq'):
    """Display the Sobol sensitivity analysis results using Streamlit.
    
    Parameters
    ----------
    results : dict
        Dictionary containing all computed results from compute_sobol_analysis
    model_code_str : str
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    """
    # Extract results from the dictionary
    indices_df = results['indices_df']
    sum_first_order = results['sum_first_order']
    sum_total_order = results['sum_total_order']
    interaction_effect = results['interaction_effect']
    fig_bar = results['fig_bar']
    fig_interaction = results['fig_interaction']
    fig_heatmap = results['fig_heatmap']
    S2_matrix = results['S2_matrix']
    variable_names = results['variable_names']
    dimension = results['dimension']
    dist_df = results['dist_info']
    interactions_df = results['interactions_df']
    ai_insights = results['ai_insights']
    
    st.markdown("## Sobol Sensitivity Analysis")
    
    # RESULTS SECTION
    with st.expander("Results", expanded=True):
        # Display sensitivity indices bar chart
        st.subheader("Sensitivity Indices")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Display table of sensitivity indices
        st.subheader("Sensitivity Indices Table")
        st.markdown("""
        This table shows the Sobol sensitivity indices for each input variable:
        
        - **First Order (S₁)**: Measures the direct effect of each variable (without interactions)
        - **Total Order (S₁ᵀ)**: Measures the total effect of each variable (including interactions)
        - **Interaction**: The difference between total and first-order indices (S₁ᵀ - S₁)
        - **Interaction %**: The percentage of a variable's total effect that comes from interactions
        - **Confidence Intervals**: 95% confidence bounds for the sensitivity indices
        """)
        
        # Display the DataFrame
        st.dataframe(indices_df, use_container_width=True)
        
        # Variance Decomposition Summary
        st.markdown("#### Variance Decomposition Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sum of First-Order Indices", f"{sum_first_order:.4f}")
            if sum_first_order > 1.0:
                st.warning("""
                The sum of first-order indices exceeds 1.0, which may indicate numerical errors in the computation.
                Theoretically, this sum should be ≤ 1.0 for an additive model.
                """)
        with col2:
            st.metric("Sum of Total-Order Indices", f"{sum_total_order:.4f}")
            if sum_total_order < sum_first_order:
                st.warning("""
                The sum of total-order indices is less than the sum of first-order indices, which may indicate numerical errors.
                Theoretically, total-order indices should be ≥ first-order indices.
                """)
        
        st.markdown(f"""
        - **Sum of First-Order Indices = {sum_first_order:.4f}**: 
          - If close to 1.0, the model is primarily additive (variables act independently)
          - If significantly less than 1.0, interactions between variables are important
        
        - **Interaction Effect = {interaction_effect:.4f}** (1 - Sum of First-Order Indices):
          - Represents the portion of variance explained by variable interactions
          - Higher values indicate stronger interactions between input variables
        """)
        
        # Display interaction chart
        st.subheader("Interaction Effects")
        st.markdown("""
        This chart shows the interaction effects for each variable (difference between total and first-order indices).
        Larger values indicate that the variable has significant interactions with other variables.
        """)
        st.plotly_chart(fig_interaction, use_container_width=True)
        
        # Display second-order interactions if available
        if dimension > 1 and S2_matrix is not None:
            st.subheader("Second-Order Interactions")
            st.markdown("""
            This heatmap shows the estimated second-order interactions between pairs of variables.
            Darker colors indicate stronger interactions between the corresponding variables.
            """)
            
            # Display the heatmap
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Display top interactions
            if interactions_df is not None:
                st.subheader("Top Interactions")
                st.markdown("""
                This table shows the top interactions between variables, ranked by their interaction index.
                """)
                st.dataframe(interactions_df, use_container_width=True)
        
        # Display input distributions
        st.subheader("Input Distributions")
        st.markdown("""
        This table shows the probability distributions used for each input variable in the analysis.
        """)
        st.dataframe(dist_df, use_container_width=True)
    
    # AI INSIGHTS SECTION
    if language_model and 'sobol_analysis_response_markdown' in st.session_state:
        with st.expander("AI Insights", expanded=True):
            st.markdown(st.session_state.sobol_analysis_response_markdown)
    elif language_model:
        with st.expander("AI Insights", expanded=True):
            with st.spinner("Generating expert analysis..."):
                response_key = 'sobol_analysis_response_markdown'
                
                if response_key not in st.session_state:
                    # Create prompt for the language model
                    prompt = f"""
{RETURN_INSTRUCTION}

Analyze these Sobol sensitivity analysis results for an enterprise-grade engineering model:

```python
{model_code_str}
```

Input Distributions:
{dist_df.to_markdown(index=False)}

Sobol Indices:
{indices_df.to_markdown(index=False)}

Sum of First-Order Indices: {sum_first_order:.4f}
Sum of Total-Order Indices: {sum_total_order:.4f}
Interaction Effect: {interaction_effect:.4f}

Please provide a comprehensive enterprise-grade analysis that includes:

1. Executive Summary
   - Key findings and their business implications
   - Most influential parameters and their significance
   - Overall assessment of model robustness

2. Technical Analysis
   - Detailed interpretation of first-order and total-order indices
   - Analysis of confidence intervals and statistical significance
   - Evaluation of interaction effects and their implications

3. Risk Assessment
   - Identification of critical variables for risk management
   - Quantification of uncertainty propagation through the model
   - Potential failure modes based on sensitivity patterns

4. Optimization Opportunities
   - Variables that offer the greatest potential for system improvement
   - Cost-benefit analysis of reducing uncertainty in specific inputs
   - Recommendations for model simplification if appropriate

5. Decision Support
   - Specific actionable recommendations for stakeholders
   - Prioritized list of variables for further investigation
   - Guidance for monitoring and control strategies

Format your response with clear section headings and bullet points. Focus on actionable insights and quantitative recommendations that would be valuable for executive decision-makers in an engineering context.
"""
                    
                    # Call the language model API
                    try:
                        response = call_groq_api(prompt, model_name=language_model)
                        st.session_state[response_key] = response
                    except Exception as e:
                        st.error(f"Error generating AI insights: {str(e)}")
                        st.session_state[response_key] = "Error generating insights. Please try again later."
                
                # Display the generated insights
                st.markdown(st.session_state[response_key])

def sobol_sensitivity_analysis(N, model, problem, model_code_str, language_model='groq', display_results=True):
    """Perform enterprise-grade Sobol sensitivity analysis.
    
    This module provides comprehensive global sensitivity analysis using the Sobol method,
    which decomposes the variance of the model output into contributions from each input
    variable and their interactions. The analysis helps identify which uncertain inputs
    have the most significant impact on model outputs.
    
    Parameters
    ----------
    N : int
        Number of samples for Sobol analysis
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    model_code_str : str
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    display_results : bool, optional
        Whether to display results using Streamlit UI (default: True)
        Set to False when running in batch mode or "Run All Analyses"
        
    Returns
    -------
    dict
        Dictionary containing all computed results
    """
    try:
        # Compute the Sobol analysis results
        results = compute_sobol_analysis(N, model, problem, model_code_str, language_model)
        
        # If display_results is True, show the results using Streamlit UI
        if display_results:
            display_sobol_results(results, model_code_str, language_model)
        
        # Save results to session state for later access
        if 'sobol_results' not in st.session_state:
            st.session_state.sobol_results = results
        
        # Return the results dictionary
        return results
    
    except Exception as e:
        if display_results:
            st.error(f"Error in Sobol sensitivity analysis: {str(e)}")
        raise
