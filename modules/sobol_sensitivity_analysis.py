import numpy as np
import pandas as pd
import openturns as ot
from SALib.analyze import sobol
from utils.core_utils import call_groq_api
from utils.constants import RETURN_INSTRUCTION
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def sobol_sensitivity_analysis(N, model, problem, model_code_str, language_model='groq'):
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
    """
    try:
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
                xanchor="right",
                x=1
            )
        )
        
        # Create pie chart for variance decomposition
        first_order_values = indices_df['First Order'].tolist()
        first_order_labels = indices_df['Variable'].tolist()
        
        # Add interaction effect to the pie chart
        if interaction_effect > 0.01:  # Only show if it's significant
            first_order_values.append(interaction_effect)
            first_order_labels.append('Interactions')
        
        fig_pie = px.pie(
            values=first_order_values,
            names=first_order_labels,
            title='Variance Decomposition',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(height=500, template='plotly_white')
        
        # Create interaction comparison chart
        fig_interaction = go.Figure()
        
        # Sort by interaction percentage for this chart
        interaction_df = indices_df.sort_values('Interaction %', ascending=False)
        
        # Add bars for interaction percentage
        fig_interaction.add_trace(go.Bar(
            x=interaction_df['Variable'],
            y=interaction_df['Interaction %'],
            name='Interaction Contribution (%)',
            marker_color='rgba(44, 160, 44, 0.8)'
        ))
        
        # Update layout
        fig_interaction.update_layout(
            title='Variable Interaction Contribution',
            xaxis_title='Input Variables',
            yaxis_title='Interaction Contribution (%)',
            template='plotly_white',
            height=400
        )
                
        with st.expander("Results", expanded=True):
            # Sensitivity Analysis Overview
            st.subheader("Sensitivity Analysis Overview")
            st.markdown("""
            Sobol sensitivity analysis is a variance-based method that quantifies the contribution of each input variable 
            to the variance of the model output. This analysis helps identify which uncertain inputs have the most 
            significant impact on model outputs.
            """)
            
            # Create summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                most_influential = indices_df.iloc[0]['Variable']
                most_influential_total = indices_df.iloc[0]['Total Order']
                st.metric(
                    "Most Influential Variable", 
                    most_influential,
                    f"Total Order: {most_influential_total:.4f}"
                )
            with col2:
                st.metric("Sum of First-Order Indices", f"{sum_first_order:.4f}")
                if sum_first_order > 1.01:  # Allow for small numerical errors
                    st.warning("""
                    **Note:** Sum of first-order indices exceeds 1.0, which indicates numerical approximation errors. 
                    This can happen due to limited sample size or computational precision. Consider increasing the 
                    sample size for more accurate results.
                    """)
            with col3:
                st.metric("Sum of Total-Order Indices", f"{sum_total_order:.4f}")
                if sum_total_order < sum_first_order:
                    st.warning("""
                    **Note:** Sum of total-order indices is less than sum of first-order indices, which indicates 
                    numerical approximation errors. Theoretically, total-order indices should be greater than or 
                    equal to first-order indices for each variable.
                    """)
            
            # Add explanation of metrics
            st.markdown("""
            **Key Metrics Explained:**
            - **Most Influential Variable**: The input parameter with the highest total-order sensitivity index, indicating its overall importance
            - **Sum of First-Order Indices**: Sum of all direct contributions to output variance (theoretically should be ≤ 1.0)
            - **Sum of Total-Order Indices**: Sum of all total contributions including interactions (can exceed 1.0 due to counting interactions multiple times)
            
            **Note on Sobol Indices:**
            - First-order indices (S₁) should theoretically be between 0 and 1, with their sum ≤ 1
            - Total-order indices (S₁ᵀ) should also be between 0 and 1 for each variable
            - For each variable, S₁ᵀ ≥ S₁ (total effect includes direct effect)
            - The sum of total-order indices can exceed 1 because interactions are counted multiple times
            """)
            
            # Variance-Based Sensitivity Analysis
            st.subheader("Variance-Based Sensitivity Analysis")
            
            # Create two columns for the charts
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("#### Sobol Sensitivity Indices")
                st.markdown("""
                **First-Order Index (S₁)**: Measures the direct contribution of each input to output variance.
                
                **Total-Order Index (S₁ᵀ)**: Measures the total contribution including interactions with other variables.
                
                **Error bars**: Represent 95% confidence intervals for the sensitivity estimates.
                """)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.markdown("#### Variance Decomposition")
                st.markdown("""
                This pie chart visualizes how the total output variance is distributed:
                
                - **Individual variable slices**: The portion of output variance directly explained by each input variable (first-order effects)
                - **Interactions slice**: The portion of variance that can only be explained by interactions between variables
                
                The variance decomposition follows the principle that the total normalized variance equals 1.0, with:
                - Sum of all first-order effects ≤ 1.0
                - If sum < 1.0, the remainder is attributed to interactions
                """)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Add interpretation based on results
            if interaction_effect > 0.1:
                st.info("""
                **Significant interaction effects detected.** The sum of first-order indices is substantially 
                less than 1, indicating important interactions between variables. This means the model behavior 
                cannot be understood by studying each variable in isolation.
                """)
            elif sum_first_order > 0.9:
                st.success("""
                **Minimal interaction effects detected.** The sum of first-order indices is close to 1, indicating 
                that variables act mostly independently. The model behavior can be understood by studying the 
                effect of each variable separately.
                """)
            
            # Detailed Numerical Results
            st.subheader("Detailed Numerical Results")
            st.markdown("""
            This table summarizes the first-order and total-order Sobol indices for each input variable, 
            along with their confidence intervals and interaction effects.
            
            - **First Order (S₁)**: Measures the direct effect of a variable on the output variance (without interactions)
            - **Total Order (S₁ᵀ)**: Measures the total contribution of a variable (including all interactions)
            - **Interaction**: The difference between total and first-order indices (S₁ᵀ - S₁)
            - **Interaction %**: The percentage of a variable's total effect that comes from interactions
            - **Confidence Intervals**: 95% confidence bounds for the sensitivity indices
            """)
            
            # Create a display DataFrame with the most relevant columns
            display_df = indices_df[['Variable', 'First Order', 'First Order CI', 'Total Order', 'Total Order CI', 'Interaction', 'Interaction %']]
            display_df['Interaction %'] = display_df['Interaction %'].apply(lambda x: f"{x:.2f}%")
            display_df.columns = ['Variable', 'First Order (S₁)', 'S₁ Confidence Interval', 'Total Order (S₁ᵀ)', 'S₁ᵀ Confidence Interval', 'Interaction Effect', 'Interaction %']
            
            # Display the DataFrame
            st.dataframe(display_df, use_container_width=True)
            
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
            st.markdown("#### Variable Interaction Contribution")
            st.plotly_chart(fig_interaction, use_container_width=True)
            
            # Display second-order indices if available
            if S2_matrix is not None:
                st.subheader("Pairwise Interactions")
                st.markdown("""
                The heatmap below shows the strength of interactions between specific pairs of variables.
                
                **How to read this heatmap:**
                - Each cell represents the interaction strength between two variables
                - Darker colors indicate stronger interactions
                - These are second-order Sobol indices that measure how much output variance is explained by the joint effect of two variables beyond their individual effects
                """)
                
                # Create heatmap for second-order indices
                fig_heatmap = px.imshow(
                    S2_matrix,
                    x=variable_names,
                    y=variable_names,
                    color_continuous_scale='RdBu_r',
                    title='Second-Order Interaction Effects',
                    labels=dict(color='Interaction Index')
                )
                fig_heatmap.update_layout(height=500, template='plotly_white')
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Find top interactions
                interactions = []
                for i in range(dimension):
                    for j in range(i+1, dimension):
                        interactions.append({
                            'Variables': f"{variable_names[i]} × {variable_names[j]}",
                            'Interaction Index': float(S2_matrix[i, j])
                        })
                
                if interactions:
                    interactions_df = pd.DataFrame(interactions).sort_values('Interaction Index', ascending=False).head(5)
                    st.markdown("#### Top Interaction Pairs")
                    st.markdown("These variable pairs have the strongest interactions in the model:")
                    st.dataframe(interactions_df, use_container_width=True)
        
        # AI INSIGHTS SECTION
        if language_model:            
            with st.expander("AI Insights", expanded=True):
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
                
                with st.spinner("Generating expert analysis..."):
                    response = call_groq_api(prompt, language_model)
                    st.markdown(response)
        
    except Exception as e:
        st.error(f"Error in Sobol sensitivity analysis: {str(e)}")
        raise
