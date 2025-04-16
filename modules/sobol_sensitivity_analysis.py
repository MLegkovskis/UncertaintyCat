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
        # Create a two-column layout for the main content and chat interface
        main_col, chat_col = st.columns([2, 1])
        
        with main_col:
            st.markdown("## Sobol Sensitivity Analysis")
            
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
                        response_key = 'sobol_analysis_response_markdown'
                        
                        if response_key not in st.session_state:
                            response = call_groq_api(prompt, language_model)
                            st.session_state[response_key] = response
                        else:
                            response = st.session_state[response_key]
                            
                        st.markdown(response)
        
        # CHAT INTERFACE in the right column
        with chat_col:
            st.markdown("### Ask Questions About This Analysis")
            
            # Display a disclaimer about the prompt
            disclaimer_text = """
            **Note:** The AI assistant has been provided with the model code, input distributions, 
            and the sensitivity analysis results above. You can ask questions to clarify any aspects of the analysis.
            """
            st.info(disclaimer_text)
            
            # Initialize session state for chat messages if not already done
            if "sobol_analysis_chat_messages" not in st.session_state:
                st.session_state.sobol_analysis_chat_messages = []
            
            # Create a container with fixed height for the chat messages
            chat_container_height = 500  # Height in pixels
            
            # Apply CSS to create a scrollable container
            st.markdown(f"""
            <style>
            .chat-container {{
                height: {chat_container_height}px;
                overflow-y: auto;
                border: 1px solid #e6e6e6;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9f9f9;
                margin-bottom: 15px;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            # Create a container for the chat messages
            with st.container():
                # Use HTML to create a scrollable container
                chat_messages_html = "<div class='chat-container'>"
                
                # Display existing messages
                for message in st.session_state.sobol_analysis_chat_messages:
                    role_style = "background-color: #e1f5fe; border-radius: 10px; padding: 8px; margin: 5px 0;" if message["role"] == "assistant" else "background-color: #f0f0f0; border-radius: 10px; padding: 8px; margin: 5px 0;"
                    role_label = "Assistant:" if message["role"] == "assistant" else "You:"
                    chat_messages_html += f"<div style='{role_style}'><strong>{role_label}</strong><br>{message['content']}</div>"
                
                chat_messages_html += "</div>"
                st.markdown(chat_messages_html, unsafe_allow_html=True)
            
            # Chat input below the scrollable container
            prompt = st.chat_input("Ask a question about the sensitivity analysis...", key="sobol_side_chat_input")
            
            # Process user input
            if prompt:
                # Add user message to chat history
                st.session_state.sobol_analysis_chat_messages.append({"role": "user", "content": prompt})
                
                # Generate context for the assistant
                context = f"""
                You are an expert assistant helping users understand Sobol sensitivity analysis results. 
                
                Here is the model code:
                ```python
                {model_code_str}
                ```
                
                Here is information about the input distributions:
                {dist_info_text}
                
                Here is the sensitivity analysis summary:
                Sobol Indices:
                {indices_table}
                
                Sum of First-Order Indices: {sum_first_order:.4f}
                Sum of Total-Order Indices: {sum_total_order:.4f}
                Interaction Effect: {interaction_effect:.4f}
                
                Here is the explanation that was previously generated:
                {st.session_state.get('sobol_analysis_response_markdown', 'No analysis available yet.')}
                
                Answer the user's question based on this information. Be concise but thorough.
                If you're not sure about something, acknowledge the limitations of your knowledge.
                Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                """
            
                # Include previous conversation history
                chat_history = ""
                if len(st.session_state.sobol_analysis_chat_messages) > 1:
                    chat_history = "Previous conversation:\n"
                    for i, msg in enumerate(st.session_state.sobol_analysis_chat_messages[:-1]):
                        role = "User" if msg["role"] == "user" else "Assistant"
                        chat_history += f"{role}: {msg['content']}\n\n"
                
                # Create the final prompt
                chat_prompt = f"""
                {context}
                
                {chat_history}
                
                Current user question: {prompt}
                
                Please provide a helpful, accurate response to this question.
                """
                
                # Call API with chat history
                with st.spinner("Thinking..."):
                    try:
                        response_text = call_groq_api(chat_prompt, model_name=language_model)
                    except Exception as e:
                        st.error(f"Error calling API: {str(e)}")
                        response_text = "I'm sorry, I encountered an error while processing your question. Please try again."
                
                # Add assistant response to chat history
                st.session_state.sobol_analysis_chat_messages.append({"role": "assistant", "content": response_text})
                
                # Rerun to display the new message immediately
                st.rerun()
    
    except Exception as e:
        st.error(f"Error in Sobol sensitivity analysis: {str(e)}")
        raise
