import numpy as np
import pandas as pd
import openturns as ot
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def fast_sensitivity_analysis(model, problem, size=400, model_code_str=None, language_model=None):
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
        
    Returns
    -------
    dict
        Dictionary containing the results of the FAST analysis
    """
    try:
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
        
        # Create a pie chart for first order indices
        fig_pie_first = px.pie(
            indices_df, 
            values='First Order', 
            names='Variable',
            title='First Order Indices Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie_first.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_first.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # Create a pie chart for total order indices
        fig_pie_total = px.pie(
            indices_df, 
            values='Total Order', 
            names='Variable',
            title='Total Order Indices Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie_total.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_total.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # Create explanatory text
        fast_explanation = """
        ### FAST Sensitivity Analysis
        
        The Fourier Amplitude Sensitivity Test (FAST) is a global sensitivity analysis method that uses 
        the Fourier decomposition of the model response to quantify the correlation between input variables 
        and output variables.
        
        #### Interpreting the Results:
        
        - **First Order Indices**: Measure the direct contribution of each input variable to the output variance.
        - **Total Order Indices**: Measure the contribution of each input variable including all its interactions with other variables.
        - **Interaction**: The difference between Total and First order indices, representing the contribution due to interactions.
        
        The indices range from 0 to 1, where values closer to 1 indicate greater sensitivity of the model to that variable.
        
        > **Note**: FAST analysis requires independent inputs, so the marginals of your distribution are used while ignoring any correlation structure.
        > For models with correlated inputs, consider the ANCOVA analysis which specifically accounts for correlations.
        """
        
        # Generate LLM insights if requested
        llm_insights = None
        if language_model and model_code_str:
            # Prepare the prompt
            prompt = f"""
            I've performed a FAST sensitivity analysis on the following model:
            ```python
            {model_code_str}
            ```
            
            The results show these first order indices:
            {', '.join([f"{name}: {float(firstOrderIndices[i]):.4f}" for i, name in enumerate(variable_names)])}
            
            And these total order indices:
            {', '.join([f"{name}: {float(totalOrderIndices[i]):.4f}" for i, name in enumerate(variable_names)])}
            
            Please provide 2-3 paragraphs of insights about:
            1. Which variables have the most influence on the model output and why
            2. The significance of any interaction effects observed
            3. How these results could inform model simplification or further analysis
            
            {RETURN_INSTRUCTION}
            """
            
            # Call the LLM
            llm_insights = call_groq_api(prompt, model_name=language_model)
        
        # Return all results
        return {
            'indices_df': indices_df,
            'fig_bar': fig_bar,
            'fig_pie_first': fig_pie_first,
            'fig_pie_total': fig_pie_total,
            'explanation': fast_explanation,
            'llm_insights': llm_insights
        }
    except Exception as e:
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
    # Create a two-column layout for the main content and chat interface
    main_col, chat_col = st.columns([2, 1])
    
    with main_col:
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
    
    # CHAT INTERFACE in the right column
    if language_model and fast_results['llm_insights']:
        with chat_col:
            st.markdown("### Ask Questions About This Analysis")
            
            # Display a disclaimer about the prompt
            disclaimer_text = """
            **Note:** The AI assistant has been provided with the model code, sensitivity indices, 
            and the analysis results above. You can ask questions to clarify any aspects of the FAST analysis.
            """
            st.info(disclaimer_text)
            
            # Initialize session state for chat messages if not already done
            if "fast_analysis_chat_messages" not in st.session_state:
                st.session_state.fast_analysis_chat_messages = []
            
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
                for message in st.session_state.fast_analysis_chat_messages:
                    role_style = "background-color: #e1f5fe; border-radius: 10px; padding: 8px; margin: 5px 0;" if message["role"] == "assistant" else "background-color: #f0f0f0; border-radius: 10px; padding: 8px; margin: 5px 0;"
                    role_label = "Assistant:" if message["role"] == "assistant" else "You:"
                    chat_messages_html += f"<div style='{role_style}'><strong>{role_label}</strong><br>{message['content']}</div>"
                
                chat_messages_html += "</div>"
                st.markdown(chat_messages_html, unsafe_allow_html=True)
            
            # Chat input below the scrollable container
            prompt = st.chat_input("Ask a question about the FAST sensitivity analysis...", key="fast_side_chat_input")
            
            # Process user input
            if prompt:
                # Add user message to chat history
                st.session_state.fast_analysis_chat_messages.append({"role": "user", "content": prompt})
                
                # Define context generator function
                def generate_context(prompt):
                    # Get variable names and indices from the results
                    variable_names = fast_results['indices_df']['Variable'].tolist()
                    first_order = fast_results['indices_df']['First Order'].tolist()
                    total_order = fast_results['indices_df']['Total Order'].tolist()
                    
                    indices_summary = ', '.join([f"{name}: First Order={first:.4f}, Total Order={total:.4f}" 
                                               for name, first, total in zip(variable_names, first_order, total_order)])
                    
                    return f"""
                    You are an expert assistant helping users understand FAST sensitivity analysis results. 
                    
                    Here is the model code:
                    ```python
                    {model_code_str if model_code_str else "Model code not available"}
                    ```
                    
                    Here is the sensitivity analysis summary:
                    {indices_summary}
                    
                    Sum of First Order Indices: {sum_first_order:.4f}
                    Sum of Total Order Indices: {sum_total_order:.4f}
                    
                    Here is the explanation that was previously generated:
                    {st.session_state.get('fast_analysis_response_markdown', 'No analysis available yet.')}
                    
                    Answer the user's question based on this information. Be concise but thorough.
                    If you're not sure about something, acknowledge the limitations of your knowledge.
                    Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                    """
                
                # Generate context for the assistant
                context = generate_context(prompt)
                
                # Include previous conversation history
                chat_history = ""
                if len(st.session_state.fast_analysis_chat_messages) > 1:
                    chat_history = "Previous conversation:\n"
                    for i, msg in enumerate(st.session_state.fast_analysis_chat_messages[:-1]):
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
                st.session_state.fast_analysis_chat_messages.append({"role": "assistant", "content": response_text})
                
                # Rerun to display the new message immediately
                st.rerun()

def fast_analysis(model, problem, size=400, model_code_str=None, language_model=None):
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
    """
    with st.spinner("Running FAST Sensitivity Analysis..."):
        fast_results = fast_sensitivity_analysis(
            model, problem, size=size, model_code_str=model_code_str,
            language_model=language_model
        )
        
        display_fast_results(fast_results, language_model, model_code_str)
