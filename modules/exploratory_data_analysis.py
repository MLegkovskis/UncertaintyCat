# modules/exploratory_data_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.subplots as sp
import scipy.stats as stats
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
import uuid
import json
import os
from groq import Groq

def exploratory_data_analysis(data, N, model, problem, model_code_str, language_model='groq'):
    """
    Perform enterprise-grade exploratory data analysis on Monte Carlo simulation results.
    
    This module provides comprehensive visualization and statistical analysis of both
    input distributions and output results, with a focus on comparing theoretical
    distributions with empirical data.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing Monte Carlo simulation results
    N : int
        Number of Monte Carlo samples
    model : callable
        The model function
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str
        String representation of the model code
    language_model : str, optional
        Language model to use for AI insights, by default 'groq'
    """
    
    # Check if data is available
    if data is None or data.empty:
        st.error("No data available for analysis.")
        return
    
    # Create a two-column layout for the main content and chat interface
    main_col, chat_col = st.columns([2, 1])
    
    with main_col:
        st.markdown("## Exploratory Data Analysis")
        
        # Identify input and output columns and rename for display
        input_columns = [col for col in data.columns if col not in ['Y'] and not col.startswith('Y')]
        output_columns = [col for col in data.columns if col == 'Y' or col.startswith('Y')]
        
        # Create display names dictionary (for UI only, doesn't change dataframe)
        display_names = {}
        for col in output_columns:
            if col == 'Y':
                display_names[col] = 'Output'
            else:
                # For columns like Y1, Y2, etc.
                output_num = col[1:]  # Extract the number after 'Y'
                display_names[col] = f'Output {output_num}'
        
        if not output_columns:
            st.error("No output columns (Y or Y1, Y2, etc.) found in the data.")
            return
        
        # RESULTS SECTION
        with st.expander("Results", expanded=True):
            # Create correlation matrix
            st.subheader("Correlation Analysis")
            
            # Create correlation matrix
            corr = data.corr()
            
            # Create a copy for display with renamed columns
            display_corr = corr.copy()
            display_corr.rename(columns=display_names, index=display_names, inplace=True)
            
            # Create a heatmap for the correlation matrix
            fig_corr = px.imshow(
                display_corr, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale="RdBu_r",
                title="Correlation Matrix"
            )
            fig_corr.update_layout(height=600, width=800)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Create pairplots for inputs vs outputs (one plot per output)
            for output_col in output_columns:
                display_output = display_names[output_col]
                st.subheader(f"Pairplot: Inputs vs {display_output}")
                
                # Create a figure with subplots - one row per 3 input variables
                rows = (len(input_columns) + 2) // 3
                
                # Create subplot titles with correlation coefficients
                subplot_titles = []
                for input_col in input_columns:
                    corr_val = corr.loc[input_col, output_col]
                    subplot_titles.append(f"{display_output} vs {input_col} | r = {corr_val:.2f}")
                
                fig = sp.make_subplots(
                    rows=rows, 
                    cols=3, 
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.15,
                    horizontal_spacing=0.05
                )
                
                # Add scatter plots for each input vs output
                for i, input_col in enumerate(input_columns):
                    row = i // 3 + 1
                    col = i % 3 + 1
                    
                    # Calculate correlation coefficient
                    corr_val = corr.loc[input_col, output_col]
                    
                    # Calculate linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        data[input_col], data[output_col]
                    )
                    
                    # Determine appropriate x-axis range with proper padding
                    x_min, x_max = data[input_col].min(), data[input_col].max()
                    x_range = x_max - x_min
                    
                    # Add 5% padding on each side
                    x_min = x_min - 0.05 * x_range
                    x_max = x_max + 0.05 * x_range
                    
                    # Generate evenly spaced points for the trendline
                    x_values = np.linspace(x_min, x_max, 100)
                    y_pred = slope * x_values + intercept
                    
                    # Create a reduced dataset for scatter plot to improve performance
                    # Use a maximum of 500 points for large datasets
                    if len(data) > 500:
                        # Create a downsampled version of the data
                        sample_size = 500
                        indices = np.random.choice(len(data), size=sample_size, replace=False)
                        scatter_x = data.iloc[indices][input_col]
                        scatter_y = data.iloc[indices][output_col]
                    else:
                        scatter_x = data[input_col]
                        scatter_y = data[output_col]
                    
                    # Add scatter plot with reduced hover information
                    fig.add_trace(
                        go.Scatter(
                            x=scatter_x,
                            y=scatter_y,
                            mode='markers',
                            marker=dict(
                                color='royalblue',
                                opacity=0.6,
                                size=8
                            ),
                            name=f"{input_col}",
                            hoverinfo='none',  # Disable hover to improve performance
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    
                    # Add trendline
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_pred,
                            mode='lines',
                            line=dict(color='firebrick', width=2),
                            name=f"Trend",
                            hoverinfo='none',  # Disable hover to improve performance
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                    
                    # Update axes labels
                    fig.update_xaxes(
                        title_text=input_col, 
                        row=row, 
                        col=col,
                        range=[x_min, x_max]  # Set explicit range to avoid bad scaling
                    )
                    
                    if col == 1:
                        fig.update_yaxes(title_text=display_output, row=row, col=col)
                
                # Update layout
                fig.update_layout(
                    height=300 * rows,
                    showlegend=False,
                    title_text=f"Relationship between {display_output} and Input Variables",
                    margin=dict(l=60, r=50, t=100, b=50)  # Increased top margin for titles
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # AI INSIGHTS SECTION
        if language_model:        
            with st.expander("AI Insights", expanded=True):
                with st.spinner("Generating AI insights..."):
                    # Prepare prompt for the AI
                    prompt = f"""
                    You are an expert in uncertainty quantification and statistics working for an enterprise-level engineering firm. Analyze the following data from a Monte Carlo simulation:
                    
                    Basic statistics:
                    {data.describe().to_string()}
                    
                    Additional statistics:
                    Skewness: {data[output_columns].skew().to_string()}
                    Kurtosis: {data[output_columns].kurtosis().to_string()}
                    
                    Correlation matrix:
                    {data.corr().to_string()}
                    
                    The model being analyzed is:
                    ```python
                    {model_code_str}
                    ```
                    
                    Based on this information, provide professional, enterprise-grade insights about:
                    1. The distribution characteristics of the output variable(s) and their implications for engineering reliability
                    2. Key relationships between inputs and outputs, focusing on the most significant correlations and their physical interpretation
                    3. Potential sources of uncertainty and their impact on decision-making and risk assessment
                    4. Recommendations for risk management based on the quantile information, with specific thresholds for different risk levels
                    5. Suggestions for further analysis or model refinement that could improve decision-making
                    6. Any interesting patterns or anomalies in the data that warrant further investigation
                    
                    Format your response with clear section headings and bullet points where appropriate. Keep your analysis concise, data-driven, and actionable for executive stakeholders. Include specific numerical values where relevant to support your conclusions.
                    
                    {RETURN_INSTRUCTION}
                    """
                    
                    # Cache the response in session state
                    response_key = 'exploratory_analysis_response_markdown'
                    
                    if response_key not in st.session_state:
                        # Call the AI API
                        response = call_groq_api(prompt, model_name=language_model)
                        st.session_state[response_key] = response
                    else:
                        response = st.session_state[response_key]
                    
                    # Display the AI insights
                    st.markdown(response)
    
    # CHAT INTERFACE in the right column
    with chat_col:
        st.markdown("### Ask Questions About This Analysis")
        
        # Display a disclaimer about the prompt
        disclaimer_text = """
        **Note:** The AI assistant has been provided with the model code, simulation data, 
        and the analysis results above. You can ask questions to clarify any aspects of the data analysis.
        """
        st.info(disclaimer_text)
        
        # Initialize session state for chat messages if not already done
        if "exploratory_analysis_chat_messages" not in st.session_state:
            st.session_state.exploratory_analysis_chat_messages = []
        
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
            for message in st.session_state.exploratory_analysis_chat_messages:
                role_style = "background-color: #e1f5fe; border-radius: 10px; padding: 8px; margin: 5px 0;" if message["role"] == "assistant" else "background-color: #f0f0f0; border-radius: 10px; padding: 8px; margin: 5px 0;"
                role_label = "Assistant:" if message["role"] == "assistant" else "You:"
                chat_messages_html += f"<div style='{role_style}'><strong>{role_label}</strong><br>{message['content']}</div>"
            
            chat_messages_html += "</div>"
            st.markdown(chat_messages_html, unsafe_allow_html=True)
        
        # Chat input below the scrollable container
        prompt = st.chat_input("Ask a question about the data analysis...", key="eda_side_chat_input")
        
        # Process user input
        if prompt:
            # Add user message to chat history
            st.session_state.exploratory_analysis_chat_messages.append({"role": "user", "content": prompt})
            
            # Define context generator function
            def generate_context(prompt):
                return f"""
                You are an expert assistant helping users understand exploratory data analysis results. 
                
                Here is the model code:
                ```python
                {model_code_str}
                ```
                
                Here is basic statistics information:
                {data.describe().to_string()}
                
                Additional statistics:
                Skewness: {data[output_columns].skew().to_string()}
                Kurtosis: {data[output_columns].kurtosis().to_string()}
                
                Correlation matrix:
                {data.corr().to_string()}
                
                Here is the explanation that was previously generated:
                {st.session_state.get('exploratory_analysis_response_markdown', 'No analysis available yet.')}
                
                Answer the user's question based on this information. Be concise but thorough.
                If you're not sure about something, acknowledge the limitations of your knowledge.
                Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                """
            
            # Generate context for the assistant
            context = generate_context(prompt)
            
            # Include previous conversation history
            chat_history = ""
            if len(st.session_state.exploratory_analysis_chat_messages) > 1:
                chat_history = "Previous conversation:\n"
                for i, msg in enumerate(st.session_state.exploratory_analysis_chat_messages[:-1]):
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
            st.session_state.exploratory_analysis_chat_messages.append({"role": "assistant", "content": response_text})
            
            # Rerun to display the new message immediately
            st.rerun()