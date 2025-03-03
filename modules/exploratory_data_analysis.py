# modules/exploratory_data_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import scipy.stats as stats
from utils.core_utils import call_groq_api
from utils.markdown_utils import RETURN_INSTRUCTION
import seaborn as sns

def exploratory_data_analysis(data, N, model, problem, model_code_str, language_model='groq'):
    """Perform exploratory data analysis on Monte Carlo simulation results."""
    st.subheader("Exploratory Data Analysis")
    
    # Check if data is available
    if data is None or data.empty:
        st.error("No data available for analysis.")
        return
    
    # Identify input and output columns
    input_columns = [col for col in data.columns if col not in ['Y'] and not col.startswith('Y')]
    output_columns = [col for col in data.columns if col == 'Y' or col.startswith('Y')]
    
    if not output_columns:
        raise KeyError("No output columns (Y or Y1, Y2, etc.) found in the data.")
    
    # Display basic statistics for all variables
    st.write("### Basic Statistics")
    st.write(data.describe())
    
    # Create histograms for each variable
    st.write("### Histograms")
    
    # Create a grid of histograms
    num_cols = len(data.columns)
    cols_per_row = 3
    num_rows = (num_cols + cols_per_row - 1) // cols_per_row
    
    for i in range(0, num_cols, cols_per_row):
        cols = st.columns(min(cols_per_row, num_cols - i))
        for j, col in enumerate(cols):
            if i + j < num_cols:
                var_name = data.columns[i + j]
                with col:
                    fig = px.histogram(data, x=var_name, title=f"Histogram of {var_name}")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Create correlation matrix
    st.write("### Correlation Matrix")
    corr = data.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    fig.update_layout(height=600, width=800)
    st.plotly_chart(fig)
    
    # Create scatter plots for inputs vs outputs
    st.write("### Scatter Plots: Inputs vs Outputs")
    
    for output_col in output_columns:
        for input_col in input_columns:
            fig = px.scatter(data, x=input_col, y=output_col, 
                            title=f"{output_col} vs {input_col}",
                            trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
    
    # Generate AI insights if requested
    if language_model:
        st.write("### AI Insights")
        with st.spinner("Generating AI insights..."):
            # Prepare prompt for the AI
            prompt = f"""
            You are an expert in uncertainty quantification and statistics. Analyze the following data from a Monte Carlo simulation:
            
            Basic statistics:
            {data.describe().to_string()}
            
            Correlation matrix:
            {data.corr().to_string()}
            
            The model being analyzed is:
            ```python
            {model_code_str}
            ```
            
            Based on this information, provide insights about:
            1. The distribution of the output variable(s)
            2. Key relationships between inputs and outputs
            3. Potential sources of uncertainty
            4. Any interesting patterns or anomalies in the data
            
            {RETURN_INSTRUCTION}
            """
            
            # Call the AI API
            response = call_groq_api(prompt)
            
            # Display the AI insights
            st.markdown(response)