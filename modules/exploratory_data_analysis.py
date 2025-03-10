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
from utils.core_utils import call_groq_api
from utils.markdown_utils import RETURN_INSTRUCTION
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union

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
    st.subheader("Exploratory Data Analysis")
    
    # Check if data is available
    if data is None or data.empty:
        st.error("No data available for analysis.")
        return
    
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
        raise KeyError("No output columns (Y or Y1, Y2, etc.) found in the data.")
    
    # Create correlation analysis
    with st.expander("Correlation Analysis", expanded=True):
        st.write("### Correlation Matrix and Pairplots")
        
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
            st.write(f"#### Pairplot: Inputs vs {display_output}")
            
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
                vertical_spacing=0.15,  # Increased vertical spacing
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
    
    # Output distribution analysis with integrated statistics
    with st.expander("Output Analysis", expanded=True):
        st.write("### Output Distribution Analysis")
        
        # Calculate statistics for outputs
        output_stats = data[output_columns].describe().copy()
        output_stats.rename(columns=display_names, inplace=True)
        
        skewness = data[output_columns].skew()
        kurtosis = data[output_columns].kurtosis()
        
        additional_stats = pd.DataFrame({
            'Skewness': skewness,
            'Kurtosis': kurtosis
        })
        additional_stats.rename(columns=display_names, inplace=True)
        
        # Display statistics tables
        st.write("#### Output Summary Statistics")
        st.dataframe(output_stats, use_container_width=True)
        
        st.write("#### Additional Output Statistics")
        st.dataframe(additional_stats, use_container_width=True)
        
        # Create a figure with subplots - one row per output
        subplot_titles = []
        for output_col in output_columns:
            display_output = display_names[output_col]
            subplot_titles.extend([f"Box Plot of {display_output}", f"Distribution of {display_output}"])
            
        fig = sp.make_subplots(
            rows=len(output_columns), 
            cols=2,
            subplot_titles=subplot_titles,
            specs=[[{"type": "box"}, {"type": "histogram"}] for _ in output_columns],
            vertical_spacing=0.15
        )
        
        for i, output_col in enumerate(output_columns):
            row = i + 1
            display_output = display_names[output_col]
            
            # Add box plot
            fig.add_trace(
                go.Box(
                    y=data[output_col],
                    name=display_output,
                    boxmean=True,
                    marker_color='royalblue',
                    boxpoints='outliers'
                ),
                row=row, col=1
            )
            
            # Add histogram with KDE
            hist_data = [data[output_col]]
            group_labels = [display_output]
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=data[output_col],
                    histnorm='probability density',
                    name=display_output,
                    marker=dict(color='royalblue', opacity=0.6)
                ),
                row=row, col=2
            )
            
            # Add KDE
            kde_x = np.linspace(data[output_col].min(), data[output_col].max(), 1000)
            kde = stats.gaussian_kde(data[output_col])
            kde_y = kde(kde_x)
            
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde_y,
                    mode='lines',
                    name='KDE',
                    line=dict(color='firebrick', width=2)
                ),
                row=row, col=2
            )
            
            # Add reference lines for mean and median
            mean_val = data[output_col].mean()
            median_val = data[output_col].median()
            
            # Add mean line to histogram
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Mean: {mean_val:.2f}",
                annotation_position="top right",
                row=row, col=2
            )
            
            # Add median line to histogram
            fig.add_vline(
                x=median_val,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Median: {median_val:.2f}",
                annotation_position="top left",
                row=row, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=400 * len(output_columns),
            showlegend=False,
            title_text="Output Distribution Analysis",
            margin=dict(l=60, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add quantile information
        st.write("#### Quantile Information")
        
        quantiles_df = pd.DataFrame()
        for output_col in output_columns:
            display_output = display_names[output_col]
            quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            quantile_values = np.quantile(data[output_col], quantiles)
            
            quantiles_df[display_output] = quantile_values
        
        quantiles_df.index = [f"{q*100}%" for q in quantiles]
        st.dataframe(quantiles_df, use_container_width=True)
    
    # Generate AI insights if requested
    if language_model:
        with st.expander("AI Insights", expanded=True):
            st.write("### AI-Generated Insights")
            with st.spinner("Generating AI insights..."):
                # Prepare prompt for the AI
                prompt = f"""
                You are an expert in uncertainty quantification and statistics working for an enterprise-level engineering firm. Analyze the following data from a Monte Carlo simulation:
                
                Basic statistics:
                {data.describe().to_string()}
                
                Additional statistics:
                Skewness: {skewness.to_string()}
                Kurtosis: {kurtosis.to_string()}
                
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
                
                # Call the AI API
                response = call_groq_api(prompt)
                
                # Display the AI insights
                st.markdown(response)