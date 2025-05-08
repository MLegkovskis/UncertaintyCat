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

def compute_exploratory_data_analysis(data, N, model, problem, model_code_str):
    """
    Perform enterprise-grade exploratory data analysis calculations without UI display.
    
    This function computes all the necessary data for exploratory data analysis, including
    correlation analysis, distribution analysis, and statistical summaries.
    
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
        
    Returns
    -------
    dict
        Dictionary containing all exploratory data analysis results
    """
    # Check if data is available
    if data is None or data.empty:
        raise ValueError("No data available for analysis.")
    
    # Identify input and output columns and rename for display
    input_columns = [col for col in data.columns if col not in ['Y'] and not col.startswith('Y')]
    output_columns = [col for col in data.columns if col == 'Y' or col.startswith('Y')]
    
    # Create display names dictionary (for UI only, doesn't change dataframe)
    display_names = {}
    for col in data.columns:
        if col.startswith('X'):
            display_names[col] = f"Input {col[1:]}"
        elif col.startswith('Y'):
            if col == 'Y':
                display_names[col] = "Output"
            else:
                # For columns like Y1, Y2, etc.
                output_num = col[1:]  # Extract the number after 'Y'
                display_names[col] = f'Output {output_num}'
    
    if not output_columns:
        raise ValueError("No output columns (Y or Y1, Y2, etc.) found in the data.")
    
    # Compute correlation matrix
    corr = data.corr()
    
    # Create a displayable version of the correlation matrix
    display_corr = corr.copy()
    
    # Create a mapping of original column names to display names for the correlation matrix
    corr_display_names = {}
    for col in corr.columns:
        if col in display_names:
            corr_display_names[col] = display_names[col]
        else:
            corr_display_names[col] = col
    
    # Create a version of the correlation matrix with zeros on the diagonal for visualization
    corr_viz = corr.copy()
    np.fill_diagonal(corr_viz.values, 0)  # Zero out the diagonal
    
    # Format the correlation values for display
    for i in range(len(display_corr.columns)):
        for j in range(len(display_corr.index)):
            if i == j:
                display_corr.iloc[i, j] = np.nan  # Set diagonal to np.nan for no text
            else:
                display_corr.iloc[i, j] = round(display_corr.iloc[i, j], 3)
    
    # Create correlation heatmap with original column names and zeroed diagonal
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_viz.values,  # Use the matrix with zeroed diagonal
        x=[corr_display_names[col] for col in corr.columns],  # Use display names
        y=[corr_display_names[col] for col in corr.index],    # Use display names
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=display_corr.values,
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig_corr.update_layout(
        title="Correlation Matrix",
        height=600,
        width=800
    )
    
    # Create combined cross cuts and regression plots
    combined_plots = {}
    regression_data = {}
    
    for output_col in output_columns:
        output_combined_plots = {}
        output_regression_data = []
        
        # Get the nominal values of the input parameters
        input_dimension = problem.getDimension()
        input_names = problem.getDescription()
        nominal_point = problem.getMean()
        
        for i, input_col in enumerate(input_columns):
            input_name = input_names[i]
            
            # Create a 2x1 subplot figure with shared x-axis (stacked vertically)
            fig = sp.make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[
                    f"Cross Cut: {input_name} vs {display_names[output_col]}",
                    f"Regression: {input_name} vs {display_names[output_col]}"
                ]
            )
            
            # 1. Add cross cut plot (top)
            # Get the marginal distribution for this input
            marginal = problem.getMarginal(i)
            
            # Get range for this input (mean ± 3*std)
            mean = marginal.getMean()[0]
            std = marginal.getStandardDeviation()[0]
            x_min = max(mean - 3*std, marginal.getRange().getLowerBound()[0])
            x_max = min(mean + 3*std, marginal.getRange().getUpperBound()[0])
            
            # Generate points for the slice
            n_points = 100
            x_values = np.linspace(x_min, x_max, n_points)
            y_values = []
            
            # For each point, create a copy of the nominal point and vary only one dimension
            for x in x_values:
                # Create a new point with the same values as nominal_point
                point = ot.Point(nominal_point)
                # Set the value at index i to x
                point[i] = x
                y = model(point)
                
                # Handle different types of model outputs
                if isinstance(y, ot.Point):
                    # Convert OpenTURNS Point to list
                    if y.getDimension() > 1:
                        # For multiple outputs, just use the one we're currently plotting
                        output_idx = output_columns.index(output_col)
                        if output_idx < y.getDimension():
                            y_values.append(y[output_idx])
                        else:
                            y_values.append(y[0])
                    else:
                        y_values.append(y[0])
                elif isinstance(y, (list, tuple, np.ndarray)) and len(y) > 1:
                    # For multiple outputs, just use the one we're currently plotting
                    output_idx = output_columns.index(output_col)
                    if output_idx < len(y):
                        y_values.append(y[output_idx])
                    else:
                        y_values.append(y[0])
                else:
                    # Handle single output (try to convert to float)
                    try:
                        y_values.append(float(y))
                    except (TypeError, ValueError):
                        # If conversion fails, use the first element or the raw value
                        if hasattr(y, '__getitem__'):
                            y_values.append(y[0])
                        else:
                            y_values.append(y)
            
            # Add cross cut trace
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name='Cross Cut',
                    line=dict(color='royalblue', width=2)
                ),
                row=1, col=1
            )
            
            # 2. Add regression plot (bottom)
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data[input_col], data[output_col]
            )
            
            # Store regression details
            output_regression_data.append({
                'input': input_name,
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err
            })
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=data[input_col],
                    y=data[output_col],
                    mode='markers',
                    name='Data Points',
                    marker=dict(
                        size=6,
                        opacity=0.6,
                        color='royalblue'
                    ),
                    hoverinfo='none'  # Disable hover for better performance
                ),
                row=2, col=1
            )
            
            # Add regression line
            x_range = np.linspace(data[input_col].min(), data[input_col].max(), 100)
            y_pred = slope * x_range + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode='lines',
                    name=f'Regression (R²={r_value**2:.3f})',
                    line=dict(color='firebrick', width=2)
                ),
                row=2, col=1
            )
            
            # Add R² annotation
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref='x domain',
                yref='y2 domain',
                text=f'R² = {r_value**2:.3f}',
                showarrow=False,
                font=dict(
                    family="Arial",
                    size=12,
                    color="black"
                ),
                align="right",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            # Update layout
            fig.update_layout(
                height=600,  # Taller to accommodate vertical layout
                width=700,   # Narrower since we're stacking vertically
                title=f"{input_name} Effect on {display_names[output_col]}",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=80, b=80)  # Add margin for title and legend
            )
            
            # Update axis labels - only need x-axis label on bottom plot since they're shared
            fig.update_yaxes(title_text="Output Value", row=1, col=1)
            fig.update_xaxes(title_text=input_name, row=2, col=1)  # Only bottom x-axis needs label
            fig.update_yaxes(title_text="Output Value", row=2, col=1)
            
            output_combined_plots[input_name] = fig
        
        combined_plots[output_col] = output_combined_plots
        regression_data[output_col] = output_regression_data
    
    # Create 2D cross cuts (contour plots) for pairs of input variables
    cross_cuts_2d = {}
    
    # Only create 2D cross cuts if we have at least 2 input variables
    if input_dimension >= 2:
        # Create a grid of 2D cross cuts
        for i in range(input_dimension):
            for j in range(i+1, input_dimension):
                input_name_i = input_names[i]
                input_name_j = input_names[j]
                
                # Get the marginal distributions
                marginal_i = problem.getMarginal(i)
                marginal_j = problem.getMarginal(j)
                
                # Get ranges for these inputs (mean ± 3*std)
                mean_i = marginal_i.getMean()[0]
                std_i = marginal_i.getStandardDeviation()[0]
                x_min_i = max(mean_i - 3*std_i, marginal_i.getRange().getLowerBound()[0])
                x_max_i = min(mean_i + 3*std_i, marginal_i.getRange().getUpperBound()[0])
                
                mean_j = marginal_j.getMean()[0]
                std_j = marginal_j.getStandardDeviation()[0]
                x_min_j = max(mean_j - 3*std_j, marginal_j.getRange().getLowerBound()[0])
                x_max_j = min(mean_j + 3*std_j, marginal_j.getRange().getUpperBound()[0])
                
                # Generate grid points
                n_points = 30  # Reduced for performance
                x_values_i = np.linspace(x_min_i, x_max_i, n_points)
                x_values_j = np.linspace(x_min_j, x_max_j, n_points)
                X, Y = np.meshgrid(x_values_i, x_values_j)
                Z = np.zeros_like(X)
                
                # Compute function values at each grid point
                for ii in range(n_points):
                    for jj in range(n_points):
                        # Create a copy of the nominal point
                        point = ot.Point(nominal_point)
                        # Set the values at indices i and j
                        point[i] = X[jj, ii]
                        point[j] = Y[jj, ii]
                        y = model(point)
                        
                        # Handle different types of model outputs
                        try:
                            if isinstance(y, ot.Point):
                                if y.getDimension() > 1:
                                    # For multiple outputs, just use the first one for the contour
                                    Z[jj, ii] = y[0]
                                else:
                                    Z[jj, ii] = y[0]
                            elif isinstance(y, (list, tuple, np.ndarray)) and len(y) > 1:
                                # For multiple outputs, just use the first one for the contour
                                Z[jj, ii] = y[0]
                            else:
                                Z[jj, ii] = float(y)
                        except (TypeError, ValueError):
                            # If conversion fails, try to get the first element
                            if hasattr(y, '__getitem__'):
                                Z[jj, ii] = y[0]
                            else:
                                Z[jj, ii] = 0  # Fallback
                
                # Create the contour plot
                fig = go.Figure(data=
                    go.Contour(
                        z=Z,
                        x=x_values_i,
                        y=x_values_j,
                        colorscale='Viridis',
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=12, color='white')
                        ),
                        colorbar=dict(
                            title="Output"
                        )
                    )
                )
                
                fig.update_layout(
                    title=f"2D Cross Cut: Output vs {input_name_i} and {input_name_j}",
                    xaxis_title=input_name_i,
                    yaxis_title=input_name_j,
                    height=500,
                    width=600
                )
                
                cross_cuts_2d[f"{input_name_i} vs {input_name_j}"] = fig
    
    # Return all results as a dictionary
    return {
        'data': data,
        'input_columns': input_columns,
        'output_columns': output_columns,
        'display_names': display_names,
        'correlation_matrix': corr,
        'display_correlation_matrix': display_corr,
        'fig_corr': fig_corr,
        'combined_plots': combined_plots,
        'regression_data': regression_data,
        'cross_cuts_2d': cross_cuts_2d
    }

def get_eda_context_for_chat(eda_results):
    """
    Generate a formatted string containing exploratory data analysis results for the global chat context.
    
    Parameters
    ----------
    eda_results : dict
        Dictionary containing the results of the exploratory data analysis
        
    Returns
    -------
    str
        Formatted string with exploratory data analysis results for chat context
    """
    context = ""
    
    # Correlation matrix
    corr_matrix = eda_results.get("display_corr")
    if corr_matrix is not None:
        context += "\n\n### Exploratory Data Analysis: Correlation Matrix\n"
        try:
            context += corr_matrix.to_markdown(index=True)
        except Exception:
            pass
    
    # Regression results
    if "regression_data" in eda_results:
        regression_data = eda_results["regression_data"]
        if isinstance(regression_data, list) and regression_data:
            reg_df = pd.DataFrame(regression_data)
            context += "\n\n### Input-Output Regression Summary\n"
            context += reg_df.to_markdown(index=False)
    
    # AI insights
    ai_insights = eda_results.get("ai_insights")
    if ai_insights:
        context += f"\n#### AI Insights\n{ai_insights}\n"
    
    # Brief summary of available visualizations
    context += "\n- Cross cuts and regression plots are available for each input-output pair.\n- 2D cross cuts and contour plots visualize interactions between pairs of parameters.\n"
    
    return context

def generate_ai_insights(analysis_results, language_model='groq'):
    """Generate AI insights for exploratory data analysis results.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing all exploratory data analysis results
    language_model : str, optional
        Language model to use ('groq' or 'openai')
        
    Returns
    -------
    str
        AI-generated insights
    """
    try:
        # Extract key information
        data = analysis_results['data']
        output_columns = analysis_results['output_columns']
        corr = analysis_results['correlation_matrix']
        
        # Check if regression_data exists
        regression_data = analysis_results.get('regression_data', {})
        if not regression_data:
            # Create a placeholder if it doesn't exist
            regression_data = {output_col: [] for output_col in output_columns}
        
        # Extract cross cuts data for numerical analysis
        cross_cuts_data = {}
        if 'combined_plots' in analysis_results:
            for output_col in analysis_results['combined_plots']:
                cross_cuts_data[output_col] = {}
                for input_name in analysis_results['combined_plots'][output_col]:
                    # Extract the regression data for this input-output pair
                    reg_data = None
                    for reg in regression_data.get(output_col, []):
                        if reg.get('input') == input_name:
                            reg_data = reg
                            break
                    
                    if reg_data:
                        cross_cuts_data[output_col][input_name] = {
                            'r_squared': reg_data.get('r_squared', 0),
                            'slope': reg_data.get('slope', 0),
                            'p_value': reg_data.get('p_value', 1)
                        }
        
        # Prepare prompt with detailed statistical information
        prompt = f"""
        Analyze these exploratory data analysis results:
        
        Correlation matrix: {corr.to_string()}
        
        Cross cuts analysis:
        {json.dumps(cross_cuts_data, indent=2)}
        
        Provide insights on:
        1. Input-output relationships
        2. Correlations between variables
        3. Key patterns in the cross cuts
        4. Recommendations for further analysis
        
        Keep your analysis concise and focused on the practical implications.
        Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
        """
        
        # Call the AI API
        from utils.core_utils import call_groq_api
        
        # Add a retry mechanism
        max_retries = 3
        retry_count = 0
        insights = None
        
        while insights is None and retry_count < max_retries:
            try:
                insights = call_groq_api(prompt, model_name=language_model)
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    st.error(f"Error generating AI insights after {max_retries} attempts: {str(e)}")
                    insights = f"""
                    ## Error Generating Insights
                    
                    There was an error connecting to the language model API: {str(e)}
                    
                    Please try again later or check your API key configuration.
                    """
        
        return insights
    
    except Exception as e:
        return f"""
        ## Error Generating Insights
        
        An error occurred while preparing data for the AI: {str(e)}
        
        Please check your data and try again.
        """

def exploratory_data_analysis(data, N, model, problem, model_code_str, language_model='groq', display_results=True):
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
    display_results : bool, optional
        Whether to display results in the Streamlit interface, by default True
        
    Returns
    -------
    dict
        Dictionary containing all exploratory data analysis results
    """
    # Compute the exploratory data analysis
    try:
        results = compute_exploratory_data_analysis(data, N, model, problem, model_code_str)
        
        # Generate AI insights if a language model is specified
        if language_model:
            with st.spinner("Generating AI insights for Exploratory Data Analysis..."):
                insights = generate_ai_insights(results, language_model=language_model)
                results['ai_insights'] = insights
        
        # Save results to session state for later access and global chat
        if 'exploratory_data_analysis_results' not in st.session_state:
            st.session_state.exploratory_data_analysis_results = results
        
        # Display results if requested
        if display_results:
            display_exploratory_data_analysis_results(results, language_model)
        
        return results
    
    except Exception as e:
        if display_results:
            st.error(f"Error in exploratory data analysis: {str(e)}")
        raise e

def display_exploratory_data_analysis_results(analysis_results, language_model='groq'):
    """
    Display enterprise-grade exploratory data analysis results.
    
    This function creates and displays interactive visualizations and insights
    for the exploratory data analysis results.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing all exploratory data analysis results from compute_exploratory_data_analysis
    language_model : str, optional
        Language model to use for AI insights ('groq' or 'openai')
    """
    st.markdown("## Exploratory Data Analysis")
    
    # RESULTS SECTION
    with st.container():
        # Display correlation matrix
        st.subheader("Correlation Analysis")
        st.plotly_chart(analysis_results['fig_corr'], use_container_width=True)
        
        # Display combined cross cuts and regression plots
        st.subheader("Input-Output Relationships & Cross Cuts")
        st.markdown("""
        These plots show how each input parameter affects the output in two ways:
        1. **Cross Cut** (top): Shows how the output changes when varying one input parameter while keeping all others at their nominal values
        2. **Regression** (bottom): Shows the relationship between the input and output values from the Monte Carlo simulation
        """)
        
        # Create output tabs if multiple outputs
        if len(analysis_results['output_columns']) > 1:
            output_tabs = st.tabs([analysis_results['display_names'][col] for col in analysis_results['output_columns']])
            
            for output_col, tab in zip(analysis_results['output_columns'], output_tabs):
                with tab:
                    # Create tabs for each input variable
                    input_tabs = st.tabs(list(analysis_results['combined_plots'][output_col].keys()))
                    for i, (input_name, tab) in enumerate(zip(analysis_results['combined_plots'][output_col].keys(), input_tabs)):
                        with tab:
                            st.plotly_chart(analysis_results['combined_plots'][output_col][input_name], use_container_width=True)
        else:
            # If there's only one output, no need for output tabs
            output_col = analysis_results['output_columns'][0]
            
            # Create tabs for each input variable
            input_tabs = st.tabs(list(analysis_results['combined_plots'][output_col].keys()))
            for i, (input_name, tab) in enumerate(zip(analysis_results['combined_plots'][output_col].keys(), input_tabs)):
                with tab:
                    st.plotly_chart(analysis_results['combined_plots'][output_col][input_name], use_container_width=True)
        
        # Display 2D cross cuts if available
        if 'cross_cuts_2d' in analysis_results and analysis_results['cross_cuts_2d']:
            st.subheader("2D Cross Cuts of the Function")
            st.markdown("""
            These contour plots show how the output changes when varying two input parameters at a time,
            while keeping all other parameters at their nominal values. This helps identify
            interaction effects between pairs of parameters.
            """)
            
            # Create tabs for each 2D cross cut
            cross_cut_2d_tabs = st.tabs(list(analysis_results['cross_cuts_2d'].keys()))
            for i, (input_pair, tab) in enumerate(zip(analysis_results['cross_cuts_2d'].keys(), cross_cut_2d_tabs)):
                with tab:
                    st.plotly_chart(analysis_results['cross_cuts_2d'][input_pair], use_container_width=True)
    
    # AI INSIGHTS SECTION
    if 'ai_insights' in analysis_results and analysis_results['ai_insights']:
        st.subheader("AI Insights")
        # Store the insights in session state for reuse in the global chat
        if 'exploratory_data_analysis_response_markdown' not in st.session_state:
            st.session_state['exploratory_data_analysis_response_markdown'] = analysis_results['ai_insights']
        st.markdown(analysis_results['ai_insights'])