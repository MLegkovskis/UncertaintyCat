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
    for col in output_columns:
        if col == 'Y':
            display_names[col] = 'Output'
        else:
            # For columns like Y1, Y2, etc.
            output_num = col[1:]  # Extract the number after 'Y'
            display_names[col] = f'Output {output_num}'
    
    if not output_columns:
        raise ValueError("No output columns (Y or Y1, Y2, etc.) found in the data.")
    
    # Create correlation matrix
    corr = data.corr()
    
    # Create a copy for display with renamed columns
    display_corr = corr.copy()
    display_corr.rename(columns=display_names, index=display_names, inplace=True)
    
    # Calculate statistics for each output
    output_stats = {}
    for output_col in output_columns:
        output_data = data[output_col]
        
        # Calculate basic statistics
        stats_dict = {
            'mean': output_data.mean(),
            'median': output_data.median(),
            'std': output_data.std(),
            'min': output_data.min(),
            'max': output_data.max(),
            'skewness': stats.skew(output_data),
            'kurtosis': stats.kurtosis(output_data),
            'q1': np.percentile(output_data, 25),
            'q3': np.percentile(output_data, 75),
            'iqr': np.percentile(output_data, 75) - np.percentile(output_data, 25)
        }
        
        # Calculate top correlations
        correlations = corr[output_col].drop(output_columns).sort_values(ascending=False)
        top_positive = correlations.head(3)
        top_negative = correlations.sort_values().head(3)
        
        stats_dict['top_positive_corr'] = top_positive
        stats_dict['top_negative_corr'] = top_negative
        
        # Store in output_stats
        output_stats[output_col] = stats_dict
    
    # Create pairplot data for each output
    pairplot_data = {}
    for output_col in output_columns:
        # For each output, select a subset of most correlated inputs (up to 5)
        correlations = corr[output_col].drop(output_columns).abs().sort_values(ascending=False)
        top_inputs = correlations.head(5).index.tolist()
        
        # Create a subset dataframe with these inputs and the output
        subset_df = data[top_inputs + [output_col]].copy()
        
        # Rename the output column for display
        subset_df.rename(columns={output_col: display_names[output_col]}, inplace=True)
        
        # Store in pairplot_data
        pairplot_data[output_col] = {
            'df': subset_df,
            'top_inputs': top_inputs,
            'display_name': display_names[output_col]
        }
    
    # Calculate distribution fit for each output
    distribution_fits = {}
    for output_col in output_columns:
        output_data = data[output_col]
        
        # Define distributions to test
        distributions_to_fit = [
            ('Normal', stats.norm),
            ('Lognormal', stats.lognorm),
            ('Gamma', stats.gamma),
            ('Weibull', stats.weibull_min),
            ('Beta', stats.beta)
        ]
        
        # Fit distributions and perform goodness-of-fit tests
        fit_results = []
        
        for dist_name, dist in distributions_to_fit:
            try:
                # Fit distribution
                params = dist.fit(output_data)
                
                # Calculate AIC and BIC
                log_likelihood = np.sum(dist.logpdf(output_data, *params))
                k = len(params)
                n = len(output_data)
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                # Perform Kolmogorov-Smirnov test
                ks_statistic, ks_pvalue = stats.kstest(output_data, dist.cdf, args=params)
                
                # Store results
                fit_results.append({
                    'Distribution': dist_name,
                    'Parameters': params,
                    'AIC': aic,
                    'BIC': bic,
                    'KS_Statistic': ks_statistic,
                    'KS_pvalue': ks_pvalue
                })
            except Exception as e:
                # Skip distributions that fail to fit
                continue
        
        # Convert fit results to DataFrame and sort by AIC
        if fit_results:
            fit_df = pd.DataFrame(fit_results)
            fit_df = fit_df.sort_values('AIC')
            best_distribution = fit_df.iloc[0]['Distribution']
            best_params = fit_df.iloc[0]['Parameters']
        else:
            fit_df = pd.DataFrame()
            best_distribution = None
            best_params = None
        
        distribution_fits[output_col] = {
            'fit_df': fit_df,
            'best_distribution': best_distribution,
            'best_params': best_params
        }
    
    # Return all results
    return {
        'data': data,
        'input_columns': input_columns,
        'output_columns': output_columns,
        'display_names': display_names,
        'corr': corr,
        'display_corr': display_corr,
        'output_stats': output_stats,
        'pairplot_data': pairplot_data,
        'distribution_fits': distribution_fits,
        'N': N,
        'model_code_str': model_code_str
    }

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
        Whether to display results using Streamlit UI (default: True)
        Set to False when running in batch mode or "Run All Analyses"
        
    Returns
    -------
    dict
        Dictionary containing all exploratory data analysis results
    """
    try:
        # Compute all exploratory data analysis results
        results = compute_exploratory_data_analysis(data, N, model, problem, model_code_str)
        
        # Save results to session state for later access
        if 'exploratory_data_results' not in st.session_state:
            st.session_state.exploratory_data_results = results
        
        # If not displaying results, just return the computed data
        if not display_results:
            return results
        
        # Extract data from results
        data = results['data']
        input_columns = results['input_columns']
        output_columns = results['output_columns']
        display_names = results['display_names']
        corr = results['corr']
        display_corr = results['display_corr']
        output_stats = results['output_stats']
        pairplot_data = results['pairplot_data']
        distribution_fits = results['distribution_fits']
        
        # Create a two-column layout for the main content and chat interface
        main_col, chat_col = st.columns([2, 1])
        
        with main_col:
            st.markdown("## Exploratory Data Analysis")
            
            # RESULTS SECTION
            with st.expander("Results", expanded=True):
                # Create correlation matrix
                st.subheader("Correlation Analysis")
                
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
                    
                    # Get the pairplot data for this output
                    pairplot_info = pairplot_data[output_col]
                    subset_df = pairplot_info['df']
                    top_inputs = pairplot_info['top_inputs']
                    
                    # Create pairplot using plotly
                    fig = px.scatter_matrix(
                        subset_df,
                        dimensions=top_inputs,
                        color=display_output,
                        opacity=0.7,
                        title=f"Pairplot of Top Correlated Inputs vs {display_output}"
                    )
                    fig.update_layout(height=800, width=800)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display correlation values for this output
                    st.markdown(f"#### Top Correlations with {display_output}")
                    
                    # Get top positive and negative correlations
                    top_pos = output_stats[output_col]['top_positive_corr']
                    top_neg = output_stats[output_col]['top_negative_corr']
                    
                    # Display in two columns
                    pos_col, neg_col = st.columns(2)
                    
                    with pos_col:
                        st.markdown("**Strongest Positive Correlations:**")
                        for var, corr_val in top_pos.items():
                            st.markdown(f"- **{var}**: {corr_val:.4f}")
                    
                    with neg_col:
                        st.markdown("**Strongest Negative Correlations:**")
                        for var, corr_val in top_neg.items():
                            st.markdown(f"- **{var}**: {corr_val:.4f}")
                
                # Distribution Analysis for each output
                for output_col in output_columns:
                    display_output = display_names[output_col]
                    st.subheader(f"Distribution Analysis: {display_output}")
                    
                    # Get the output data and stats
                    output_data = data[output_col]
                    stats_dict = output_stats[output_col]
                    fit_info = distribution_fits[output_col]
                    
                    # Display summary statistics
                    st.markdown("#### Summary Statistics")
                    
                    # Create a summary table
                    summary_df = pd.DataFrame({
                        'Statistic': [
                            'Mean', 
                            'Median', 
                            'Standard Deviation', 
                            'Minimum',
                            'Maximum',
                            'Skewness',
                            'Kurtosis',
                            'Interquartile Range (IQR)'
                        ],
                        'Value': [
                            f"{stats_dict['mean']:.6f}",
                            f"{stats_dict['median']:.6f}",
                            f"{stats_dict['std']:.6f}",
                            f"{stats_dict['min']:.6f}",
                            f"{stats_dict['max']:.6f}",
                            f"{stats_dict['skewness']:.4f} ({'Positively Skewed' if stats_dict['skewness'] > 0 else 'Negatively Skewed' if stats_dict['skewness'] < 0 else 'Symmetric'})",
                            f"{stats_dict['kurtosis']:.4f} ({'Leptokurtic (heavy-tailed)' if stats_dict['kurtosis'] > 0 else 'Platykurtic (light-tailed)' if stats_dict['kurtosis'] < 0 else 'Mesokurtic (normal-like)'})",
                            f"{stats_dict['iqr']:.6f}"
                        ]
                    })
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Create distribution visualization
                    st.markdown("#### Distribution Visualization")
                    
                    # Create a figure with histogram and box plot
                    fig_dist = sp.make_subplots(
                        rows=2, 
                        cols=1,
                        subplot_titles=["Histogram with KDE", "Box Plot"],
                        vertical_spacing=0.2,
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Add histogram
                    fig_dist.add_trace(
                        go.Histogram(
                            x=output_data,
                            histnorm='probability density',
                            name="Histogram",
                            marker=dict(color='royalblue', opacity=0.7)
                        ),
                        row=1, col=1
                    )
                    
                    # Add KDE
                    kde_x = np.linspace(min(output_data), max(output_data), 1000)
                    kde = stats.gaussian_kde(output_data)
                    kde_y = kde(kde_x)
                    
                    fig_dist.add_trace(
                        go.Scatter(
                            x=kde_x,
                            y=kde_y,
                            mode='lines',
                            name='KDE',
                            line=dict(color='firebrick', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add best fit distribution if available
                    best_distribution = fit_info['best_distribution']
                    best_params = fit_info['best_params']
                    
                    if best_distribution:
                        # Generate points for the fitted distribution
                        x_range = np.linspace(min(output_data), max(output_data), 1000)
                        
                        if best_distribution == 'Normal':
                            y_fit = stats.norm.pdf(x_range, *best_params)
                            dist_name = f"Normal(μ={best_params[0]:.4f}, σ={best_params[1]:.4f})"
                        elif best_distribution == 'Lognormal':
                            y_fit = stats.lognorm.pdf(x_range, *best_params)
                            dist_name = f"Lognormal(s={best_params[0]:.4f}, loc={best_params[1]:.4f}, scale={best_params[2]:.4f})"
                        elif best_distribution == 'Gamma':
                            y_fit = stats.gamma.pdf(x_range, *best_params)
                            dist_name = f"Gamma(a={best_params[0]:.4f}, loc={best_params[1]:.4f}, scale={best_params[2]:.4f})"
                        elif best_distribution == 'Weibull':
                            y_fit = stats.weibull_min.pdf(x_range, *best_params)
                            dist_name = f"Weibull(c={best_params[0]:.4f}, loc={best_params[1]:.4f}, scale={best_params[2]:.4f})"
                        elif best_distribution == 'Beta':
                            y_fit = stats.beta.pdf(x_range, *best_params)
                            dist_name = f"Beta(a={best_params[0]:.4f}, b={best_params[1]:.4f}, loc={best_params[2]:.4f}, scale={best_params[3]:.4f})"
                        
                        fig_dist.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=y_fit,
                                mode='lines',
                                name=f'Best Fit: {dist_name}',
                                line=dict(color='green', width=2)
                            ),
                            row=1, col=1
                        )
                    
                    # Add box plot
                    fig_dist.add_trace(
                        go.Box(
                            x=output_data,
                            name=display_output,
                            boxmean=True,
                            marker_color='royalblue'
                        ),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig_dist.update_layout(
                        height=600,
                        title_text=f"Distribution Analysis for {display_output}",
                        showlegend=True
                    )
                    
                    # Update axis labels
                    fig_dist.update_xaxes(title_text=display_output, row=2, col=1)
                    fig_dist.update_yaxes(title_text="Probability Density", row=1, col=1)
                    
                    # Display the figure
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Display distribution fitting results if available
                    fit_df = fit_info['fit_df']
                    if not fit_df.empty:
                        st.markdown("#### Distribution Fitting Results")
                        
                        # Format the dataframe for display
                        display_fit_df = fit_df.copy()
                        display_fit_df = display_fit_df[['Distribution', 'AIC', 'BIC', 'KS_Statistic', 'KS_pvalue']]
                        display_fit_df['AIC'] = display_fit_df['AIC'].map('{:.2f}'.format)
                        display_fit_df['BIC'] = display_fit_df['BIC'].map('{:.2f}'.format)
                        display_fit_df['KS_Statistic'] = display_fit_df['KS_Statistic'].map('{:.4f}'.format)
                        display_fit_df['KS_pvalue'] = display_fit_df['KS_pvalue'].map('{:.4f}'.format)
                        
                        st.dataframe(display_fit_df, use_container_width=True)
                        
                        # Add explanation of the metrics
                        st.markdown("""
                        **Interpretation of Metrics:**
                        - **AIC (Akaike Information Criterion)**: Lower values indicate better fit. Penalizes complexity.
                        - **BIC (Bayesian Information Criterion)**: Lower values indicate better fit. Penalizes complexity more strongly than AIC.
                        - **KS_Statistic**: Kolmogorov-Smirnov test statistic. Lower values indicate better fit.
                        - **KS_pvalue**: p-value for the KS test. Higher values indicate better fit. Values above 0.05 suggest the distribution is a good fit.
                        """)
            
            # AI Insights Section
            if language_model:
                with st.expander("AI Insights", expanded=True):
                    # Prepare the prompt
                    # Use the first output for simplicity
                    primary_output = output_columns[0]
                    display_output = display_names[primary_output]
                    stats_dict = output_stats[primary_output]
                    
                    # Get top correlations for the primary output
                    top_pos = output_stats[primary_output]['top_positive_corr']
                    top_neg = output_stats[primary_output]['top_negative_corr']
                    
                    # Format correlations for the prompt
                    corr_text = "Top Positive Correlations:\n"
                    for var, corr_val in top_pos.items():
                        corr_text += f"- {var}: {corr_val:.4f}\n"
                    
                    corr_text += "\nTop Negative Correlations:\n"
                    for var, corr_val in top_neg.items():
                        corr_text += f"- {var}: {corr_val:.4f}\n"
                    
                    prompt = f"""
{RETURN_INSTRUCTION}

Analyze these exploratory data analysis results for an enterprise-grade engineering model:

```python
{model_code_str}
```

Sample Size: {N}

Output Statistics for {display_output}:
- Mean: {stats_dict['mean']:.6f}
- Median: {stats_dict['median']:.6f}
- Standard Deviation: {stats_dict['std']:.6f}
- Skewness: {stats_dict['skewness']:.4f}
- Kurtosis: {stats_dict['kurtosis']:.4f}

{corr_text}

Please provide a comprehensive enterprise-grade analysis of these exploratory data analysis results. Your analysis should include:

1. Executive Summary
   - Key findings and their business/engineering implications
   - Assessment of the output distribution characteristics
   - Overall evaluation of the input-output relationships

2. Technical Analysis
   - Interpretation of the output distribution (shape, center, spread)
   - Analysis of the correlation patterns between inputs and outputs
   - Identification of the most influential input variables
   - Assessment of potential nonlinearities or interactions

3. Risk Assessment
   - Implications of the distribution characteristics for risk management
   - Potential areas of concern based on the correlations
   - Reliability considerations based on the data patterns

4. Recommendations
   - Guidance for system optimization based on the exploratory analysis
   - Suggestions for further analysis or model refinement
   - Variables that should be closely monitored or controlled

Focus on actionable insights that would be valuable for executive decision-makers in an engineering context.
"""
                    
                    with st.spinner("Generating expert analysis..."):
                        if 'exploratory_analysis_response' not in st.session_state:
                            response = call_groq_api(prompt, model_name=language_model)
                            st.session_state.exploratory_analysis_response = response
                        else:
                            response = st.session_state.exploratory_analysis_response
                        
                        st.markdown(response)
        
        # CHAT INTERFACE in the right column
        with chat_col:
            st.markdown("### Ask Questions About This Analysis")
            
            # Display a disclaimer about the prompt
            disclaimer_text = """
            **Note:** The AI assistant has been provided with the model code and the 
            exploratory data analysis results. You can ask questions to clarify any aspects of the analysis.
            """
            st.info(disclaimer_text)
            
            # Initialize session state for chat messages if not already done
            if "exploratory_analysis_chat_messages" not in st.session_state:
                st.session_state.exploratory_analysis_chat_messages = []
            
            # Create chat interface
            create_chat_interface(
                "exploratory_analysis_chat",
                lambda prompt: f"""
                You are an expert assistant helping users understand exploratory data analysis results. 
                
                Here is the model code:
                ```python
                {model_code_str}
                ```
                
                Here is the exploratory data analysis summary for {display_output}:
                - Mean: {stats_dict['mean']:.6f}
                - Median: {stats_dict['median']:.6f}
                - Standard Deviation: {stats_dict['std']:.6f}
                - Skewness: {stats_dict['skewness']:.4f}
                - Kurtosis: {stats_dict['kurtosis']:.4f}
                
                {corr_text}
                
                Here is the explanation that was previously generated:
                {st.session_state.get('eda_response', 'No analysis available yet.')}
                
                Answer the user's question: {prompt}
                
                Be concise but thorough. Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                """,
                input_placeholder="Ask a question about the exploratory analysis...",
                disclaimer_text="Ask questions about the exploratory data analysis results.",
                language_model=language_model
            )
        
        return results
        
    except Exception as e:
        if display_results:
            st.error(f"Error in exploratory data analysis: {str(e)}")
        raise

def display_exploratory_data_results(analysis_results, language_model='groq'):
    """
    Display exploratory data analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing exploratory data analysis results
    language_model : str, optional
        Language model to use for AI insights, by default 'groq'
    """
    # Extract results
    df = analysis_results.get('data')
    input_names = analysis_results.get('input_names')
    output_names = analysis_results.get('output_names')
    histograms = analysis_results.get('histograms')
    scatter_plots = analysis_results.get('scatter_plots')
    correlation_matrix = analysis_results.get('correlation_matrix')
    summary_stats = analysis_results.get('summary_stats')
    eda_interpretation = analysis_results.get('eda_interpretation')
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(summary_stats, use_container_width=True)
    
    # Display histograms
    st.subheader("Histograms")
    for hist in histograms:
        st.plotly_chart(hist, use_container_width=True)
    
    # Display scatter plots
    st.subheader("Scatter Plots")
    for scatter in scatter_plots:
        st.plotly_chart(scatter, use_container_width=True)
    
    # Display correlation matrix
    st.subheader("Correlation Matrix")
    st.plotly_chart(correlation_matrix, use_container_width=True)
    
    # Display AI interpretation if available
    if eda_interpretation:
        st.subheader("AI Insights")
        st.markdown(eda_interpretation)