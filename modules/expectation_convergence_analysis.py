import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
from utils.constants import RETURN_INSTRUCTION
from utils.core_utils import call_groq_api, create_chat_interface
from utils.model_utils import get_ot_model
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import uuid
import json
import os
from groq import Groq

def compute_expectation_convergence_analysis(model, problem, model_code_str, N_samples=8000):
    """Perform enterprise-grade expectation convergence analysis calculations without UI display.
    
    This function computes all the necessary data for convergence analysis, including
    mean and standard deviation convergence, distribution analysis, and statistical tests.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str
        String representation of the model code
    N_samples : int, optional
        Maximum number of samples for convergence analysis
        
    Returns
    -------
    dict
        Dictionary containing all convergence analysis results
    """
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Use the distribution directly
    distribution = problem

    # Create OpenTURNS model if needed
    ot_model = get_ot_model(model)

    # Define the input random vector and the output random vector
    input_random_vector = ot.RandomVector(distribution)
    output_random_vector = ot.CompositeRandomVector(ot_model, input_random_vector)

    # Run expectation simulation algorithm
    expectation_algo = ot.ExpectationSimulationAlgorithm(output_random_vector)
    expectation_algo.setMaximumOuterSampling(N_samples)
    expectation_algo.setBlockSize(1)
    expectation_algo.setCoefficientOfVariationCriterionType("NONE")

    expectation_algo.run()
    result = expectation_algo.getResult()

    # Extract convergence data
    graph = expectation_algo.drawExpectationConvergence()
    data = graph.getDrawable(0).getData()

    # Convert OpenTURNS Samples to numpy arrays
    sample_sizes = np.array([s[0] for s in data[:, 0]])
    mean_estimates = np.array([m[0] for m in data[:, 1]])

    # Calculate 95% confidence intervals using the final standard deviation
    final_std_dev = result.getStandardDeviation()[0]
    initial_sample_size = sample_sizes[-1]
    standard_errors = final_std_dev * np.sqrt(initial_sample_size / sample_sizes)
    z_value = 1.96  # 95% confidence
    lower_bounds = mean_estimates - z_value * standard_errors
    upper_bounds = mean_estimates + z_value * standard_errors
    
    # Use the required sample size from convergence analysis for distribution analysis
    convergence_sample_size = int(initial_sample_size)
    
    # Generate samples for distribution analysis
    input_sample = distribution.getSample(convergence_sample_size)
    output_sample = ot_model(input_sample)
    Y_values = np.array(output_sample).flatten()
    
    # Calculate statistics for distribution analysis
    mean_Y = np.mean(Y_values)
    std_Y = np.std(Y_values)
    conf_int = [mean_Y - 1.96 * std_Y, mean_Y + 1.96 * std_Y]
    skewness = stats.skew(Y_values)
    kurtosis = stats.kurtosis(Y_values)
    q1, q3 = np.percentile(Y_values, [25, 75])
    iqr = q3 - q1
    
    # Calculate standard deviation convergence
    std_dev_estimates = []
    
    # Compute running standard deviation for each sample size
    for i, size in enumerate(sample_sizes):
        if i == 0:
            # For the first point, use the standard deviation from the first sample
            std_dev_estimates.append(np.std(Y_values[:int(size)]))
        else:
            # For subsequent points, compute standard deviation based on accumulated samples
            std_dev_estimates.append(np.std(Y_values[:int(size)]))
    
    std_dev_estimates = np.array(std_dev_estimates)
    
    # Create inputs DataFrame for the prompt
    dimension = distribution.getDimension()
    input_parameters = []
    for i in range(dimension):
        marginal = distribution.getMarginal(i)
        name = marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}"
        input_parameters.append({
            'Variable': name,
            'Distribution': marginal.__class__.__name__,
            'Parameters': list(marginal.getParameter())
        })
    inputs_df = pd.DataFrame(input_parameters)
    
    # Fit distributions to the output data
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
            params = dist.fit(Y_values)
            
            # Calculate AIC and BIC
            log_likelihood = np.sum(dist.logpdf(Y_values, *params))
            k = len(params)
            n = len(Y_values)
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, ks_pvalue = stats.kstest(Y_values, dist.cdf, args=params)
            
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
    
    # Return all results
    return {
        'sample_sizes': sample_sizes,
        'mean_estimates': mean_estimates,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'std_dev_estimates': std_dev_estimates,
        'final_std_dev': final_std_dev,
        'convergence_sample_size': convergence_sample_size,
        'Y_values': Y_values,
        'mean_Y': mean_Y,
        'std_Y': std_Y,
        'conf_int': conf_int,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'fit_df': fit_df,
        'best_distribution': best_distribution,
        'best_params': best_params,
        'input_parameters': input_parameters,
        'inputs_df': inputs_df,
        'model_code_str': model_code_str
    }

def expectation_convergence_analysis(model, problem, model_code_str, N_samples=8000, language_model='groq', display_results=True):
    """Perform enterprise-grade expectation convergence analysis.
    
    This module provides comprehensive analysis of Monte Carlo convergence,
    including both mean and standard deviation convergence, distribution analysis,
    and statistical tests for convergence.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str
        String representation of the model code
    N_samples : int, optional
        Maximum number of samples for convergence analysis
    language_model : str, optional
        Language model to use for AI insights
    display_results : bool, optional
        Whether to display results using Streamlit UI (default: True)
        Set to False when running in batch mode or "Run All Analyses"
        
    Returns
    -------
    dict
        Dictionary containing all convergence analysis results
    """
    try:
        # Compute all convergence analysis results
        results = compute_expectation_convergence_analysis(model, problem, model_code_str, N_samples)
        
        # Save results to session state for later access
        if 'expectation_convergence_results' not in st.session_state:
            st.session_state.expectation_convergence_results = results
        
        # If not displaying results, just return the computed data
        if not display_results:
            return results
            
        # Extract data from results
        sample_sizes = results['sample_sizes']
        mean_estimates = results['mean_estimates']
        lower_bounds = results['lower_bounds']
        upper_bounds = results['upper_bounds']
        std_dev_estimates = results['std_dev_estimates']
        final_std_dev = results['final_std_dev']
        convergence_sample_size = results['convergence_sample_size']
        Y_values = results['Y_values']
        mean_Y = results['mean_Y']
        std_Y = results['std_Y']
        conf_int = results['conf_int']
        skewness = results['skewness']
        kurtosis = results['kurtosis']
        q1 = results['q1']
        q3 = results['q3']
        iqr = results['iqr']
        fit_df = results['fit_df']
        best_distribution = results['best_distribution']
        best_params = results['best_params']
        input_parameters = results['input_parameters']
        inputs_df = results['inputs_df']
        
        # RESULTS SECTION    
        main_col, chat_col = st.columns([2, 1])
        
        with main_col:
            with st.expander("Results", expanded=True):
                # Convergence Statistics
                st.subheader("Convergence Statistics")
                
                # Create a summary table for quick reference
                summary_df = pd.DataFrame({
                    'Metric': [
                        'Final Mean Estimate', 
                        'Standard Deviation', 
                        '95% Confidence Interval', 
                        'Required Sample Size',
                        'Relative Standard Error',
                        'Coefficient of Variation'
                    ],
                    'Value': [
                        f"{mean_estimates[-1]:.6f}",
                        f"{final_std_dev:.6f}",
                        f"[{lower_bounds[-1]:.6f}, {upper_bounds[-1]:.6f}]",
                        f"{convergence_sample_size}",
                        f"{(final_std_dev / mean_estimates[-1]):.4%}",
                        f"{(final_std_dev / np.abs(mean_estimates[-1])):.4%}"
                    ]
                })
                st.dataframe(summary_df, use_container_width=True)
                
                # Combined Convergence Visualization
                st.subheader("Convergence Visualization")
                
                # Create Plotly figure with 2x2 subplots for combined convergence analysis
                fig_convergence = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        "Mean Convergence (Linear Scale)", 
                        "Mean Convergence (Log Scale)",
                        "Standard Deviation Convergence (Linear Scale)",
                        "Standard Deviation Convergence (Log Scale)"
                    ),
                    shared_xaxes=True,
                    vertical_spacing=0.15,
                    horizontal_spacing=0.08
                )
                
                # --- Mean Estimate Convergence Plot (Linear Scale) ---
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=mean_estimates,
                        mode='lines',
                        name='Mean Estimate',
                        line=dict(color='#1f77b4', width=2)
                    ),
                    row=1, col=1
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=upper_bounds,
                        mode='lines',
                        name='95% CI Upper',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=lower_bounds,
                        mode='lines',
                        name='95% Confidence Interval',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.2)'
                    ),
                    row=1, col=1
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=[sample_sizes[-1]],
                        y=[mean_estimates[-1]],
                        mode='markers',
                        name='Final Mean',
                        marker=dict(color='red', size=10)
                    ),
                    row=1, col=1
                )
                
                # --- Mean Estimate Convergence Plot (Log Scale) ---
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=mean_estimates,
                        mode='lines',
                        name='Mean Estimate',
                        line=dict(color='#1f77b4', width=2),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=upper_bounds,
                        mode='lines',
                        name='95% CI Upper',
                        line=dict(width=0),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=lower_bounds,
                        mode='lines',
                        name='95% CI Lower',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=[sample_sizes[-1]],
                        y=[mean_estimates[-1]],
                        mode='markers',
                        name='Final Mean',
                        marker=dict(color='red', size=10),
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # --- Standard Deviation Convergence Plot (Linear Scale) ---
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=std_dev_estimates,
                        mode='lines',
                        name='Std Dev Estimate',
                        line=dict(color='#ff7f0e', width=2)
                    ),
                    row=2, col=1
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=[sample_sizes[-1]],
                        y=[std_dev_estimates[-1]],
                        mode='markers',
                        name='Final Std Dev',
                        marker=dict(color='red', size=10)
                    ),
                    row=2, col=1
                )
                
                # --- Standard Deviation Convergence Plot (Log Scale) ---
                fig_convergence.add_trace(
                    go.Scatter(
                        x=sample_sizes,
                        y=std_dev_estimates,
                        mode='lines',
                        name='Std Dev Estimate',
                        line=dict(color='#ff7f0e', width=2),
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                fig_convergence.add_trace(
                    go.Scatter(
                        x=[sample_sizes[-1]],
                        y=[std_dev_estimates[-1]],
                        mode='markers',
                        name='Final Std Dev',
                        marker=dict(color='red', size=10),
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                # Update layout and axes
                fig_convergence.update_layout(
                    height=800,
                    width=800,
                    title_text="Monte Carlo Convergence Analysis",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Set log scale for right column plots
                fig_convergence.update_xaxes(type="log", row=1, col=2)
                fig_convergence.update_xaxes(type="log", row=2, col=2)
                
                # Update axis labels
                fig_convergence.update_xaxes(title_text="Number of Samples", row=2, col=1)
                fig_convergence.update_xaxes(title_text="Number of Samples", row=2, col=2)
                fig_convergence.update_yaxes(title_text="Mean Estimate", row=1, col=1)
                fig_convergence.update_yaxes(title_text="Standard Deviation", row=2, col=1)
                
                # Display the figure
                st.plotly_chart(fig_convergence, use_container_width=True)
                
                # Output Distribution Analysis
                st.subheader("Output Distribution Analysis")
                
                # Create a 1x2 subplot for distribution analysis
                fig_dist = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Output Distribution Histogram", "Q-Q Plot"),
                    horizontal_spacing=0.1
                )
                
                # Histogram with KDE
                hist_data = [Y_values]
                group_labels = ['Output']
                
                # Add histogram trace
                fig_dist.add_trace(
                    go.Histogram(
                        x=Y_values,
                        histnorm='probability density',
                        name='Histogram',
                        marker_color='#1f77b4',
                        opacity=0.7,
                        nbinsx=30
                    ),
                    row=1, col=1
                )
                
                # Add KDE trace if we have a best fit distribution
                if best_distribution:
                    # Generate points for the fitted distribution
                    x_range = np.linspace(min(Y_values), max(Y_values), 1000)
                    
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
                            line=dict(color='red', width=2)
                        ),
                        row=1, col=1
                    )
                
                # Add Q-Q plot
                # Calculate theoretical quantiles from a normal distribution
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(Y_values)))
                
                # Sort the output values
                sorted_Y = np.sort(Y_values)
                
                fig_dist.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_Y,
                        mode='markers',
                        name='Q-Q Plot',
                        marker=dict(color='#1f77b4')
                    ),
                    row=1, col=2
                )
                
                # Add reference line
                min_val = min(theoretical_quantiles.min(), sorted_Y.min())
                max_val = max(theoretical_quantiles.max(), sorted_Y.max())
                
                fig_dist.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Reference Line',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=2
                )
                
                # Update layout
                fig_dist.update_layout(
                    height=500,
                    width=800,
                    title_text="Output Distribution Analysis",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update axis labels
                fig_dist.update_xaxes(title_text="Output Value", row=1, col=1)
                fig_dist.update_yaxes(title_text="Probability Density", row=1, col=1)
                fig_dist.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
                fig_dist.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
                
                # Display the figure
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Display distribution statistics
                st.subheader("Distribution Statistics")
                
                # Create a summary table for distribution statistics
                stats_df = pd.DataFrame({
                    'Statistic': [
                        'Mean', 
                        'Standard Deviation', 
                        '95% Confidence Interval', 
                        'Skewness',
                        'Kurtosis',
                        'Interquartile Range (IQR)',
                        'Best Fit Distribution'
                    ],
                    'Value': [
                        f"{mean_Y:.6f}",
                        f"{std_Y:.6f}",
                        f"[{conf_int[0]:.6f}, {conf_int[1]:.6f}]",
                        f"{skewness:.4f} ({'Positively Skewed' if skewness > 0 else 'Negatively Skewed' if skewness < 0 else 'Symmetric'})",
                        f"{kurtosis:.4f} ({'Leptokurtic (heavy-tailed)' if kurtosis > 0 else 'Platykurtic (light-tailed)' if kurtosis < 0 else 'Mesokurtic (normal-like)'})",
                        f"{iqr:.6f}",
                        f"{best_distribution if best_distribution else 'None'}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
                
                # Display distribution fitting results if available
                if not fit_df.empty:
                    st.subheader("Distribution Fitting Results")
                    
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
                    prompt = f"""
{RETURN_INSTRUCTION}

Analyze these Monte Carlo convergence results for an enterprise-grade engineering model:

```python
{model_code_str}
```

Convergence Statistics:
- Final Mean Estimate: {mean_estimates[-1]:.6f}
- Standard Deviation: {final_std_dev:.6f}
- 95% Confidence Interval: [{lower_bounds[-1]:.6f}, {upper_bounds[-1]:.6f}]
- Required Sample Size: {convergence_sample_size}
- Relative Standard Error: {(final_std_dev / mean_estimates[-1]):.4%}
- Coefficient of Variation: {(final_std_dev / np.abs(mean_estimates[-1])):.4%}

Distribution Statistics:
- Mean: {mean_Y:.6f}
- Standard Deviation: {std_Y:.6f}
- Skewness: {skewness:.4f}
- Kurtosis: {kurtosis:.4f}
- Best Fit Distribution: {best_distribution if best_distribution else "None"}

Please provide a comprehensive enterprise-grade analysis of these Monte Carlo convergence results. Your analysis should include:

1. Executive Summary
   - Key findings and their business/engineering implications
   - Assessment of convergence quality and reliability
   - Overall evaluation of the Monte Carlo simulation results

2. Technical Analysis
   - Evaluation of mean and standard deviation convergence
   - Analysis of the output distribution characteristics
   - Assessment of the required sample size for reliable results
   - Interpretation of skewness, kurtosis, and other statistical measures

3. Risk Assessment
   - Confidence in the Monte Carlo estimates
   - Potential sources of uncertainty or bias
   - Implications for decision-making based on these results

4. Recommendations
   - Guidance for using these results in engineering decisions
   - Suggestions for improving simulation efficiency or accuracy
   - Advice on sample size requirements for future simulations

Focus on actionable insights that would be valuable for executive decision-makers in an engineering context.
"""
                    
                    with st.spinner("Generating expert analysis..."):
                        if 'convergence_analysis_response' not in st.session_state:
                            response = call_groq_api(prompt, model_name=language_model)
                            st.session_state.convergence_analysis_response = response
                        else:
                            response = st.session_state.convergence_analysis_response
                        
                        st.markdown(response)
        
        # CHAT INTERFACE in the right column
        with chat_col:
            st.markdown("### Ask Questions About This Analysis")
            
            # Display a disclaimer about the prompt
            disclaimer_text = """
            **Note:** The AI assistant has been provided with the model code and the 
            convergence analysis results. You can ask questions to clarify any aspects of the analysis.
            """
            st.info(disclaimer_text)
            
            # Initialize session state for chat messages if not already done
            if "convergence_analysis_chat_messages" not in st.session_state:
                st.session_state.convergence_analysis_chat_messages = []
            
            # Create chat interface
            create_chat_interface(
                "convergence_analysis_chat",
                lambda prompt: f"""
                You are an expert assistant helping users understand Monte Carlo convergence analysis results. 
                
                Here is the model code:
                ```python
                {model_code_str}
                ```
                
                Here is the convergence analysis summary:
                - Final Mean Estimate: {mean_estimates[-1]:.6f}
                - Standard Deviation: {final_std_dev:.6f}
                - 95% Confidence Interval: [{lower_bounds[-1]:.6f}, {upper_bounds[-1]:.6f}]
                - Required Sample Size: {convergence_sample_size}
                - Relative Standard Error: {(final_std_dev / mean_estimates[-1]):.4%}
                - Coefficient of Variation: {(final_std_dev / np.abs(mean_estimates[-1])):.4%}

Distribution Statistics:
- Mean: {mean_Y:.6f}
- Standard Deviation: {std_Y:.6f}
- Skewness: {skewness:.4f}
- Kurtosis: {kurtosis:.4f}
- Best Fit Distribution: {best_distribution if best_distribution else "None"}

Here is the explanation that was previously generated:
{st.session_state.get('convergence_analysis_response', 'No analysis available yet.')}
                
                Answer the user's question: {prompt}
                
                Be concise but thorough. Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                """,
                input_placeholder="Ask a question about the convergence analysis...",
                disclaimer_text="Ask questions about the Monte Carlo convergence analysis results.",
                language_model=language_model
            )
        
        return results
        
    except Exception as e:
        if display_results:
            st.error(f"Error in expectation convergence analysis: {str(e)}")
        raise

def expectation_convergence_analysis_joint(model, problem, model_code_str, N_samples=8000, language_model='groq', display_results=True):
    """
    Analyze convergence of Monte Carlo estimation with joint analysis.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str
        String representation of the model code
    N_samples : int, optional
        Maximum number of samples for convergence analysis
    language_model : str, optional
        Language model to use for AI insights
    display_results : bool, optional
        Whether to display results using Streamlit UI (default: True)
        
    Returns
    -------
    dict
        Dictionary containing all convergence analysis results
    """
    # This function is kept for backward compatibility but now calls the main function
    return expectation_convergence_analysis(model, problem, model_code_str, N_samples, language_model, display_results)

def display_expectation_convergence_results(analysis_results, language_model='groq'):
    """
    Display expectation convergence analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing expectation convergence analysis results
    language_model : str, optional
        Language model to use for AI insights, by default 'groq'
    """
    # Extract results
    mean_convergence_fig = analysis_results.get('mean_convergence_fig')
    std_convergence_fig = analysis_results.get('std_convergence_fig')
    distribution_fig = analysis_results.get('distribution_fig')
    qq_plot = analysis_results.get('qq_plot')
    statistical_tests = analysis_results.get('statistical_tests')
    convergence_interpretation = analysis_results.get('convergence_interpretation')
    
    # Display mean convergence plot
    st.subheader("Mean Convergence")
    st.plotly_chart(mean_convergence_fig, use_container_width=True)
    
    # Display standard deviation convergence plot
    st.subheader("Standard Deviation Convergence")
    st.plotly_chart(std_convergence_fig, use_container_width=True)
    
    # Display distribution plot
    st.subheader("Output Distribution")
    st.plotly_chart(distribution_fig, use_container_width=True)
    
    # Display QQ plot
    st.subheader("Quantile-Quantile Plot")
    st.plotly_chart(qq_plot, use_container_width=True)
    
    # Display statistical tests
    st.subheader("Statistical Tests")
    st.dataframe(statistical_tests, use_container_width=True)
    
    # Display AI interpretation if available
    if convergence_interpretation:
        st.subheader("AI Insights")
        st.markdown(convergence_interpretation)