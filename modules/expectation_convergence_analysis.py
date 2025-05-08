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

def get_distribution_code_string(dist):
    """
    Generate a string representing the distribution code.
    
    Parameters
    ----------
    dist : ot.Distribution
        OpenTURNS distribution
        
    Returns
    -------
    str
        Distribution code string
    """
    # Try to get the actual OpenTURNS class name
    try:
        ot_class_name = dist.getName()
    except Exception:
        try:
            ot_class_name = dist.getClassName()
            if ot_class_name.endswith("Implementation"):
                ot_class_name = ot_class_name[:-14]  # Remove "Implementation"
        except Exception:
            ot_class_name = dist.getClassName().replace("Factory", "")
    
    # Format with full precision
    params = dist.getParameter()
    param_str = ", ".join([str(p) for p in params])
    formula_str = f"ot.{ot_class_name}({param_str})"
    
    return formula_str

def compute_expectation_convergence_analysis(model, problem, model_code_str=None, N_samples=10000):
    """Perform enterprise-grade expectation convergence analysis calculations without UI display.
    
    This function computes all the necessary data for convergence analysis, including
    mean and standard deviation convergence, distribution analysis, and statistical tests.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str, optional
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
    
    # Try to fit distributions to the output data using the distribution_fitting module
    fit_df = pd.DataFrame()
    best_distribution = None
    best_params = None
    best_distribution_name = "None"
    ot_distribution_type = "None"
    
    try:
        # Import functionality from distribution_fitting module
        from modules.distribution_fitting import get_distribution_factories, fit_distribution
        
        # Get distribution factories for continuous univariate distributions
        factories = get_distribution_factories()["Continuous Univariate"]
        
        # Prepare data for fitting - ensure we have enough samples
        Y_values_for_fitting = Y_values
        
        # If we have fewer than 1000 samples, generate more using the model
        if len(Y_values_for_fitting) < 1000:
            try:
                # Generate additional samples if needed
                additional_samples_needed = max(0, 1000 - len(Y_values_for_fitting))
                if additional_samples_needed > 0:
                    additional_sample = ot.LHSExperiment(problem, additional_samples_needed).generate()
                    additional_Y = np.array(ot_model(additional_sample))
                    if additional_Y.ndim > 1 and additional_Y.shape[1] > 1:
                        additional_Y = additional_Y[:, 0]  # Take first output if multivariate
                    Y_values_for_fitting = np.concatenate([Y_values_for_fitting, additional_Y])
            except Exception as e:
                # If additional sampling fails, continue with what we have
                st.warning(f"Could not generate additional samples for distribution fitting: {str(e)}")
        
        # Fit distributions and collect results
        results = []
        
        # Try to fit all available distributions
        for factory in factories:
            try:
                # Get factory name
                factory_name = factory.getClassName().replace("Factory", "")
                
                # Fit distribution
                fitted_dist, fitted_stats = fit_distribution(Y_values_for_fitting, factory)
                
                if fitted_dist is not None and fitted_stats:
                    # Store results
                    results.append({
                        'Distribution': factory_name,
                        'Parameters': list(fitted_dist.getParameter()),
                        'AIC': fitted_stats.get("AIC", float('inf')),
                        'BIC': fitted_stats.get("BIC", float('inf')),
                        'KS_Statistic': 1.0 - fitted_stats.get("KS p-value", 0),  # Convert p-value to statistic
                        'KS_pvalue': fitted_stats.get("KS p-value", 0),
                        'LogLikelihood': -fitted_stats.get("AIC", 0) / 2 + len(fitted_dist.getParameter()),  # Approximate
                        'OT_Distribution': fitted_dist
                    })
            except Exception as e:
                # Skip distributions that fail to fit
                continue
        
        # Create DataFrame from results
        if results:
            fit_df = pd.DataFrame(results)
            
            # Find best distribution based on AIC
            if not fit_df.empty:
                best_idx = fit_df['AIC'].idxmin()
                best_distribution_name = fit_df.loc[best_idx, 'Distribution']
                ot_distribution_type = get_distribution_code_string(fit_df.loc[best_idx, 'OT_Distribution'])
                best_distribution = fit_df.loc[best_idx, 'OT_Distribution']
                best_params = fit_df.loc[best_idx, 'Parameters']
        
    except Exception as e:
        # If fitting fails, just continue without distribution fitting
        st.warning(f"Distribution fitting failed: {str(e)}")
    
    # Create input distributions info for the prompt
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
    
    # Create mean convergence visualization
    fig_mean_convergence = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Mean Convergence (Linear Scale)</b>", 
            "<b>Mean Convergence (Log Scale)</b>"
        ),
        horizontal_spacing=0.15  # Increase spacing between subplots
    )
    
    # --- Mean Estimate Convergence Plot (Linear Scale) ---
    fig_mean_convergence.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=mean_estimates,
            mode='lines',
            name='Mean Estimate',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig_mean_convergence.add_trace(
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
    
    fig_mean_convergence.add_trace(
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
    
    fig_mean_convergence.add_trace(
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
    fig_mean_convergence.add_trace(
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
    
    fig_mean_convergence.add_trace(
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
    
    fig_mean_convergence.add_trace(
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
    
    fig_mean_convergence.add_trace(
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
    
    # Update layout and axes
    fig_mean_convergence.update_layout(
        height=500,
        width=900,  # Increase overall width
        title_text="Monte Carlo Mean Convergence Analysis",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Position below the plot
            xanchor="center",
            x=0.5    # Center horizontally
        ),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=100)  # Increase bottom margin for legend
    )
    
    # Set log scale for right column plots
    fig_mean_convergence.update_xaxes(type="log", row=1, col=2)
    
    # Update axis labels
    fig_mean_convergence.update_xaxes(title_text="Number of Samples", row=1, col=1)
    fig_mean_convergence.update_xaxes(title_text="Number of Samples (Log Scale)", row=1, col=2)
    fig_mean_convergence.update_yaxes(title_text="Mean Estimate", row=1, col=1)
    fig_mean_convergence.update_yaxes(title_text="Mean Estimate", row=1, col=2)
    
    # Create standard deviation convergence visualization
    fig_std_convergence = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "<b>Standard Deviation Convergence (Linear Scale)</b>",
            "<b>Standard Deviation Convergence (Log Scale)</b>"
        ),
        horizontal_spacing=0.15  # Increase spacing between subplots
    )
    
    # --- Standard Deviation Convergence Plot (Linear Scale) ---
    fig_std_convergence.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=std_dev_estimates,
            mode='lines',
            name='Std Dev Estimate',
            line=dict(color='#ff7f0e', width=2)
        ),
        row=1, col=1
    )
    
    fig_std_convergence.add_trace(
        go.Scatter(
            x=[sample_sizes[-1]],
            y=[std_dev_estimates[-1]],
            mode='markers',
            name='Final Std Dev',
            marker=dict(color='red', size=10)
        ),
        row=1, col=1
    )
    
    # --- Standard Deviation Convergence Plot (Log Scale) ---
    fig_std_convergence.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=std_dev_estimates,
            mode='lines',
            name='Std Dev Estimate',
            line=dict(color='#ff7f0e', width=2),
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig_std_convergence.add_trace(
        go.Scatter(
            x=[sample_sizes[-1]],
            y=[std_dev_estimates[-1]],
            mode='markers',
            name='Final Std Dev',
            marker=dict(color='red', size=10),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout and axes
    fig_std_convergence.update_layout(
        height=500,
        width=900,  # Increase overall width
        title_text="Monte Carlo Standard Deviation Convergence Analysis",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Position below the plot
            xanchor="center",
            x=0.5    # Center horizontally
        ),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=100)  # Increase bottom margin for legend
    )
    
    # Set log scale for right column plots
    fig_std_convergence.update_xaxes(type="log", row=1, col=2)
    
    # Update axis labels
    fig_std_convergence.update_xaxes(title_text="Number of Samples", row=1, col=1)
    fig_std_convergence.update_xaxes(title_text="Number of Samples (Log Scale)", row=1, col=2)
    fig_std_convergence.update_yaxes(title_text="Standard Deviation", row=1, col=1)
    fig_std_convergence.update_yaxes(title_text="Standard Deviation", row=1, col=2)
    
    # Create distribution visualization
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["Output Distribution", "Q-Q Plot"])
    
    # Add histogram trace
    fig_dist.add_trace(
        go.Histogram(
            x=Y_values,
            histnorm='probability density',
            name='Output Distribution',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add KDE trace if we have a best fit distribution
    display_name = "No distribution fitted"
    ot_distribution_type = "None"
    if best_distribution is not None:
        # Use the get_distribution_code_string function to get the proper distribution code
        ot_distribution_type = get_distribution_code_string(best_distribution)
        
        # Use the OpenTURNS distribution type for display
        display_name = ot_distribution_type
        
        # Generate points for the fitted distribution
        x_range = np.linspace(min(Y_values), max(Y_values), 1000)
        
        try:
            # Use OpenTURNS to calculate PDF values
            x_sample = ot.Sample(x_range.reshape(-1, 1))
            pdf_values = np.array([best_distribution.computePDF(ot.Point([x])) for x in x_range])
            
            fig_dist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=pdf_values,
                    mode='lines',
                    name=f'Best Fit: {display_name}',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
        except Exception as e:
            # If visualization fails, just continue without the distribution curve
            pass
    
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
        width=900,  # Increase overall width
        title_text="Output Distribution Analysis",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Position below the plot
            xanchor="center",
            x=0.5    # Center horizontally
        ),
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=100)  # Increase bottom margin for legend
    )
    
    # Update axis labels
    fig_dist.update_xaxes(title_text="Output Value", row=1, col=1)
    fig_dist.update_yaxes(title_text="Probability Density", row=1, col=1)
    fig_dist.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig_dist.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    
    # Create summary table for convergence statistics
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
    
    # Create summary table for distribution statistics
    dist_stats_df = pd.DataFrame({
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
            f"{best_distribution_name if best_distribution_name else 'None'}"
        ]
    })
    
    # Create quantile information
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_values = np.quantile(Y_values, quantiles)
    
    quantiles_df = pd.DataFrame({
        'Quantile': [f"{q*100}%" for q in quantiles],
        'Value': quantile_values
    })
    
    # Return all results
    return {
        'Y_values': Y_values,
        'sample_sizes': sample_sizes,
        'mean_estimates': mean_estimates,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'std_dev_estimates': std_dev_estimates,
        'final_mean': mean_estimates[-1],
        'final_std_dev': std_dev_estimates[-1],
        'skewness': skewness,
        'kurtosis': kurtosis,
        'convergence_sample_size': convergence_sample_size,
        'best_distribution_name': best_distribution_name,
        'ot_distribution_type': ot_distribution_type,
        'best_distribution': best_distribution,
        'best_params': best_params,
        'fit_df': fit_df,
        'inputs_df': inputs_df,
        'mean_convergence_fig': fig_mean_convergence,
        'std_convergence_fig': fig_std_convergence,
        'distribution_fig': fig_dist,
        'summary_df': summary_df,
        'dist_stats_df': dist_stats_df,
        'quantiles_df': quantiles_df,
        'display_name': display_name
    }

def expectation_convergence_analysis(model, problem, model_code_str=None, N_samples=10000, language_model='groq', display_results=True):
    """Perform enterprise-grade expectation convergence analysis.
    
    This function analyzes how the mean and standard deviation of a model output
    converge as the number of Monte Carlo samples increases. It also examines
    the output distribution characteristics and provides AI-driven insights.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str, optional
        String representation of the model code
    N_samples : int, optional
        Maximum number of samples for convergence analysis
    language_model : str, optional
        Language model to use for AI insights ('groq' or 'openai')
    display_results : bool, optional
        Whether to display results in the Streamlit interface
        
    Returns
    -------
    dict
        Dictionary containing all convergence analysis results
    """
    try:
        # Compute the expectation convergence analysis
        analysis_results = compute_expectation_convergence_analysis(model, problem, model_code_str, N_samples)
        
        # Generate AI insights
        if 'ai_insights' not in analysis_results or not analysis_results['ai_insights']:
            with st.spinner("Generating AI insights..."):
                insights = generate_ai_insights(analysis_results, language_model)
                analysis_results['ai_insights'] = insights
        
        # Display results if requested
        if display_results:
            display_expectation_convergence_analysis(analysis_results, language_model)
        
        return analysis_results
    
    except Exception as e:
        st.error(f"Error in expectation convergence analysis: {str(e)}")
        return None

def expectation_convergence_analysis_joint(model, problem, model_code_str=None, N_samples=10000, language_model='groq', display_results=True):
    """Perform enterprise-grade expectation convergence analysis for joint outputs.
    
    This function handles models with multiple outputs, running separate convergence
    analyses for each output dimension and combining the results.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str, optional
        String representation of the model code
    N_samples : int, optional
        Maximum number of samples for convergence analysis
    language_model : str, optional
        Language model to use for AI insights ('groq' or 'openai')
    display_results : bool, optional
        Whether to display results in the Streamlit interface
        
    Returns
    -------
    dict
        Dictionary containing all convergence analysis results for each output dimension
    """
    try:
        # Create a wrapper function for each output dimension
        ot_model = get_ot_model(model)
        
        # Get the output dimension
        input_random_vector = ot.RandomVector(problem)
        output_random_vector = ot.CompositeRandomVector(ot_model, input_random_vector)
        output_dimension = output_random_vector.getDimension()
        
        # If output is scalar, just use the regular analysis
        if output_dimension == 1:
            return expectation_convergence_analysis(model, problem, model_code_str, N_samples, language_model, display_results)
        
        # For multidimensional outputs, create a separate analysis for each dimension
        all_results = {}
        
        for i in range(output_dimension):
            # Create a wrapper function that extracts the i-th output
            def wrapper_model_i(X):
                Y = model(X)
                if isinstance(Y, (list, tuple, np.ndarray)):
                    return Y[i]
                else:
                    return Y
            
            # Add dimension info to the model code
            dimension_model_code = f"""# Dimension {i+1} of the original model
{model_code_str}

# Extracting dimension {i+1}
def dimension_{i+1}_model(X):
    Y = model(X)
    return Y[{i}]
"""
            
            # Run analysis for this dimension
            st.markdown(f"### Analyzing Output Dimension {i+1}")
            
            # Compute results for this dimension
            results_i = compute_expectation_convergence_analysis(wrapper_model_i, problem, dimension_model_code, N_samples)
            
            # Generate AI insights for this dimension
            if language_model:
                with st.spinner(f"Generating insights for dimension {i+1}..."):
                    insights = generate_ai_insights(results_i, language_model)
                    results_i['ai_insights'] = insights
            
            # Display results for this dimension if requested
            if display_results:
                st.markdown(f"## Results for Output Dimension {i+1}")
                display_expectation_convergence_analysis(results_i, language_model)
            
            # Store results for this dimension
            all_results[f"dimension_{i+1}"] = results_i
        
        return all_results
    
    except Exception as e:
        st.error(f"Error in joint expectation convergence analysis: {str(e)}")
        return None

def display_expectation_convergence_results(analysis_results, language_model='groq'):
    """
    Display enterprise-grade expectation convergence analysis results.
    This function creates and displays interactive visualizations and insights for the expectation convergence analysis results.
    """
    if not analysis_results:
        st.error("No analysis results to display. Please run the analysis first.")
        return

    # Display convergence analysis results
    with st.container():
        st.markdown("## Expectation Convergence Analysis Results")
        st.markdown("""
        This analysis examines how the mean and standard deviation estimates converge as the number of Monte Carlo samples increases.
        The convergence plots help determine if enough samples have been used for reliable estimates.
        """)

        # Mean convergence visualization
        if 'mean_convergence_fig' in analysis_results:
            st.write("##### Mean Convergence Analysis")
            st.plotly_chart(analysis_results['mean_convergence_fig'], use_container_width=True)

        # Standard deviation convergence visualization
        if 'std_convergence_fig' in analysis_results:
            st.write("##### Standard Deviation Convergence Analysis")
            st.plotly_chart(analysis_results['std_convergence_fig'], use_container_width=True)

        # Output distribution visualization
        if 'distribution_fig' in analysis_results:
            st.write("##### Output Distribution Analysis")
            st.plotly_chart(analysis_results['distribution_fig'], use_container_width=True)

        # Distribution fitting results
        if 'fit_df' in analysis_results and not analysis_results['fit_df'].empty:
            st.write("##### Distribution Fitting Results")
            try:
                display_df = analysis_results['fit_df'][['Distribution', 'AIC', 'BIC', 'KS_Statistic', 'KS_pvalue']].copy()
                display_df['AIC'] = display_df['AIC'].map('{:.2f}'.format)
                display_df['BIC'] = display_df['BIC'].map('{:.2f}'.format)
                display_df['KS_Statistic'] = display_df['KS_Statistic'].map('{:.4f}'.format)
                display_df['KS_pvalue'] = display_df['KS_pvalue'].map('{:.4f}'.format)
                display_df = display_df.rename(columns={
                    'KS_Statistic': 'KS Statistic',
                    'KS_pvalue': 'KS p-value'
                })
                display_df = display_df.sort_values('AIC')
                display_df = display_df.head(1)
                if 'display_name' in analysis_results and analysis_results['display_name']:
                    st.markdown(f"**Best Fit: {analysis_results['display_name']}**")
                st.dataframe(display_df, hide_index=True, use_container_width=True)
                if 'best_distribution_name' in analysis_results and analysis_results['best_distribution_name'] != "None":
                    ot_dist_type = analysis_results.get('ot_distribution_type', analysis_results['best_distribution_name'])
                    if 'best_params' in analysis_results and analysis_results['best_params']:
                        param_str = ", ".join([f"{p:.4f}" for p in analysis_results['best_params']])
            except Exception as e:
                st.warning(f"Error formatting distribution fitting results: {str(e)}")
                st.dataframe(analysis_results['fit_df'], use_container_width=True)
        else:
            st.info("No distribution fitting results available. This may happen if none of the standard distributions provided a good fit to the data.")

        # Two columns for statistics tables
        col1, col2 = st.columns(2)
        if 'summary_df' in analysis_results:
            with col1:
                st.write("##### Convergence Statistics")
                st.dataframe(analysis_results['summary_df'], hide_index=True, use_container_width=True)
        if 'dist_stats_df' in analysis_results:
            with col2:
                st.write("##### Distribution Statistics")
                st.dataframe(analysis_results['dist_stats_df'], hide_index=True, use_container_width=True)

        # Quantiles
        if 'quantiles_df' in analysis_results:
            st.write("##### Output Quantiles")
            st.dataframe(analysis_results['quantiles_df'], hide_index=True, use_container_width=True)

        # AI insights
        if 'ai_insights' in analysis_results:
            st.write("##### AI-Generated Insights")
            st.markdown(analysis_results['ai_insights'])

        # Display any other available results
        for key, value in analysis_results.items():
            if key not in ['mean_convergence_fig', 'std_convergence_fig', 'distribution_fig', 
                        'summary_df', 'dist_stats_df', 'quantiles_df', 'fit_df', 'ai_insights',
                        'inputs_df', 'Y_values', 'sample_sizes', 'mean_estimates', 'lower_bounds', 
                        'upper_bounds', 'std_dev_estimates', 'final_std_dev', 'convergence_sample_size', 
                        'mean_Y', 'std_Y', 'conf_int', 'skewness', 'kurtosis', 'q1', 'q3', 'iqr', 
                        'best_distribution', 'best_params', 'best_distribution_name', 'input_parameters',
                        'ot_distribution_type', 'display_name']:
                if hasattr(value, 'to_html'):
                    st.write(f"##### {key.replace('_', ' ').title()}")
                    st.dataframe(value, use_container_width=True)
                elif isinstance(value, (str, int, float)):
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def get_expectation_convergence_context_for_chat(exp_results):
    """
    Generate a formatted string containing expectation convergence analysis results for the global chat context.
    
    Parameters
    ----------
    exp_results : dict
        Dictionary containing the results of the expectation convergence analysis
        
    Returns
    -------
    str
        Formatted string with expectation convergence analysis results for chat context
    """
    context = ""
    
    # Main statistics
    mean_Y = exp_results.get("mean_Y")
    std_Y = exp_results.get("std_Y")
    conf_int = exp_results.get("conf_int")
    skewness = exp_results.get("skewness")
    kurtosis = exp_results.get("kurtosis")
    quantiles = exp_results.get("quantiles") if "quantiles" in exp_results else None
    best_distribution = exp_results.get("best_distribution_name")
    best_params = exp_results.get("best_params")
    prob_exceedance = exp_results.get("prob_exceedance") if "prob_exceedance" in exp_results else None
    ai_insights = exp_results.get("ai_insights")
    
    context += "\n\n### Expectation Convergence Analysis Results\n"
    if mean_Y is not None and std_Y is not None:
        context += f"- **Estimated Mean Output:** {mean_Y:.4f}\n"
        context += f"- **Estimated Std Dev:** {std_Y:.4f}\n"
    if conf_int is not None:
        context += f"- **95% Confidence Interval:** [{conf_int[0]:.4f}, {conf_int[1]:.4f}]\n"
    if skewness is not None:
        context += f"- **Skewness:** {skewness:.4f}\n"
    if kurtosis is not None:
        context += f"- **Kurtosis:** {kurtosis:.4f}\n"
    if quantiles is not None:
        context += "- **Quantiles:**\n"
        for q, v in quantiles.items():
            context += f"    - {q}: {v:.4f}\n"
    if best_distribution is not None:
        context += f"- **Best Fit Distribution:** {best_distribution}\n"
    if best_params is not None:
        context += f"- **Best Fit Parameters:** {best_params}\n"
    if prob_exceedance is not None:
        context += f"- **Probability of Exceeding Threshold:** {prob_exceedance}\n"
    if ai_insights is not None:
        context += f"\n#### AI Insights\n{ai_insights}\n"
    
    # Add output distribution table if available
    if "fit_df" in exp_results and exp_results["fit_df"] is not None:
        fit_df = exp_results["fit_df"]
        try:
            context += "\n**Distribution Fit Results:**\n"
            context += fit_df.to_markdown(index=False)
        except Exception:
            pass
    
    return context

def generate_ai_insights(analysis_results, language_model='groq'):
    """Generate AI-powered insights for expectation convergence analysis.
    
    This function uses a language model to generate insights about the
    convergence analysis results, focusing on statistical significance,
    distribution characteristics, and recommendations.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing all convergence analysis results
    language_model : str, optional
        Language model to use for insights ('groq' or 'openai')
        
    Returns
    -------
    str
        AI-generated insights in markdown format
    """
    try:
        # Extract key information
        mean_estimate = analysis_results.get('final_mean', 'N/A')
        std_dev = analysis_results.get('final_std_dev', 'N/A')
        convergence_sample_size = analysis_results.get('convergence_sample_size', 'N/A')
        skewness = analysis_results.get('skewness', 'N/A')
        kurtosis = analysis_results.get('kurtosis', 'N/A')
        
        # Get distribution information if available
        best_distribution_name = analysis_results.get('best_distribution_name', 'None')
        best_params = analysis_results.get('best_params', [])
        
        # Get the detailed distribution name if available
        display_name = analysis_results.get('display_name', best_distribution_name)
        
        # Analyze Q-Q plot data if available
        qq_plot_description = "Not available"
        try:
            if 'Y_values' in analysis_results:
                Y_values = analysis_results['Y_values']
                # Calculate theoretical quantiles from a normal distribution
                from scipy import stats
                import numpy as np
                
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(Y_values)))
                sorted_Y = np.sort(Y_values)
                
                # Calculate correlation between theoretical and empirical quantiles
                qq_correlation = np.corrcoef(theoretical_quantiles, sorted_Y)[0, 1]
                
                # Analyze deviations from the reference line
                # Compare first quarter, middle, and last quarter of points
                n = len(sorted_Y)
                first_quarter = int(n * 0.25)
                third_quarter = int(n * 0.75)
                
                # Calculate average deviation in each section
                lower_deviation = np.mean(sorted_Y[:first_quarter] - stats.norm.ppf(np.linspace(0.01, 0.25, first_quarter)))
                middle_deviation = np.mean(sorted_Y[first_quarter:third_quarter] - stats.norm.ppf(np.linspace(0.25, 0.75, third_quarter-first_quarter)))
                upper_deviation = np.mean(sorted_Y[third_quarter:] - stats.norm.ppf(np.linspace(0.75, 0.99, n-third_quarter)))
                
                # Create a description of the Q-Q plot
                qq_plot_description = f"""
                Q-Q Plot Analysis:
                - Correlation between theoretical and empirical quantiles: {qq_correlation:.4f} (1.0 would be perfect normal)
                - Lower tail (bottom 25% of points): {'Above' if lower_deviation > 0 else 'Below'} the reference line by average of {abs(lower_deviation):.4f}
                - Middle section (middle 50% of points): {'Above' if middle_deviation > 0 else 'Below'} the reference line by average of {abs(middle_deviation):.4f}
                - Upper tail (top 25% of points): {'Above' if upper_deviation > 0 else 'Below'} the reference line by average of {abs(upper_deviation):.4f}
                """
                
                # Add interpretation hints
                if lower_deviation < 0 and upper_deviation > 0:
                    qq_plot_description += "- Pattern suggests: S-shaped curve (right-skewed distribution)"
                elif lower_deviation > 0 and upper_deviation < 0:
                    qq_plot_description += "- Pattern suggests: Reverse S-shaped curve (left-skewed distribution)"
                elif lower_deviation > 0 and upper_deviation > 0:
                    qq_plot_description += "- Pattern suggests: Points generally above the line (heavier tails than normal)"
                elif lower_deviation < 0 and upper_deviation < 0:
                    qq_plot_description += "- Pattern suggests: Points generally below the line (lighter tails than normal)"
                else:
                    qq_plot_description += "- Pattern suggests: Close to normal distribution"
        except Exception as e:
            qq_plot_description = f"Q-Q Plot Analysis: Error analyzing Q-Q plot data: {str(e)}"
        
        # Create a prompt for the AI
        prompt = f"""
You are an expert in uncertainty quantification and statistical analysis. Based on the following Monte Carlo convergence analysis results, provide insights about the model's behavior, convergence characteristics, and output distribution.

Key results:
- Mean estimate: {mean_estimate}
- Standard deviation: {std_dev}
- Sample size for convergence: {convergence_sample_size}
- Skewness: {skewness} (positive means right-skewed, negative means left-skewed)
- Kurtosis: {kurtosis} (positive means heavier tails than normal, negative means lighter tails)
- Best-fit distribution: {display_name}

{qq_plot_description}

Please provide insights on:
1. The convergence behavior of the model (how quickly it converges, whether more samples might be needed)
2. The uncertainty in the model output (what the standard deviation tells us about the model's variability)
3. The distribution of the model output (what the shape of the distribution means in practical terms)
4. Q-Q plot interpretation: Based on the Q-Q plot data provided, explain what this tells us about the distribution compared to a normal distribution
5. What the best-fit distribution ({display_name}) tells us about the underlying process
6. How this distribution could be used for risk assessment or decision-making
7. Any recommendations for further analysis or model improvement

Keep your response concise but informative, focusing on practical implications rather than theoretical details.
"""
        
        # Use the call_groq_api utility function
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
                    
                    Please check the analysis results manually:
                    - Mean estimate: {mean_estimate}
                    - Standard deviation: {std_dev}
                    - Distribution shape: {'Positively Skewed' if analysis_results['skewness'] > 0 else 'Negatively Skewed' if analysis_results['skewness'] < 0 else 'Symmetric'}
                    """
                import time
                time.sleep(2)  # Wait before retrying
        
        return insights
        
    except Exception as e:
        st.error(f"Error generating AI insights: {str(e)}")
        return f"""
        ## Error Generating Insights
        
        There was an error preparing the analysis for AI insights: {str(e)}
        
        Please examine the analysis results manually.
        """