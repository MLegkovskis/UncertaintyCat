import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
from utils.core_utils import call_groq_api
from utils.markdown_utils import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def expectation_convergence_analysis(model, problem, model_code_str, N_samples=8000, language_model='groq'):
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
    
    # Generate samples for distribution analysis
    sample_size = 5000
    input_sample = distribution.getSample(sample_size)
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
    
    # Create Plotly figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        subplot_titles=(
            "Mean Convergence (Linear Scale)", 
            "Mean Convergence (Log Scale)",
            "Output Distribution Analysis"
        ),
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # --- Mean Estimate Convergence Plot (Linear Scale) ---
    fig.add_trace(
        go.Scatter(
            x=sample_sizes,
            y=mean_estimates,
            mode='lines',
            name='Mean Estimate',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
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
    
    fig.add_trace(
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
    
    fig.add_trace(
        go.Scatter(
            x=[sample_sizes[-1]],
            y=[mean_estimates[-1]],
            mode='markers',
            name='Final Estimate',
            marker=dict(color='red', size=10)
        ),
        row=1, col=1
    )
    
    # --- Mean Estimate Convergence Plot (Log Scale) ---
    fig.add_trace(
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
    
    fig.add_trace(
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
    
    fig.add_trace(
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
    
    fig.add_trace(
        go.Scatter(
            x=[sample_sizes[-1]],
            y=[mean_estimates[-1]],
            mode='markers',
            name='Final Estimate',
            marker=dict(color='red', size=10),
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update x-axis to log scale for the second plot
    fig.update_xaxes(type='log', row=1, col=2)
    
    # --- Distribution Analysis ---
    # Create histogram with KDE
    hist_data = [Y_values]
    group_labels = ['Output Values']
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=Y_values,
            nbinsx=50,
            name='Histogram',
            marker_color='rgba(44, 160, 44, 0.7)',
            marker_line=dict(color='black', width=1),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add vertical lines for statistics
    fig.add_vline(
        x=mean_Y,
        line=dict(color='blue', width=2, dash='dash'),
        row=2, col=1
    )
    
    fig.add_vline(
        x=conf_int[0],
        line=dict(color='purple', width=2, dash='dash'),
        row=2, col=1
    )
    
    fig.add_vline(
        x=conf_int[1],
        line=dict(color='purple', width=2, dash='dash'),
        row=2, col=1
    )
    
    # Add annotations for statistics
    fig.add_annotation(
        x=mean_Y,
        y=1.05,
        text=f"Mean: {mean_Y:.4f}",
        showarrow=False,
        font=dict(color='blue'),
        xref="x3",
        yref="paper"
    )
    
    fig.add_annotation(
        x=conf_int[0],
        y=0.95,
        text=f"95% CI Lower: {conf_int[0]:.4f}",
        showarrow=False,
        font=dict(color='purple'),
        xref="x3",
        yref="paper"
    )
    
    fig.add_annotation(
        x=conf_int[1],
        y=0.95,
        text=f"95% CI Upper: {conf_int[1]:.4f}",
        showarrow=False,
        font=dict(color='purple'),
        xref="x3",
        yref="paper"
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Expectation Convergence Analysis",
        title_font=dict(size=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Sample Size", row=1, col=1)
    fig.update_yaxes(title_text="Mean Estimate", row=1, col=1)
    
    fig.update_xaxes(title_text="Sample Size (Log Scale)", row=1, col=2)
    fig.update_yaxes(title_text="Mean Estimate", row=1, col=2)
    
    fig.update_xaxes(title_text="Output Value", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    # Prepare convergence summary
    convergence_summary = f"""
### Convergence Analysis Results

#### Basic Statistics
- Final Mean Estimate: {mean_estimates[-1]:.6f}
- Standard Deviation: {final_std_dev:.6f}
- 95% Confidence Interval: [{lower_bounds[-1]:.6f}, {upper_bounds[-1]:.6f}]
- Required Sample Size: {int(initial_sample_size)}

#### Distribution Characteristics
- Skewness: {skewness:.4f}
- Kurtosis: {kurtosis:.4f}
- Interquartile Range: {iqr:.4f}
- 95% Prediction Interval: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]

#### Convergence Metrics
- Relative Standard Error: {(final_std_dev / mean_estimates[-1]):.4%}
- Coefficient of Variation: {(final_std_dev / np.abs(mean_estimates[-1])):.4%}
"""

    # Prepare the prompt for AI insights
    prompt = f"""
{RETURN_INSTRUCTION}

Given the following user-defined model defined in Python code:

```python
{model_code_str}
```

and the following uncertain input distributions:

{inputs_df.to_markdown(index=False)}

Given the following comprehensive convergence analysis results:

{convergence_summary}

Please provide a detailed enterprise-grade analysis that includes:

1. Statistical Significance
   - Evaluate the convergence behavior and its statistical significance
   - Assess the reliability of the Monte Carlo estimates
   - Discuss the implications of the confidence intervals

2. Distribution Analysis
   - Interpret the shape characteristics (skewness, kurtosis)
   - Analyze the spread and central tendency
   - Evaluate the normality assumption and its implications

3. Risk Assessment
   - Identify key risk factors based on the distribution tails
   - Assess the robustness of the model predictions
   - Provide recommendations for risk mitigation

4. Operational Insights
   - Suggest optimal sample sizes for different precision requirements
   - Discuss the trade-off between computational cost and accuracy
   - Provide guidelines for monitoring and quality control

5. Recommendations
   - Suggest potential model improvements
   - Propose additional analyses that could enhance understanding
   - Provide specific action items for stakeholders

Format your response with clear section headings and bullet points. Focus on actionable insights and quantitative recommendations where possible.
"""

    # Display results in expandable sections
    with st.expander("Convergence Statistics", expanded=True):
        st.write("### Key Convergence Metrics")
        st.markdown(convergence_summary)
        
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
                f"{int(initial_sample_size)}",
                f"{(final_std_dev / mean_estimates[-1]):.4%}",
                f"{(final_std_dev / np.abs(mean_estimates[-1])):.4%}"
            ]
        })
        st.dataframe(summary_df, use_container_width=True)
    
    with st.expander("Convergence Visualization", expanded=True):
        st.write("### Monte Carlo Convergence Analysis")
        st.write("This visualization shows how the mean estimate converges as the sample size increases, along with the output distribution.")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation of the plots
        st.write("""
        **Interpretation Guide:**
        - **Linear Scale Plot**: Shows the raw convergence behavior of the mean estimate
        - **Log Scale Plot**: Highlights early convergence behavior and is useful for identifying the minimum required sample size
        - **Distribution Analysis**: Shows the shape and spread of the output distribution
        """)
    
    with st.expander("Distribution Analysis", expanded=True):
        st.write("### Output Distribution Characteristics")
        
        # Create a distribution statistics table
        dist_stats_df = pd.DataFrame({
            'Statistic': [
                'Mean', 
                'Standard Deviation', 
                'Skewness', 
                'Kurtosis',
                'Interquartile Range',
                '95% Prediction Interval'
            ],
            'Value': [
                f"{mean_Y:.6f}",
                f"{std_Y:.6f}",
                f"{skewness:.4f}",
                f"{kurtosis:.4f}",
                f"{iqr:.4f}",
                f"{conf_int[0]:.4f}, {conf_int[1]:.4f}"
            ]
        })
        st.dataframe(dist_stats_df, use_container_width=True)
        
        # Add quantile information
        st.write("#### Quantile Information")
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantile_values = np.quantile(Y_values, quantiles)
        
        quantiles_df = pd.DataFrame({
            'Quantile': [f"{q*100}%" for q in quantiles],
            'Value': quantile_values
        })
        st.dataframe(quantiles_df, use_container_width=True)
        
        # Create enhanced distribution visualization
        st.write("#### Distribution Visualization")
        
        # Create a figure with subplots for box plot and distribution
        fig_dist = make_subplots(
            rows=1, 
            cols=2,
            subplot_titles=["Box Plot of Output", "Distribution of Output"],
            specs=[[{"type": "box"}, {"type": "histogram"}]],
            horizontal_spacing=0.1
        )
        
        # Add box plot
        fig_dist.add_trace(
            go.Box(
                y=Y_values,
                name="Output",
                boxmean=True,
                marker_color='royalblue',
                boxpoints='outliers'
            ),
            row=1, col=1
        )
        
        # Add histogram with KDE
        fig_dist.add_trace(
            go.Histogram(
                x=Y_values,
                histnorm='probability density',
                name="Output",
                marker=dict(color='royalblue', opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Add KDE
        kde_x = np.linspace(min(Y_values), max(Y_values), 1000)
        kde = stats.gaussian_kde(Y_values)
        kde_y = kde(kde_x)
        
        fig_dist.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='KDE',
                line=dict(color='firebrick', width=2)
            ),
            row=1, col=2
        )
        
        # Add reference lines for mean and median
        mean_val = np.mean(Y_values)
        median_val = np.median(Y_values)
        
        # Add mean line to histogram
        fig_dist.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right",
            row=1, col=2
        )
        
        # Add median line to histogram
        fig_dist.add_vline(
            x=median_val,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top left",
            row=1, col=2
        )
        
        # Update layout
        fig_dist.update_layout(
            height=500,
            showlegend=False,
            title_text="Output Distribution Analysis",
            margin=dict(l=60, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Add interpretation of skewness and kurtosis
        st.write("#### Distribution Shape Interpretation")
        
        if abs(skewness) < 0.5:
            skew_interpretation = "The distribution is approximately symmetric."
        elif skewness < -0.5:
            skew_interpretation = "The distribution is negatively skewed (left-tailed), with more extreme values on the left."
        else:
            skew_interpretation = "The distribution is positively skewed (right-tailed), with more extreme values on the right."
            
        if abs(kurtosis) < 0.5:
            kurt_interpretation = "The distribution has a similar tail weight to a normal distribution."
        elif kurtosis < -0.5:
            kurt_interpretation = "The distribution has lighter tails than a normal distribution (platykurtic)."
        else:
            kurt_interpretation = "The distribution has heavier tails than a normal distribution (leptokurtic), indicating more extreme values."
        
        st.write(f"**Skewness**: {skew_interpretation}")
        st.write(f"**Kurtosis**: {kurt_interpretation}")
    
    with st.expander("AI-Generated Insights", expanded=True):
        st.write("### Expert Analysis")
        with st.spinner("Generating expert analysis..."):
            response_markdown = call_groq_api(prompt, model_name=language_model)
            st.markdown(response_markdown)

def expectation_convergence_analysis_joint(model, problem, model_code_str, N_samples=8000, language_model='groq'):
    """Analyze convergence of Monte Carlo estimation with joint analysis."""
    # This function is kept for backward compatibility but now calls the main function
    expectation_convergence_analysis(model, problem, model_code_str, N_samples, language_model)