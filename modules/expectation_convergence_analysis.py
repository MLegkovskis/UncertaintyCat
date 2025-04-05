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
    
    # RESULTS SECTION    
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
        
        # Update x-axis to log scale for the second column
        fig_convergence.update_xaxes(type='log', row=1, col=2)
        fig_convergence.update_xaxes(type='log', row=2, col=2)
        
        # Update layout
        fig_convergence.update_layout(
            height=800,  # Increased height for 2x2 layout
            width=1000,
            title_text="Monte Carlo Convergence Analysis",
            title_font=dict(size=16),
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
        fig_convergence.update_xaxes(title_text="Sample Size", row=2, col=1)
        fig_convergence.update_yaxes(title_text="Mean Estimate", row=1, col=1)
        fig_convergence.update_yaxes(title_text="Std Dev Estimate", row=2, col=1)
        
        fig_convergence.update_xaxes(title_text="Sample Size (Log Scale)", row=2, col=2)
        fig_convergence.update_yaxes(title_text="Mean Estimate", row=1, col=2)
        fig_convergence.update_yaxes(title_text="Std Dev Estimate", row=2, col=2)
        
        st.plotly_chart(fig_convergence, use_container_width=True)
        
        # Add explanation of the plots
        st.markdown("""
        **Interpretation Guide:**
        - **Linear Scale Plots**: Show the raw convergence behavior of the estimates
        - **Log Scale Plots**: Highlight early convergence behavior and are useful for identifying the minimum required sample size
        - The **95% confidence interval** for the mean represents the range where we are 95% confident the true mean lies
        - Stable standard deviation estimates indicate reliable uncertainty quantification
        """)
        
        # Output Distribution Analysis
        st.subheader("Output Distribution Analysis")
        
        # Add note about convergence-based analysis
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
        <strong>Note:</strong> This distribution analysis is performed using {convergence_sample_size} samples, which corresponds to the converged sample size from the analysis above. This ensures that the distribution characteristics are representative of the true underlying distribution.
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown("#### Quantile Information")
        quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        quantile_values = np.quantile(Y_values, quantiles)
        
        quantiles_df = pd.DataFrame({
            'Quantile': [f"{q*100}%" for q in quantiles],
            'Value': quantile_values
        })
        st.dataframe(quantiles_df, use_container_width=True)
        
        # Create enhanced distribution visualization
        st.markdown("#### Distribution Visualization")
        
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
        st.markdown("#### Distribution Shape Interpretation")
        
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
        
        st.markdown(f"**Skewness**: {skew_interpretation}")
        st.markdown(f"**Kurtosis**: {kurt_interpretation}")
    
    # AI INSIGHTS SECTION
    if language_model:        
        with st.expander("AI Insights", expanded=True):
            # Prepare convergence summary for the prompt
            convergence_summary = f"""
### Convergence Analysis Results

#### Basic Statistics
- Final Mean Estimate: {mean_estimates[-1]:.6f}
- Standard Deviation: {final_std_dev:.6f}
- 95% Confidence Interval: [{lower_bounds[-1]:.6f}, {upper_bounds[-1]:.6f}]
- Required Sample Size: {convergence_sample_size}

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
   - Discuss the implications of the 95% confidence interval for the mean ({lower_bounds[-1]:.6f} to {upper_bounds[-1]:.6f}), emphasizing that we are 95% confident that the true mean output lies within this range

2. Distribution Analysis
   - Interpret the shape characteristics (skewness, kurtosis)
   - Analyze the spread and central tendency
   - Evaluate the normality assumption and its implications
   - Note that this distribution analysis is performed at the sample size corresponding to convergence ({convergence_sample_size} samples)

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
            
            with st.spinner("Generating expert analysis..."):
                response_key = 'convergence_analysis_response_markdown'
                
                if response_key not in st.session_state:
                    response_markdown = call_groq_api(prompt, model_name=language_model)
                    st.session_state[response_key] = response_markdown
                else:
                    response_markdown = st.session_state[response_key]
                
                st.markdown(response_markdown)
            
            # Initialize session state for chat
            if "convergence_analysis_chat_messages" not in st.session_state:
                st.session_state.convergence_analysis_chat_messages = []
            
            # Add chat interface for convergence analysis
            st.write("### Ask Questions About This Analysis")
            
            # Display a disclaimer about the prompt
            disclaimer_text = """
            **Note:** The AI assistant has been provided with the model code, input distributions, 
            and the convergence analysis results above. You can ask questions to clarify any aspects of the analysis.
            """
            
            # Define context generator function
            def generate_context(prompt):
                return f"""
                You are an expert assistant helping users understand convergence analysis results. 
                
                Here is the model code:
                ```python
                {model_code_str}
                ```
                
                Here is information about the input distributions:
                {inputs_df.to_markdown(index=False)}
                
                Here is the convergence analysis summary:
                {convergence_summary}
                
                Here is the explanation that was previously generated:
                {response_markdown}
                
                Answer the user's question based on this information. Be concise but thorough.
                If you're not sure about something, acknowledge the limitations of your knowledge.
                Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                """
            
            # Create the chat interface
            create_chat_interface(
                session_key="convergence_analysis",
                context_generator=generate_context,
                input_placeholder="Ask a question about the convergence analysis...",
                disclaimer_text=disclaimer_text,
                language_model=language_model
            )

def expectation_convergence_analysis_joint(model, problem, model_code_str, N_samples=8000, language_model='groq'):
    """Analyze convergence of Monte Carlo estimation with joint analysis."""
    # This function is kept for backward compatibility but now calls the main function
    expectation_convergence_analysis(model, problem, model_code_str, N_samples, language_model)