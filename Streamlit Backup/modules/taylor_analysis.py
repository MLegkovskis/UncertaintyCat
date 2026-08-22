# modules/taylor_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model

def compute_taylor_indices(model, problem, h=1e-6):
    """Compute Taylor-based sensitivity indices."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")

    # Get input names and dimension
    dimension = problem.getDimension()
    input_names = []
    nominal_point = np.zeros(dimension)
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        name = marginal.getDescription()[0]
        input_names.append(name if name != "" else f"X{i+1}")
        nominal_point[i] = marginal.getMean()[0]

    # Convert to OpenTURNS point for model evaluation
    ot_nominal_point = ot.Point(nominal_point)
    
    # Compute nominal value - handle both scalar and vector outputs
    nominal_value_raw = model(ot_nominal_point)
    
    # Check if the output is a scalar or vector
    if isinstance(nominal_value_raw, ot.Point) and nominal_value_raw.getDimension() > 1:
        # For this analysis, we'll focus on the first output component
        nominal_value = nominal_value_raw[0]
        is_vector_output = True
    else:
        # Handle the case where it's a scalar or a Point with dimension 1
        if isinstance(nominal_value_raw, ot.Point):
            nominal_value = nominal_value_raw[0]
        else:
            nominal_value = float(nominal_value_raw)
        is_vector_output = False

    # Compute gradients
    gradients = np.zeros(dimension)
    for i in range(dimension):
        perturbed_point = nominal_point.copy()
        perturbed_point[i] += h
        
        # Convert to OpenTURNS point for model evaluation
        ot_perturbed_point = ot.Point(perturbed_point)
        perturbed_value_raw = model(ot_perturbed_point)
        
        # Handle vector outputs by taking the first component
        if is_vector_output:
            perturbed_value = perturbed_value_raw[0]
        else:
            if isinstance(perturbed_value_raw, ot.Point):
                perturbed_value = perturbed_value_raw[0]
            else:
                perturbed_value = float(perturbed_value_raw)
            
        gradients[i] = (perturbed_value - nominal_value) / h

    # Compute variances
    variances = np.array([problem.getMarginal(i).getStandardDeviation()[0]**2 for i in range(dimension)])

    # Compute sensitivity indices
    total_variance = np.sum(gradients**2 * variances)
    sensitivity_indices = (gradients**2 * variances) / total_variance if total_variance > 0 else np.zeros(dimension)

    # Create results dictionary
    results = {
        'nominal_point': nominal_point,
        'nominal_value': nominal_value,
        'gradients': gradients,
        'variances': variances,
        'sensitivity_indices': sensitivity_indices,
        'input_names': input_names,
        'is_vector_output': is_vector_output
    }

    return results

def validate_taylor_surrogate(model, problem, n_validation=100, random_seed=42):
    """
    Validate the Taylor expansion as a linear surrogate model.
    
    Parameters
    ----------
    model : ot.Function
        The original model to validate against
    problem : ot.Distribution
        The input distribution
    n_validation : int
        Number of validation points
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing validation metrics
    """
    # Set random seed for reproducibility
    ot.RandomGenerator.SetSeed(random_seed)
    
    # Get dimension and nominal point
    dimension = problem.getDimension()
    nominal_point = np.array([problem.getMarginal(i).getMean()[0] for i in range(dimension)])
    
    # Convert nominal_point to OpenTURNS point for model evaluation
    ot_nominal_point = ot.Point(nominal_point)
    
    # Compute nominal value - handle both scalar and vector outputs
    nominal_value_raw = model(ot_nominal_point)
    
    # Check if the output is a scalar or vector
    if isinstance(nominal_value_raw, ot.Point) and nominal_value_raw.getDimension() > 1:
        # For this analysis, we'll focus on the first output component
        nominal_value = nominal_value_raw[0]
        is_vector_output = True
    else:
        # Handle the case where it's a scalar or a Point with dimension 1
        if isinstance(nominal_value_raw, ot.Point):
            nominal_value = nominal_value_raw[0]
        else:
            nominal_value = float(nominal_value_raw)
        is_vector_output = False
    
    # Compute gradients for Taylor expansion
    h = 1e-6
    gradients = np.zeros(dimension)
    for i in range(dimension):
        perturbed_point = nominal_point.copy()
        perturbed_point[i] += h
        
        # Convert to OpenTURNS point for model evaluation
        ot_perturbed_point = ot.Point(perturbed_point)
        perturbed_value_raw = model(ot_perturbed_point)
        
        # Handle vector outputs by taking the first component
        if is_vector_output:
            perturbed_value = perturbed_value_raw[0]
        else:
            if isinstance(perturbed_value_raw, ot.Point):
                perturbed_value = perturbed_value_raw[0]
            else:
                perturbed_value = float(perturbed_value_raw)
            
        gradients[i] = (perturbed_value - nominal_value) / h
    
    # Generate validation points
    validation_sample = problem.getSample(n_validation)
    
    # Evaluate original model at validation points
    original_outputs = np.zeros(n_validation)
    
    for i in range(n_validation):
        output_raw = model(validation_sample[i])
        
        # Process outputs to ensure we have the right format
        if is_vector_output or isinstance(output_raw, ot.Point):
            # Extract the first component if it's a vector output
            if isinstance(output_raw, ot.Point):
                original_outputs[i] = output_raw[0]
            else:
                original_outputs[i] = float(output_raw[0])
        else:
            # It's a scalar
            original_outputs[i] = float(output_raw)
    
    # Evaluate Taylor surrogate at validation points
    surrogate_outputs = np.zeros(n_validation)
    for i in range(n_validation):
        point = np.array([validation_sample[i][j] for j in range(dimension)])
        # First-order Taylor approximation: f(x) ≈ f(x0) + ∇f(x0)·(x-x0)
        surrogate_outputs[i] = nominal_value + np.sum(gradients * (point - nominal_point))
    
    # Calculate validation metrics
    errors = original_outputs - surrogate_outputs
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    
    # Calculate R² (coefficient of determination)
    ss_tot = np.sum((original_outputs - np.mean(original_outputs))**2)
    ss_res = np.sum(errors**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Calculate normalized RMSE
    range_output = np.max(original_outputs) - np.min(original_outputs)
    nrmse = rmse / range_output if range_output > 0 else float('inf')
    
    # Create validation data for plotting
    validation_data = {
        'original_values': original_outputs.flatten(),
        'taylor_values': surrogate_outputs.flatten(),
        'errors': errors.flatten(),
        'points': [np.array([validation_sample[i][j] for j in range(dimension)]) for i in range(n_validation)]
    }
    
    # Return validation metrics
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r_squared,
        'nrmse': nrmse,
        'validation_data': validation_data,
        'is_vector_output': is_vector_output
    }

def create_taylor_dataframe(results):
    """Create a DataFrame with Taylor analysis results."""
    df = pd.DataFrame({
        'Variable': results['input_names'],
        'Nominal_Point': results['nominal_point'],
        'Gradient': results['gradients'],
        'Variance': results['variances'],
        'Sensitivity_Index': results['sensitivity_indices']
    })
    # Sort by sensitivity index in descending order
    df = df.sort_values('Sensitivity_Index', ascending=False)
    # Format the sensitivity index as percentage
    df['Sensitivity_Index (%)'] = df['Sensitivity_Index'] * 100
    return df

def compute_taylor_analysis(model, problem, model_code_str=None, language_model='groq'):
    """
    Compute Taylor analysis without UI components.
    
    This function performs a first-order Taylor expansion to approximate the model
    and estimate the sensitivity of the output to each input variable.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
        
    Returns
    -------
    dict
        Dictionary containing Taylor analysis results
    """
    # Compute Taylor indices
    taylor_results = compute_taylor_indices(model, problem)
    
    # Validate the Taylor approximation
    validation_results = validate_taylor_surrogate(model, problem)
    
    # Extract results
    input_names = taylor_results['input_names']
    sensitivity_indices = taylor_results['sensitivity_indices']
    gradients = taylor_results['gradients']
    variances = taylor_results['variances']
    nominal_point = taylor_results['nominal_point']
    nominal_value = taylor_results['nominal_value']
    
    # Create DataFrame
    taylor_df = create_taylor_dataframe(taylor_results)
    
    # Create bar chart for sensitivity indices
    bar_chart = go.Figure()
    
    # Add sensitivity indices
    bar_chart.add_trace(go.Bar(
        x=input_names,
        y=sensitivity_indices,
        name='Sensitivity Index',
        marker_color='rgba(31, 119, 180, 0.8)'
    ))
    
    # Update layout
    bar_chart.update_layout(
        title='Taylor-based Sensitivity Indices',
        xaxis_title='Input Variables',
        yaxis_title='Sensitivity Index',
        template='plotly_white',
        height=500
    )
    
    # Create scatter plot for validation
    validation_plot = go.Figure()
    
    # Add scatter plot
    validation_plot.add_trace(go.Scatter(
        x=validation_results['validation_data']['original_values'],
        y=validation_results['validation_data']['taylor_values'],
        mode='markers',
        marker=dict(
            color=validation_results['validation_data']['errors'],
            colorscale='RdBu_r',
            colorbar=dict(title='Error'),
            size=8
        ),
        name='Validation Points'
    ))
    
    # Add diagonal line (perfect prediction)
    min_val = min(min(validation_results['validation_data']['original_values']), min(validation_results['validation_data']['taylor_values']))
    max_val = max(max(validation_results['validation_data']['original_values']), max(validation_results['validation_data']['taylor_values']))
    
    validation_plot.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Perfect Prediction'
    ))
    
    # Update layout
    validation_plot.update_layout(
        title='Taylor Approximation Validation',
        xaxis_title='Original Model Output',
        yaxis_title='Taylor Approximation',
        template='plotly_white',
        height=500
    )
    
    # Create gradient plot
    gradient_plot = go.Figure()
    
    # Add gradient bars
    gradient_plot.add_trace(go.Bar(
        x=input_names,
        y=gradients,
        name='Gradient',
        marker_color='rgba(44, 160, 44, 0.8)'
    ))
    
    # Update layout
    gradient_plot.update_layout(
        title='Model Gradients at Nominal Point',
        xaxis_title='Input Variables',
        yaxis_title='Gradient (∂Y/∂X)',
        template='plotly_white',
        height=500
    )
    
    # Generate AI insights if requested
    ai_insights = None
    if model_code_str and language_model:
        # Create distribution information for the prompt
        dist_info = []
        for i, name in enumerate(input_names):
            marginal = problem.getMarginal(i)
            dist_info.append({
                'Variable': name,
                'Distribution': marginal.__class__.__name__,
                'Parameters': str(list(marginal.getParameter())),
                'Mean': float(marginal.getMean()[0]),
                'Std': float(marginal.getStandardDeviation()[0])
            })
        dist_df = pd.DataFrame(dist_info)
        
        # Format the model code for inclusion in the prompt
        if model_code_str:
            model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])
        else:
            model_code_formatted = "Model code not available"
        
        # Create the prompt
        prompt = f"""
{RETURN_INSTRUCTION}

## Taylor Sensitivity Analysis Results

I need your expert analysis of these Taylor sensitivity analysis results for the following model:

```python
{model_code_formatted}
```

### Input Distributions:
{dist_df.to_markdown(index=False)}

### Nominal Point:
{pd.DataFrame({'Variable': input_names, 'Value': nominal_point}).to_markdown(index=False)}

### Nominal Output Value:
{nominal_value}

### Taylor Sensitivity Analysis Results:
{taylor_df.to_markdown(index=False)}

### Validation Results:
- R² Score: {validation_results['r2']:.4f}
- Mean Absolute Error: {validation_results['mae']:.4f}
- Root Mean Squared Error: {validation_results['rmse']:.4f}

Please provide a comprehensive analysis that includes:

1. **Executive Summary**
   - Key findings and their practical implications
   - Most influential parameters identified by the Taylor analysis
   - Assessment of the Taylor approximation's accuracy

2. **Technical Analysis**
   - Interpretation of the sensitivity indices and gradients
   - Explanation of how the Taylor expansion approximates the model
   - Analysis of the linearity assumption and its validity for this model

3. **Recommendations**
   - Practical guidance for model simplification or optimization
   - Suggestions for further analysis or validation
   - Variables that should be prioritized for uncertainty reduction

Format your response with clear section headings and bullet points where appropriate.
Focus on actionable insights that would be valuable for decision-makers.
"""
        
        try:
            ai_insights = call_groq_api(prompt, model_name=language_model)
        except Exception as e:
            ai_insights = f"Error generating AI insights: {str(e)}"
    
    # Create a metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['R² Score', 'Mean Absolute Error', 'Root Mean Squared Error'],
        'Value': [
            f"{validation_results['r2']:.4f}",
            f"{validation_results['mae']:.4f}",
            f"{validation_results['rmse']:.4f}"
        ]
    })
    
    # Return all results in a dictionary with the correct keys for the comprehensive report
    return {
        # Data
        "taylor_df": taylor_df,
        "nominal_point": nominal_point,
        "nominal_value": nominal_value,
        "gradients": gradients,
        "variances": variances,
        "sensitivity_indices": sensitivity_indices,
        "input_names": input_names,
        "validation_metrics": metrics_df,
        
        # Figures
        "bar_chart": bar_chart,
        "validation_plot": validation_plot,
        "gradient_plot": gradient_plot,
        
        # AI insights
        "ai_insights": ai_insights
    }

def display_taylor_results(analysis_results, model_code_str=None, language_model='groq'):
    """
    Display Taylor analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing Taylor analysis results
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    """
    # Extract results from the analysis_results dictionary
    taylor_df = analysis_results['taylor_df']
    nominal_point = analysis_results['nominal_point']
    nominal_value = analysis_results['nominal_value']
    gradients = analysis_results['gradients']
    variances = analysis_results['variances']
    sensitivity_indices = analysis_results['sensitivity_indices']
    input_names = analysis_results['input_names']
    validation_metrics = analysis_results['validation_metrics']
    bar_chart = analysis_results['bar_chart']
    validation_plot = analysis_results['validation_plot']
    gradient_plot = analysis_results['gradient_plot']
    ai_insights = analysis_results.get('ai_insights')
    
    # Results Section
    with st.container():
        # Overview
        st.subheader("Taylor Analysis Overview")
        st.markdown(f"""
        The Taylor expansion is performed around the nominal point (typically the mean of each input variable).
        The nominal value of the output is **{nominal_value:.6f}**.
        
        The Taylor importance factors are calculated as:
        
        $$S_i = \\frac{{(\\partial f / \\partial x_i)^2 \\cdot \\sigma_i^2}}{{\\sum_j (\\partial f / \\partial x_j)^2 \\cdot \\sigma_j^2}}$$
        
        where:
        - $\\partial f / \\partial x_i$ is the partial derivative of the model with respect to variable $i$
        - $\\sigma_i^2$ is the variance of variable $i$
        
        These factors represent the contribution of each variable to the output variance, assuming the model
        can be approximated by a first-order Taylor expansion.
        """)
        
        # Create metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nominal Output Value", f"{nominal_value:.6f}")
        with col2:
            st.metric("Sum of Importance Factors", f"{sum(sensitivity_indices):.2f}%")
        with col3:
            st.metric("Linear Model R²", f"{validation_metrics['Value'][0]}")
        
        # Model Validation
        st.subheader("Linear Surrogate Model Validation")
        st.markdown("""
        The Taylor analysis creates a linear surrogate model based on the gradients at the nominal point.
        This plot compares the predictions of this linear model with the original model to assess its accuracy.
        A good fit indicates that the Taylor importance factors are reliable.
        """)
        
        # Display the plot
        st.plotly_chart(validation_plot, width='stretch')
        
        # Add warning if R² is low
        if float(validation_metrics['Value'][0]) < 0.7:
            st.warning("""
            **Warning**: The linear surrogate model does not adequately approximate the original model.
            Taylor importance factors may not provide reliable sensitivity information.
            Consider using a more advanced sensitivity analysis method like Sobol indices.
            """)
        elif float(validation_metrics['Value'][0]) < 0.9:
            st.info("""
            **Note**: The linear surrogate model provides a moderate approximation of the original model.
            Taylor importance factors should be interpreted with caution.
            """)
        
        # Taylor Importance Factors
        st.subheader("Taylor Importance Factors")
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Most Influential Variable", input_names[np.argmax(sensitivity_indices)], f"{max(sensitivity_indices)*100:.2f}%")
        with col2:
            st.metric("Number of Variables", f"{len(taylor_df)}")
        with col3:
            st.metric("Variables with >5% Influence", f"{sum(sensitivity_indices > 0.05)}")
        
        # Create the bar chart
        st.plotly_chart(bar_chart, width='stretch')
        
        # Detailed Results
        st.subheader("Detailed Results")
        
        # Create a display dataframe with formatted values
        display_df = pd.DataFrame({
            'Variable': input_names,
            'Nominal_Point': nominal_point,
            'Gradient': gradients,
            'Variance': variances,
            'Sensitivity_Index': sensitivity_indices,
            'Sensitivity_Index (%)': sensitivity_indices * 100
        })
        
        # Display the dataframe
        st.dataframe(display_df, width='stretch')
    
    # AI Insights section (only if ai_insights is available)
    if ai_insights:
        st.markdown("### AI-Generated Expert Analysis")
        st.markdown(ai_insights)

def taylor_analysis(model, problem, model_code_str=None, language_model='groq', display_results=True):
    """
    Perform enterprise-grade Taylor analysis on the model.
    
    This analysis uses a first-order Taylor expansion to approximate the model
    and estimate the sensitivity of the output to each input variable.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    display_results : bool, optional
        Whether to display results in the UI (default is True)
        
    Returns
    -------
    dict
        Dictionary containing Taylor analysis results
    """
    try:
        # Compute Taylor analysis results
        analysis_results = compute_taylor_analysis(model, problem, model_code_str, language_model)
        
        # Display results in the Streamlit interface if requested
        if display_results:
            display_taylor_results(analysis_results, model_code_str, language_model)
        
        return analysis_results
    
    except Exception as e:
        if display_results:
            st.error(f"Error in Taylor analysis: {str(e)}")
        raise

def get_taylor_context_for_chat(taylor_results):
    """
    Generate a formatted string containing Taylor analysis results for the global chat context.
    
    Parameters
    ----------
    taylor_results : dict
        Dictionary containing the results of the Taylor analysis
        
    Returns
    -------
    str
        Formatted string with Taylor analysis results for chat context
    """
    context = ""
    
    # Try to get a DataFrame from the results
    taylor_indices_df = None
    if "indices_df" in taylor_results:
        taylor_indices_df = taylor_results["indices_df"]
    elif "input_names" in taylor_results and "sensitivity_indices" in taylor_results:
        taylor_indices_df = pd.DataFrame({
            'Variable': taylor_results['input_names'],
            'Sensitivity Index': taylor_results['sensitivity_indices']
        })
        taylor_indices_df = taylor_indices_df.sort_values('Sensitivity Index', ascending=False)
    
    if taylor_indices_df is not None:
        context += "\n\n### Taylor Sensitivity Analysis Results\n"
        context += taylor_indices_df.to_markdown(index=False)
    
    return context
