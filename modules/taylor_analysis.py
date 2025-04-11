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
        'original': original_outputs.flatten(),
        'surrogate': surrogate_outputs.flatten(),
        'errors': errors.flatten(),
        'points': [np.array([validation_sample[i][j] for j in range(dimension)]) for i in range(n_validation)]
    }
    
    # Return validation metrics
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
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
    
    # Validate the Taylor surrogate model
    validation_results = validate_taylor_surrogate(model, problem, n_validation=100)
    
    # Create Taylor dataframe
    taylor_df = create_taylor_dataframe(taylor_results)
    
    # Create detailed dataframe
    detailed_df = pd.DataFrame({
        'Variable': taylor_results['input_names'],
        'Nominal_Point': taylor_results['nominal_point'],
        'Gradient': taylor_results['gradients'],
        'Variance': taylor_results['variances'],
        'Sensitivity_Index': taylor_results['sensitivity_indices'],
        'Sensitivity_Index (%)': taylor_results['sensitivity_indices'] * 100
    })
    
    # Sort by importance
    taylor_df = taylor_df.sort_values('Sensitivity_Index (%)', ascending=False).reset_index(drop=True)
    detailed_df = detailed_df.sort_values('Sensitivity_Index', ascending=False).reset_index(drop=True)
    
    # Get most influential variable
    most_influential = taylor_df.iloc[0]['Variable']
    most_influential_value = taylor_df.iloc[0]['Sensitivity_Index (%)']
    
    # Calculate sum of importance factors
    sum_importance = taylor_df['Sensitivity_Index (%)'].sum()
    
    # Generate AI insights if a language model is provided
    ai_insights = None
    if language_model and model_code_str:
        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')]) if model_code_str else ""
        
        # Prepare the inputs description
        input_parameters = []
        dimension = problem.getDimension()
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = taylor_results['input_names'][i]
            dist_type = marginal.__class__.__name__
            params = marginal.getParameter()
            input_parameters.append(f"- **{name}**: {dist_type} distribution with parameters {list(params)}")
        
        inputs_description = '\n'.join(input_parameters)
        
        # Create a markdown table for the Taylor indices
        taylor_table = taylor_df.to_markdown(index=False)
        
        # Create the prompt
        prompt = f"""
        {RETURN_INSTRUCTION}
        
        I've performed a Taylor sensitivity analysis on the following model:
        
        ```python
        {model_code_formatted}
        ```
        
        with the following input distributions:
        
        {inputs_description}
        
        Here are the Taylor sensitivity indices:
        
        {taylor_table}
        
        The model was evaluated at the nominal point (typically the mean of each input variable).
        The nominal value of the output is {taylor_results['nominal_value']:.6f}.
        
        The linear surrogate model has an R² of {validation_results['r_squared']:.4f} and RMSE of {validation_results['rmse']:.4f}.
        
        Please provide an expert analysis of these Taylor sensitivity results, including:
        
        1. An explanation of what Taylor sensitivity indices represent and how they're calculated
        2. An interpretation of the most important variables and their influence
        3. A discussion of the validity of the linear approximation based on the R² value
        4. Recommendations for further analysis or model refinement
        
        Keep your response concise, technical, and focused on actionable insights.
        """
        
        try:
            ai_insights = call_groq_api(prompt, model_name=language_model)
        except Exception as e:
            ai_insights = f"Unable to generate AI insights: {str(e)}"
    
    # Return all results in a dictionary
    return {
        'taylor_results': taylor_results,
        'validation_results': validation_results,
        'taylor_df': taylor_df,
        'detailed_df': detailed_df,
        'most_influential': most_influential,
        'most_influential_value': most_influential_value,
        'sum_importance': sum_importance,
        'ai_insights': ai_insights
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
    taylor_results = analysis_results['taylor_results']
    validation_results = analysis_results['validation_results']
    taylor_df = analysis_results['taylor_df']
    detailed_df = analysis_results['detailed_df']
    most_influential = analysis_results['most_influential']
    most_influential_value = analysis_results['most_influential_value']
    sum_importance = analysis_results['sum_importance']
    ai_insights = analysis_results.get('ai_insights')
    
    # Results Section
    with st.expander("Results", expanded=True):
        # Overview
        st.subheader("Taylor Analysis Overview")
        st.markdown(f"""
        The Taylor expansion is performed around the nominal point (typically the mean of each input variable).
        The nominal value of the output is **{taylor_results['nominal_value']:.6f}**.
        
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
            st.metric("Nominal Output Value", f"{taylor_results['nominal_value']:.6f}")
        with col2:
            st.metric("Sum of Importance Factors", f"{sum_importance:.2f}%")
        with col3:
            st.metric("Linear Model R²", f"{validation_results['r_squared']:.4f}")
        
        # Model Validation
        st.subheader("Linear Surrogate Model Validation")
        st.markdown("""
        The Taylor analysis creates a linear surrogate model based on the gradients at the nominal point.
        This plot compares the predictions of this linear model with the original model to assess its accuracy.
        A good fit indicates that the Taylor importance factors are reliable.
        """)
        
        # Extract validation data and R²
        validation_data = validation_results['validation_data']
        r2_value = validation_results['r_squared']
        
        # Create the scatter plot
        fig_validation = px.scatter(
            x=validation_data['original'],
            y=validation_data['surrogate'],
            labels={'x': 'Original Model', 'y': 'Linear Surrogate Model'},
            title='Original vs. Linear Surrogate Model Predictions'
        )
        
        # Add the perfect prediction line
        min_val = min(min(validation_data['original']), min(validation_data['surrogate']))
        max_val = max(max(validation_data['original']), max(validation_data['surrogate']))
        fig_validation.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        # Update layout
        fig_validation.update_layout(
            template='plotly_white',
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Display the plot
        st.plotly_chart(fig_validation, use_container_width=True)
        
        # Add warning if R² is low
        if r2_value < 0.7:
            st.warning("""
            **Warning**: The linear surrogate model does not adequately approximate the original model.
            Taylor importance factors may not provide reliable sensitivity information.
            Consider using a more advanced sensitivity analysis method like Sobol indices.
            """)
        elif r2_value < 0.9:
            st.info("""
            **Note**: The linear surrogate model provides a moderate approximation of the original model.
            Taylor importance factors should be interpreted with caution.
            """)
        
        # Taylor Importance Factors
        st.subheader("Taylor Importance Factors")
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Most Influential Variable", most_influential, f"{most_influential_value:.2f}%")
        with col2:
            st.metric("Number of Variables", f"{len(taylor_df)}")
        with col3:
            st.metric("Variables with >5% Influence", f"{sum(taylor_df['Sensitivity_Index (%)'] > 5)}")
        
        # Create the bar chart
        fig_bar = px.bar(
            taylor_df,
            x='Variable',
            y='Sensitivity_Index (%)',
            title='Taylor Importance Factors (%)',
            text='Sensitivity_Index (%)',
            color='Sensitivity_Index (%)',
            color_continuous_scale='Viridis'
        )
        
        # Update layout
        fig_bar.update_layout(
            template='plotly_white',
            height=500,
            coloraxis_showscale=False,
            yaxis_title='Importance (%)',
            xaxis_title='Variable'
        )
        
        # Format text
        fig_bar.update_traces(
            texttemplate='%{text:.2f}%',
            textposition='outside'
        )
        
        # Display the plot
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed Results
        st.subheader("Detailed Results")
        
        # Create a display dataframe with formatted values
        display_df = detailed_df.copy()
        display_df['Gradient'] = display_df['Gradient'].apply(lambda x: f"{x:.6f}")
        display_df['Variance'] = display_df['Variance'].apply(lambda x: f"{x:.6f}")
        display_df['Sensitivity_Index'] = display_df['Sensitivity_Index'].apply(lambda x: f"{x:.6f}")
        display_df['Sensitivity_Index (%)'] = display_df['Sensitivity_Index (%)'].apply(lambda x: f"{x:.2f}%")
        
        # Display the dataframe
        st.dataframe(display_df, use_container_width=True)
    
    # AI Insights section (only if ai_insights is available)
    if ai_insights:
        with st.expander("AI Insights", expanded=True):
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
