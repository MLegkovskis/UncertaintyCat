# modules/taylor_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
from utils.core_utils import call_groq_api
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

def taylor_analysis(model, problem, model_code_str=None, language_model='groq'):
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
    """
    try:
        # Ensure problem is an OpenTURNS distribution
        if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
            raise ValueError("Problem must be an OpenTURNS distribution")
        
        # Get input names
        dimension = problem.getDimension()
        input_names = []
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = marginal.getDescription()[0]
            input_names.append(name if name != "" else f"X{i+1}")
        
        # Define the output as the result of the model applied to the input vector
        input_vector = ot.RandomVector(problem)
        output_vector = ot.CompositeRandomVector(model, input_vector)

        # Perform Taylor expansion moments and importance factors analysis
        taylor = ot.TaylorExpansionMoments(output_vector)
        importance_factors = taylor.getImportanceFactors()

        # Extract the importance factors and multiply by 100 for percentage representation
        importance_values = [importance_factors[i] * 100 for i in range(len(input_names))]
        variable_names = input_names

        # Create a DataFrame for the importance factors
        taylor_df = pd.DataFrame({
            'Variable': variable_names,
            'Importance Factor (%)': importance_values
        })
        
        # Sort by importance factor in descending order
        taylor_df = taylor_df.sort_values('Importance Factor (%)', ascending=False)
        
        # Validate the Taylor surrogate model
        validation_results = validate_taylor_surrogate(model, problem, n_validation=100)
        
        # Compute additional Taylor analysis results
        taylor_results = compute_taylor_indices(model, problem)
        detailed_df = create_taylor_dataframe(taylor_results)
        
        # Results Section
        with st.expander("Results", expanded=True):
            st.subheader("Taylor Expansion Sensitivity Analysis")
            st.markdown("""
            Taylor expansion sensitivity analysis approximates the model using a first-order Taylor series
            and estimates the contribution of each input variable to the output variance.
            
            This method is computationally efficient but relies on the assumption that the model
            behaves linearly around the nominal point (usually the mean of the input distributions).
            """)
            
            # Surrogate Model Validation
            st.subheader("Linear Surrogate Model Validation")
            st.markdown("""
            Taylor importance factors are only meaningful when the first-order Taylor expansion
            is a good approximation of the original model. The metrics below assess how well
            the linear surrogate model approximates the original model.
            """)
            
            # Create metrics in columns
            col1, col2, col3 = st.columns(3)
            
            # Display R² with color coding
            r2_value = validation_results['r_squared']
            r2_color = 'normal'
            if r2_value < 0.7:
                r2_color = 'off'
                r2_warning = "Poor linear approximation"
            elif r2_value < 0.9:
                r2_color = 'normal'  
                r2_warning = "Moderate approximation"
            else:
                r2_warning = "Good approximation"
                
            col1.metric("R² Score", f"{r2_value:.4f}", delta=r2_warning, delta_color=r2_color)
            col2.metric("RMSE", f"{validation_results['rmse']:.4f}")
            col3.metric("Normalized RMSE", f"{validation_results['nrmse']:.4f}")
            
            # Add interpretation guidance
            st.markdown(f"""
            **Interpretation Guide:**
            - **R² Score**: Measures how well the linear model explains the variance in the original model
                - R² > 0.9: Excellent linear approximation
                - 0.7 < R² < 0.9: Good linear approximation
                - R² < 0.7: Poor linear approximation, Taylor indices may be misleading
            
            - **RMSE**: Root Mean Square Error between the original model and the linear surrogate
            
            - **Normalized RMSE**: RMSE divided by the range of the output, provides a scale-independent error metric
                - NRMSE < 0.1: Excellent approximation
                - 0.1 < NRMSE < 0.2: Good approximation
                - NRMSE > 0.2: Poor approximation
            """)
            
            # Create scatter plot of original vs surrogate model predictions
            validation_data = validation_results['validation_data']
            
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
                ⚠️ **Warning**: The linear surrogate model does not adequately approximate the original model.
                Taylor importance factors may not provide reliable sensitivity information.
                Consider using a more advanced sensitivity analysis method like Sobol indices.
                """)
            elif r2_value < 0.9:
                st.info("""
                ℹ️ **Note**: The linear surrogate model provides a moderate approximation of the original model.
                Taylor importance factors should be interpreted with caution.
                """)
            
            # Taylor Importance Factors
            st.subheader("Taylor Importance Factors")
            
            # Get most influential variable
            most_influential = taylor_df.iloc[0]['Variable']
            most_influential_value = taylor_df.iloc[0]['Importance Factor (%)']
            
            # Calculate sum of importance factors
            sum_importance = taylor_df['Importance Factor (%)'].sum()
            
            # Create summary metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Most Influential Variable", 
                    most_influential,
                    f"Importance: {most_influential_value:.2f}%"
                )
            with col2:
                st.metric("Sum of Importance Factors", f"{sum_importance:.2f}%")
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
            
            with tab1:
                # Create bar chart using Plotly
                fig_bar = px.bar(
                    taylor_df,
                    x='Variable',
                    y='Importance Factor (%)',
                    color='Importance Factor (%)',
                    color_continuous_scale='Viridis',
                    title='Taylor Importance Factors'
                )
                
                # Update layout
                fig_bar.update_layout(
                    xaxis_title='Input Variable',
                    yaxis_title='Importance Factor (%)',
                    template='plotly_white',
                    height=500
                )
                
                # Display the bar chart
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab2:
                # Create pie chart using Plotly
                fig_pie = px.pie(
                    taylor_df,
                    values='Importance Factor (%)',
                    names='Variable',
                    title='Taylor Importance Factors',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                
                # Update layout
                fig_pie.update_layout(
                    template='plotly_white',
                    height=500,
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
                )
                
                # Display the pie chart
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Display the importance factors table
            st.subheader("Importance Factors Table")
            st.dataframe(taylor_df, use_container_width=True)
            
            # Detailed Analysis
            st.subheader("Detailed Taylor Analysis Results")
            
            # Display the detailed results table
            st.dataframe(detailed_df[['Variable', 'Nominal_Point', 'Gradient', 'Variance', 'Sensitivity_Index (%)']], use_container_width=True)
            
            # Display explanation of the columns
            st.markdown("""
            **Table Explanation:**
            - **Variable**: Input variable name
            - **Nominal_Point**: Value at which the Taylor expansion is computed (typically the mean)
            - **Gradient**: Partial derivative of the model with respect to the variable at the nominal point
            - **Variance**: Variance of the input variable
            - **Sensitivity_Index (%)**: Contribution of the variable to the output variance, calculated as (Gradient² × Variance) / Total Variance
            """)
            
            # Create a heatmap of gradients
            gradient_df = detailed_df.copy()
            gradient_df['Abs_Gradient'] = np.abs(gradient_df['Gradient'])
            gradient_df = gradient_df.sort_values('Abs_Gradient', ascending=False)
            
            # Create the heatmap
            fig_heatmap = px.imshow(
                np.array([gradient_df['Gradient']]),
                x=gradient_df['Variable'],
                color_continuous_scale='RdBu_r',
                title='Model Gradients at Nominal Point',
                labels=dict(color='Gradient Value')
            )
            
            # Update layout
            fig_heatmap.update_layout(
                template='plotly_white',
                height=300,
                yaxis_visible=False
            )
            
            # Display the heatmap
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Add explanation of gradients
            st.markdown("""
            **Gradient Interpretation:**
            - **Positive gradient**: Variable has a positive effect on the output (increasing the variable increases the output)
            - **Negative gradient**: Variable has a negative effect on the output (increasing the variable decreases the output)
            - **Larger magnitude**: Variable has a stronger local effect on the output
            """)
        
        # Generate prompt for AI insights
        # Prepare the data for the API call
        taylor_md_table = taylor_df.to_markdown(index=False, floatfmt=".2f")
        
        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')]) if model_code_str else ""
        
        # Prepare the inputs description
        input_parameters = []
        dimension = problem.getDimension()
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = problem.getDescription()[i]
            dist_type = marginal.__class__.__name__
            params = marginal.getParameter()
            input_parameters.append(f"- **{name}**: {dist_type} distribution with parameters {list(params)}")
        
        inputs_description = '\n'.join(input_parameters)
        
        # Add validation results to the prompt
        validation_info = f"""
        The linear surrogate model validation results:
        - R² Score: {validation_results['r_squared']:.4f}
        - RMSE: {validation_results['rmse']:.4f}
        - Normalized RMSE: {validation_results['nrmse']:.4f}
        """
        
        # Prepare the prompt
        prompt = f"""
        {RETURN_INSTRUCTION}
        
        Given the following user-defined model defined in Python code:
        
        ```python
        {model_code_formatted}
        ```
        
        and the following uncertain input distributions:
        
        {inputs_description}
        
        The results of the Taylor expansion importance factors are given below:
        
        {taylor_md_table}
        
        {validation_info}
        
        Please provide an expert analysis of the Taylor expansion sensitivity results:
        
        1. **Methodology Overview**
           - Explain the mathematical basis of the Taylor expansion method in sensitivity analysis
           - Discuss how the importance factors are calculated using the derivatives of the model
           - Explain the limitations of this approach and when it is most appropriate
        
        2. **Results Interpretation**
           - Interpret the importance factors, focusing on which variables have the most significant impact on the output variance
           - Discuss the physical or mathematical reasons behind the importance of these variables
           - Explain what the gradient signs tell us about how each variable affects the output
        
        3. **Surrogate Model Validity**
           - Assess whether the linear surrogate model is a good approximation based on the validation metrics
           - Explain the implications if the R² score is low
           - Recommend alternative approaches if the linear approximation is inadequate
        
        4. **Recommendations**
           - Suggest which variables should be prioritized for uncertainty reduction
           - Provide guidance on how these results can inform decision-making
           - Identify any potential limitations or caveats in the analysis
        
        Format your response with clear section headings and bullet points. Focus on actionable insights and quantitative recommendations.
        """
        
        # Display AI insights
        if language_model:
            with st.expander("AI Insights", expanded=True):
                # Check if the results are already in session state
                if 'taylor_response_markdown' not in st.session_state:
                    # Call the AI API
                    with st.spinner("Generating expert analysis..."):
                        response_markdown = call_groq_api(prompt, model_name=language_model)
                    # Store the response in session state
                    st.session_state.taylor_response_markdown = response_markdown
                else:
                    response_markdown = st.session_state.taylor_response_markdown
                
                # Display the response
                st.markdown(response_markdown)
    
    except Exception as e:
        st.error(f"Error in Taylor analysis: {str(e)}")
        raise
