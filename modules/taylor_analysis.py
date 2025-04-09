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
        # Create a two-column layout for the main content and chat interface
        main_col, chat_col = st.columns([2, 1])
        
        with main_col:
            # Compute Taylor indices
            with st.spinner("Computing Taylor indices..."):
                taylor_results = compute_taylor_indices(model, problem)
            
            # Validate the Taylor surrogate model
            with st.spinner("Validating Taylor surrogate model..."):
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
                
                # Get most influential variable
                most_influential = taylor_df.iloc[0]['Variable']
                most_influential_value = taylor_df.iloc[0]['Sensitivity_Index (%)']
                
                # Calculate sum of importance factors
                sum_importance = taylor_df['Sensitivity_Index (%)'].sum()
                
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
                        y='Sensitivity_Index (%)',
                        color='Sensitivity_Index (%)',
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
                        values='Sensitivity_Index (%)',
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
                
                # Explanation of gradients
                st.markdown("""
                **Gradient Interpretation:**
                - **Positive Gradient**: An increase in the input variable leads to an increase in the output
                - **Negative Gradient**: An increase in the input variable leads to a decrease in the output
                - **Magnitude**: The absolute value indicates how sensitive the output is to small changes in the input
                """)
            
            # Prepare data for AI analysis
            # Format model code for display
            model_code_formatted = model_code_str if model_code_str else "Model code not available"
            
            # Create a markdown table of the Taylor indices
            taylor_md_table = "| Variable | Importance Factor (%) | Gradient | Variance |\n"
            taylor_md_table += "|----------|----------------------|----------|----------|\n"
            
            for _, row in detailed_df.iterrows():
                taylor_md_table += f"| {row['Variable']} | {row['Sensitivity_Index (%)']: .2f}% | {row['Gradient']: .6f} | {row['Variance']: .6f} |\n"
            
            # Create a description of the input distributions
            inputs_description = "Input distributions:\n"
            for i in range(problem.getDimension()):
                marginal = problem.getMarginal(i)
                name = taylor_results['input_names'][i]
                mean = marginal.getMean()[0]
                std = marginal.getStandardDeviation()[0]
                dist_type = str(marginal).split('(')[0]
                inputs_description += f"- {name}: {dist_type} distribution with mean={mean:.4f}, std={std:.4f}\n"
            
            # Add validation information
            validation_info = f"""
            Surrogate model validation:
            - R² Score: {validation_results['r_squared']:.4f}
            - RMSE: {validation_results['rmse']:.6f}
            - Normalized RMSE: {validation_results['nrmse']:.4f}
            """
            
            # Create the prompt for AI analysis
            prompt = f"""
            You are an expert in uncertainty quantification and sensitivity analysis.
            
            Analyze the following Taylor expansion sensitivity analysis results for a mathematical model.
            
            Model code:
            ```python
            {model_code_formatted}
            ```
            
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
        
        # CHAT INTERFACE in the right column
        if language_model:
            with chat_col:
                st.markdown("### Ask Questions About This Analysis")
                
                # Display a disclaimer about the prompt
                disclaimer_text = """
                **Note:** The AI assistant has been provided with the model code, input distributions, 
                and the Taylor analysis results above. You can ask questions to clarify any aspects of the analysis.
                """
                st.info(disclaimer_text)
                
                # Initialize session state for chat messages if not already done
                if "taylor_analysis_chat_messages" not in st.session_state:
                    st.session_state.taylor_analysis_chat_messages = []
                
                # Create a container with fixed height for the chat messages
                chat_container_height = 500  # Height in pixels
                
                # Apply CSS to create a scrollable container
                st.markdown(f"""
                <style>
                .chat-container {{
                    height: {chat_container_height}px;
                    overflow-y: auto;
                    border: 1px solid #e6e6e6;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #f9f9f9;
                    margin-bottom: 15px;
                }}
                </style>
                """, unsafe_allow_html=True)
                
                # Create a container for the chat messages
                with st.container():
                    # Use HTML to create a scrollable container
                    chat_messages_html = "<div class='chat-container'>"
                    
                    # Display existing messages
                    for message in st.session_state.taylor_analysis_chat_messages:
                        role_style = "background-color: #e1f5fe; border-radius: 10px; padding: 8px; margin: 5px 0;" if message["role"] == "assistant" else "background-color: #f0f0f0; border-radius: 10px; padding: 8px; margin: 5px 0;"
                        role_label = "Assistant:" if message["role"] == "assistant" else "You:"
                        chat_messages_html += f"<div style='{role_style}'><strong>{role_label}</strong><br>{message['content']}</div>"
                    
                    chat_messages_html += "</div>"
                    st.markdown(chat_messages_html, unsafe_allow_html=True)
                
                # Chat input below the scrollable container
                prompt = st.chat_input("Ask a question about the Taylor analysis...", key="taylor_side_chat_input")
                
                # Process user input
                if prompt:
                    # Add user message to chat history
                    st.session_state.taylor_analysis_chat_messages.append({"role": "user", "content": prompt})
                    
                    # Define context generator function
                    def generate_context(prompt):
                        # Format the indices for the context
                        indices_summary = ', '.join([f"{row['Variable']}: Gradient={row['Gradient']:.4f}, Sensitivity Index={row['Sensitivity_Index']:.4f}" 
                                                  for _, row in detailed_df.iterrows()])
                        
                        return f"""
                        You are an expert assistant helping users understand Taylor sensitivity analysis results. 
                        
                        Here is the model code:
                        ```python
                        {model_code_formatted if model_code_formatted else "Model code not available"}
                        ```
                        
                        Here is information about the input distributions:
                        {inputs_description}
                        
                        Here is the Taylor analysis summary:
                        {indices_summary}
                        
                        Nominal point: {taylor_results['nominal_point']}
                        Nominal value: {taylor_results['nominal_value']:.6f}
                        
                        Here is the explanation that was previously generated:
                        {st.session_state.get('taylor_response_markdown', 'No analysis available yet.')}
                        
                        Answer the user's question based on this information. Be concise but thorough.
                        If you're not sure about something, acknowledge the limitations of your knowledge.
                        Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                        Explain the difference between gradients and sensitivity indices if asked.
                        """
                    
                    # Generate context for the assistant
                    context = generate_context(prompt)
                    
                    # Include previous conversation history
                    chat_history = ""
                    if len(st.session_state.taylor_analysis_chat_messages) > 1:
                        chat_history = "Previous conversation:\n"
                        for i, msg in enumerate(st.session_state.taylor_analysis_chat_messages[:-1]):
                            role = "User" if msg["role"] == "user" else "Assistant"
                            chat_history += f"{role}: {msg['content']}\n\n"
                    
                    # Create the final prompt
                    chat_prompt = f"""
                    {context}
                    
                    {chat_history}
                    
                    Current user question: {prompt}
                    
                    Please provide a helpful, accurate response to this question.
                    """
                    
                    # Call API with chat history
                    with st.spinner("Thinking..."):
                        try:
                            response_text = call_groq_api(chat_prompt, model_name=language_model)
                        except Exception as e:
                            st.error(f"Error calling API: {str(e)}")
                            response_text = "I'm sorry, I encountered an error while processing your question. Please try again."
                    
                    # Add assistant response to chat history
                    st.session_state.taylor_analysis_chat_messages.append({"role": "assistant", "content": response_text})
                    
                    # Rerun to display the new message immediately
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error in Taylor analysis: {str(e)}")
        raise
