import streamlit as st
import numpy as np
import openturns as ot
import otmorris
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

def run_morris_analysis(model, problem, n_trajectories=10, n_levels=5):
    """
    Run Morris sensitivity analysis on the given model and problem.
    
    Parameters:
    -----------
    model : ot.Function
        The OpenTURNS model function
    problem : ot.JointDistribution
        The joint distribution of input variables
    n_trajectories : int
        Number of trajectories for Morris analysis
    n_levels : int
        Number of levels for the Morris grid
        
    Returns:
    --------
    Dict with Morris analysis results
    """
    # Get input dimension and names
    input_dim = model.getInputDimension()
    input_names = problem.getDescription()
    
    # Create Morris experiment
    bounds = problem.getRange()
    levels = [n_levels] * input_dim
    
    # Create Morris experiment using grid design
    morris_experiment = otmorris.MorrisExperimentGrid(levels, n_trajectories)
    
    # Generate the design of experiments in [0, 1]^input_dim
    X_unit = morris_experiment.generate()
    
    # Transform to the actual distribution bounds
    lower_bound = bounds.getLowerBound()
    upper_bound = bounds.getUpperBound()
    X = ot.Sample(X_unit.getSize(), input_dim)
    
    # Manually transform each point to avoid Point + Sample operation
    for i in range(X_unit.getSize()):
        for j in range(input_dim):
            X[i, j] = lower_bound[j] + X_unit[i, j] * (upper_bound[j] - lower_bound[j])
    
    # Evaluate the model on the design
    Y = model(X)
    
    # Create Morris object and compute effects
    morris = otmorris.Morris(X, Y, bounds)
    
    # Get Morris indices
    mean_effects = morris.getMeanElementaryEffects()
    mean_abs_effects = morris.getMeanAbsoluteElementaryEffects()
    std_effects = morris.getStandardDeviationElementaryEffects()
    
    # Check for NaN or Inf values and replace them
    mean_abs_effects_clean = []
    std_effects_clean = []
    
    for i in range(len(mean_abs_effects)):
        mean_val = mean_abs_effects[i]
        std_val = std_effects[i]
        
        # Replace NaN or Inf with 0 for mean
        if np.isnan(mean_val) or np.isinf(mean_val):
            mean_val = 0.0
        
        # Replace NaN or Inf with 0 for std
        if np.isnan(std_val) or np.isinf(std_val):
            std_val = 0.0
        
        mean_abs_effects_clean.append(mean_val)
        std_effects_clean.append(std_val)
    
    # Create results dictionary with cleaned values
    results = {
        'mean_effects': morris.getMeanElementaryEffects(),
        'mean_abs_effects': mean_abs_effects_clean,
        'std_effects': std_effects_clean,
        'input_names': input_names,
        'morris_object': morris
    }
    
    return results

def plot_morris_results_plotly(results):
    """
    Create Plotly plots for Morris analysis results.
    
    Parameters:
    -----------
    results : Dict
        Results from run_morris_analysis
        
    Returns:
    --------
    Tuple of Plotly figures
    """
    mean_abs_effects = results['mean_abs_effects']
    std_effects = results['std_effects']
    input_names = results['input_names']
    
    # Convert to lists to avoid pandas scalar value error
    mean_abs_effects_list = [float(val) for val in mean_abs_effects]
    std_effects_list = [float(val) for val in std_effects]
    input_names_list = [str(name) for name in input_names]
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Variable': input_names_list,
        'Mean Absolute Effects': mean_abs_effects_list,
        'Standard Deviation': std_effects_list
    })
    
    # Sort by mean absolute effects for the bar chart (descending order)
    df_sorted = df.sort_values('Mean Absolute Effects', ascending=True)  # Reverse order for horizontal bar chart
    
    # Create bar plot for mean absolute effects using Plotly
    fig1 = px.bar(
        df_sorted, 
        x='Mean Absolute Effects', 
        y='Variable', 
        orientation='h',
        labels={'Mean Absolute Effects': 'Mean Absolute Effects', 'Variable': 'Input Variables'},
        title='Morris Mean Absolute Elementary Effects',
        color='Mean Absolute Effects',
        color_continuous_scale='Blues'
    )
    
    fig1.update_layout(
        font=dict(size=14),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', autorange="reversed")  # Reverse y-axis to put highest at top
    )
    
    # Create scatter plot (μ* vs σ) using Plotly - without text labels on points
    fig2 = px.scatter(
        df, 
        x='Mean Absolute Effects', 
        y='Standard Deviation',
        labels={
            'Mean Absolute Effects': 'Mean Absolute Effects (μ*)', 
            'Standard Deviation': 'Standard Deviation of Effects (σ)'
        },
        title='Morris Analysis: μ* vs σ',
        color='Mean Absolute Effects',
        size=[30] * len(df),  # Slightly larger point size
        hover_name='Variable',  # Use variable name in hover
        color_continuous_scale='Viridis'
    )
    
    # Add reference line
    max_val = max(df['Mean Absolute Effects'].max(), df['Standard Deviation'].max()) * 1.1
    fig2.add_trace(
        go.Scatter(
            x=[0, max_val], 
            y=[0, max_val], 
            mode='lines', 
            line=dict(color='red', dash='dash', width=1),
            name='μ* = σ',
            hoverinfo='none'
        )
    )
    
    # Improve hover information
    fig2.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Mean Abs Effect: %{x:.4f}<br>Std Dev: %{y:.4f}<extra></extra>'
    )
    
    fig2.update_layout(
        font=dict(size=14),
        height=600,
        width=700,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='lightgray')
    )
    
    return fig1, fig2

def identify_non_influential_variables(results, threshold_percentage=5):
    """
    Identify non-influential variables based on Morris results.
    
    Parameters:
    -----------
    results : Dict
        Results from run_morris_analysis
    threshold_percentage : float
        Percentage threshold below which variables are considered non-influential
        
    Returns:
    --------
    List of non-influential variable names and their indices
    """
    mean_abs_effects = results['mean_abs_effects']
    input_names = results['input_names']
    
    # Convert to lists to avoid scalar value issues
    mean_abs_effects_list = [float(val) for val in mean_abs_effects]
    input_names_list = [str(name) for name in input_names]
    
    # Calculate threshold as percentage of maximum effect
    max_effect = max(mean_abs_effects_list)
    threshold = max_effect * (threshold_percentage / 100)
    
    # Identify variables with effects below threshold
    non_influential = []
    for i, (name, effect) in enumerate(zip(input_names_list, mean_abs_effects_list)):
        if effect < threshold:
            non_influential.append((name, i, effect))
    
    return non_influential

def get_recommended_fixed_values(problem, var_indices):
    """
    Get recommended fixed values for variables based on their distributions.
    
    Parameters:
    -----------
    problem : ot.JointDistribution
        The joint distribution of input variables
    var_indices : List[int]
        Indices of variables to fix
        
    Returns:
    --------
    DataFrame with variable names, recommended values, and rationale
    """
    recommendations = []
    input_names = problem.getDescription()
    
    for idx in var_indices:
        dist = problem.getMarginal(idx)
        name = str(input_names[idx])
        
        # Try different methods to get a good fixed value
        try:
            # First try the mean
            mean_value = float(dist.getMean()[0])
            recommendations.append({
                'Variable': name,
                'Recommended Value': mean_value,
                'Rationale': 'Mean of distribution'
            })
        except:
            try:
                # If mean doesn't work, try median
                median_value = float(dist.computeQuantile(0.5)[0])
                recommendations.append({
                    'Variable': name,
                    'Recommended Value': median_value,
                    'Rationale': 'Median (50th percentile)'
                })
            except:
                # If all else fails, use midpoint of range
                bounds = dist.getRange()
                midpoint = (float(bounds.getLowerBound()[0]) + float(bounds.getUpperBound()[0])) / 2
                recommendations.append({
                    'Variable': name,
                    'Recommended Value': midpoint,
                    'Rationale': 'Midpoint of range'
                })
    
    return pd.DataFrame(recommendations)

def morris_analysis(model, problem, code_snippet, language_model=None):
    """
    Main function to run Morris analysis in Streamlit.
    
    Parameters:
    -----------
    model : ot.Function
        The OpenTURNS model function
    problem : ot.JointDistribution
        The joint distribution of input variables
    code_snippet : str
        The model code snippet
    language_model : str, optional
        Name of the language model to use for explanations
    """
    st.write("Morris analysis is a screening method used to identify which input variables have significant effects on the model output.")
    st.write("It helps to reduce model dimensionality by identifying variables that can be fixed at nominal values.")
    
    # Get input dimension and names
    input_dim = model.getInputDimension()
    input_names = problem.getDescription()
    
    # UI controls
    col1, col2 = st.columns(2)
    with col1:
        n_trajectories = st.slider("Number of trajectories", min_value=5, max_value=50, value=10, 
                                  help="More trajectories provide better estimates but increase computation time")
    with col2:
        n_levels = st.slider("Number of levels", min_value=3, max_value=10, value=5,
                            help="Number of levels in the Morris grid design")
    
    threshold = st.slider("Threshold for non-influential variables (%)", min_value=1, max_value=20, value=5,
                         help="Variables with effects below this percentage of the maximum effect are considered non-influential")
    
    # Add a Run Analysis button
    if st.button("Run Morris Analysis", key="run_morris_analysis_btn"):
        with st.spinner("Running Morris analysis..."):
            try:
                # Run Morris analysis
                results = run_morris_analysis(model, problem, n_trajectories, n_levels)
                
                # Plot results using Plotly
                fig1, fig2 = plot_morris_results_plotly(results)
                
                # Display plots in styled sections
                st.subheader("Morris Mean Absolute Elementary Effects")
                st.plotly_chart(fig1, use_container_width=True)
                
                st.subheader("Morris Analysis: μ* vs σ")
                st.write("Variables in the top-right corner have high influence and non-linear effects or interactions. Variables near the origin have low influence on the output.")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Identify non-influential variables
                non_influential = identify_non_influential_variables(results, threshold)
                
                if non_influential:
                    st.subheader("Non-influential Variables")
                    st.write(f"The following variables have effects below {threshold}% of the maximum effect and can potentially be fixed:")
                    
                    # Create a DataFrame for display
                    non_infl_df = pd.DataFrame(non_influential, columns=['Variable', 'Index', 'Effect'])
                    non_infl_df = non_infl_df.sort_values('Effect')
                    
                    # Calculate max effect for reference
                    max_effect = max(results['mean_abs_effects'])
                    
                    # Create an enhanced DataFrame with more information
                    enhanced_df = pd.DataFrame({
                        'Variable': non_infl_df['Variable'],
                        'Effect Value': non_infl_df['Effect'].round(6),
                        'Relative Effect (%)': (non_infl_df['Effect'] / max_effect * 100).round(2)
                    })
                    
                    # Display enhanced table
                    st.dataframe(enhanced_df, use_container_width=True)
                    
                    # Create a visual representation of the effects
                    st.write("Visual Comparison of Non-influential Variables")
                    
                    # Create a horizontal bar chart with Plotly for the non-influential variables
                    fig_non_infl = px.bar(
                        enhanced_df.sort_values('Effect Value', ascending=True), 
                        x='Effect Value',
                        y='Variable',
                        orientation='h',
                        text='Relative Effect (%)',
                        color='Effect Value',
                        color_continuous_scale='Blues',
                        labels={'Effect Value': 'Morris Effect', 'Variable': 'Input Variables'},
                        title='Relative Influence of Non-influential Variables'
                    )
                    
                    # Improve the layout
                    fig_non_infl.update_traces(
                        texttemplate='%{text}%',
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Effect: %{x:.6f}<br>Relative: %{text}%<extra></extra>'
                    )
                    
                    fig_non_infl.update_layout(
                        font=dict(size=12),
                        height=max(300, len(non_influential) * 50),  # Dynamic height based on number of variables
                        margin=dict(l=20, r=20, t=50, b=20),
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray', autorange="reversed")
                    )
                    
                    # Display the chart
                    st.plotly_chart(fig_non_infl, use_container_width=True)
                    
                    # Add contextual information
                    with st.expander("Understanding Morris Effects"):
                        st.write("""
                        ### Interpreting Morris Effects
                        
                        **What are Morris Effects?**  
                        Morris effects measure how much each input variable influences the model output. 
                        
                        **How to interpret the values:**
                        - **Effect Value**: The absolute mean elementary effect, which quantifies the average impact of changing the variable.
                        - **Relative Effect (%)**: The effect as a percentage of the maximum effect observed across all variables.
                        
                        **Decision making:**
                        - Variables with very low effects (below the threshold) can often be fixed at nominal values without significantly affecting model outputs.
                        - This simplifies the model by reducing its dimensionality, making it easier to understand and computationally more efficient.
                        """)
                    
                    # Get and display recommended fixed values
                    var_indices = [idx for _, idx, _ in non_influential]
                    recommendations = get_recommended_fixed_values(problem, var_indices)
                    
                    st.subheader("Recommended Fixed Values")
                    st.write("You can fix these variables at the following values in your model:")
                    st.dataframe(recommendations)
                    
                    # Add explanation with LLM if available
                    if language_model:
                        st.subheader("Model Simplification Insights")
                        
                        # Create a placeholder for the insights
                        insights_placeholder = st.empty()
                        
                        with st.spinner("Generating insights..."):
                            from utils.core_utils import call_groq_api
                            
                            # Prepare the prompt
                            prompt = f"""
                            Based on Morris sensitivity analysis, these variables have minimal impact on the model output:
                            {', '.join([name for name, _, _ in non_influential])}
                            
                            The model code is:
                            ```python
                            {code_snippet}
                            ```
                            
                            Explain in 2-3 paragraphs:
                            1. Why these variables might have low influence on this specific model
                            2. How the model could be simplified by fixing these variables
                            3. What physical or mathematical insights this provides about the system
                            """
                            
                            # Call the LLM
                            response = call_groq_api(prompt, model_name=language_model)
                            
                            # Display the response in the placeholder
                            insights_placeholder.write(response)
                else:
                    st.write("No non-influential variables were identified. All variables appear to have significant effects on the output.")
                    
                st.write("Morris analysis completed successfully!")
            except Exception as e:
                st.error(f"Error in Morris analysis: {str(e)}")
                st.warning("Please try with different parameters or check your model implementation.")

def dimensionality_reduction_page(current_code, model, problem, selected_language_model):
    """
    Main function to render the dimensionality reduction page in Streamlit.
    
    Parameters:
    -----------
    current_code : str
        The current model code
    model : ot.Function or None
        The OpenTURNS model function (if already loaded)
    problem : ot.JointDistribution or None
        The joint distribution of input variables (if already loaded)
    selected_language_model : str or None
        Name of the language model to use for explanations
    """
    st.subheader("Dimensionality Reduction with Morris Method")
    
    # First, check if we have a model from the code editor
    if not current_code and model is None:
        st.write("Please define your model in the Model Definition section first.")
        
        st.write("How to use the Morris Analysis:")
        st.write("""
        1. First, define or load your model in the **Model Definition** section
        2. Click the **Run Analysis** button to perform the Morris analysis
        
        The Morris method helps identify which input variables have minimal impact on your model's output.
        This allows you to create simplified models by fixing non-influential variables at nominal values.
        """)
    else:
        # If we have code but no model yet, try to execute it
        if current_code and model is None:
            try:
                # Execute the model code in a fresh namespace
                local_namespace = {}
                exec(current_code, local_namespace)
                
                # Check that both model and problem are defined
                if 'model' in local_namespace and 'problem' in local_namespace:
                    model = local_namespace['model']
                    problem = local_namespace['problem']
            except Exception as e:
                st.error(f"Error executing model code: {str(e)}")
        
        # Create a card for the configuration
        st.write("Morris Analysis Configuration")
        
        # Configuration controls
        col1, col2 = st.columns(2)
        with col1:
            n_trajectories = st.slider("Number of trajectories", min_value=5, max_value=50, value=10, 
                                      help="More trajectories provide better estimates but increase computation time")
        with col2:
            n_levels = st.slider("Number of levels", min_value=3, max_value=10, value=5,
                                help="Number of levels in the Morris grid design")
        
        threshold = st.slider("Threshold for non-influential variables (%)", min_value=1, max_value=20, value=5,
                             help="Variables with effects below this percentage of the maximum effect are considered non-influential")
        
        run_button = st.button("Run Morris Analysis", key="run_morris_analysis")
        st.write("")
        
        if run_button and model is not None and problem is not None:
            with st.spinner("Running Morris analysis..."):
                try:
                    # Run Morris analysis
                    results = run_morris_analysis(model, problem, n_trajectories, n_levels)
                    
                    # Plot results using Plotly
                    fig1, fig2 = plot_morris_results_plotly(results)
                    
                    # Display plots in styled sections
                    st.subheader("Morris Mean Absolute Elementary Effects")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    st.subheader("Morris Analysis: μ* vs σ")
                    st.write("Variables in the top-right corner have high influence and non-linear effects or interactions. Variables near the origin have low influence on the output.")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Identify non-influential variables
                    non_influential = identify_non_influential_variables(results, threshold)
                    
                    if non_influential:
                        st.subheader("Non-influential Variables")
                        st.write(f"The following variables have effects below {threshold}% of the maximum effect and can potentially be fixed:")
                        
                        # Create a DataFrame for display
                        non_infl_df = pd.DataFrame(non_influential, columns=['Variable', 'Index', 'Effect'])
                        non_infl_df = non_infl_df.sort_values('Effect')
                        
                        # Calculate max effect for reference
                        max_effect = max(results['mean_abs_effects'])
                        
                        # Create an enhanced DataFrame with more information
                        enhanced_df = pd.DataFrame({
                            'Variable': non_infl_df['Variable'],
                            'Effect Value': non_infl_df['Effect'].round(6),
                            'Relative Effect (%)': (non_infl_df['Effect'] / max_effect * 100).round(2)
                        })
                        
                        # Display enhanced table
                        st.dataframe(enhanced_df, use_container_width=True)
                        
                        # Create a visual representation of the effects
                        st.write("Visual Comparison of Non-influential Variables")
                        
                        # Create a horizontal bar chart with Plotly for the non-influential variables
                        fig_non_infl = px.bar(
                            enhanced_df.sort_values('Effect Value', ascending=True), 
                            x='Effect Value',
                            y='Variable',
                            orientation='h',
                            text='Relative Effect (%)',
                            color='Effect Value',
                            color_continuous_scale='Blues',
                            labels={'Effect Value': 'Morris Effect', 'Variable': 'Input Variables'},
                            title='Relative Influence of Non-influential Variables'
                        )
                        
                        # Improve the layout
                        fig_non_infl.update_traces(
                            texttemplate='%{text}%',
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Effect: %{x:.6f}<br>Relative: %{text}%<extra></extra>'
                        )
                        
                        fig_non_infl.update_layout(
                            font=dict(size=12),
                            height=max(300, len(non_influential) * 50),  # Dynamic height based on number of variables
                            margin=dict(l=20, r=20, t=50, b=20),
                            plot_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='lightgray'),
                            yaxis=dict(showgrid=True, gridcolor='lightgray', autorange="reversed")
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig_non_infl, use_container_width=True)
                        
                        # Add contextual information
                        with st.expander("Understanding Morris Effects"):
                            st.write("""
                            ### Interpreting Morris Effects
                            
                            **What are Morris Effects?**  
                            Morris effects measure how much each input variable influences the model output. 
                            
                            **How to interpret the values:**
                            - **Effect Value**: The absolute mean elementary effect, which quantifies the average impact of changing the variable.
                            - **Relative Effect (%)**: The effect as a percentage of the maximum effect observed across all variables.
                            
                            **Decision making:**
                            - Variables with very low effects (below the threshold) can often be fixed at nominal values without significantly affecting model outputs.
                            - This simplifies the model by reducing its dimensionality, making it easier to understand and computationally more efficient.
                            """)
                        
                        # Get and display recommended fixed values
                        var_indices = [idx for _, idx, _ in non_influential]
                        recommendations = get_recommended_fixed_values(problem, var_indices)
                        
                        st.subheader("Recommended Fixed Values")
                        st.write("You can fix these variables at the following values in your model:")
                        st.dataframe(recommendations)
                        
                        # Add explanation with LLM if available
                        if selected_language_model:
                            st.subheader("Model Simplification Insights")
                            
                            # Create a placeholder for the insights
                            insights_placeholder = st.empty()
                            
                            with st.spinner("Generating insights..."):
                                from utils.core_utils import call_groq_api
                                
                                # Prepare the prompt
                                prompt = f"""
                                Based on Morris sensitivity analysis, these variables have minimal impact on the model output:
                                {', '.join([name for name, _, _ in non_influential])}
                                
                                The model code is:
                                ```python
                                {current_code}
                                ```
                                
                                Explain in 2-3 paragraphs:
                                1. Why these variables might have low influence on this specific model
                                2. How the model could be simplified by fixing these variables
                                3. What physical or mathematical insights this provides about the system
                                """
                                
                                # Call the LLM
                                response = call_groq_api(prompt, model_name=selected_language_model)
                                
                                # Display the response in the placeholder
                                insights_placeholder.write(response)
                    else:
                        st.write("No non-influential variables were identified. All variables appear to have significant effects on the output.")
                        
                    st.write("Morris analysis completed successfully!")
                except Exception as e:
                    st.error(f"Error in Morris analysis: {str(e)}")
                    st.warning("Please try with different parameters or check your model implementation.")
