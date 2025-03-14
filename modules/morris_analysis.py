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
    
    # Add text labels for only the most important variables (top 3)
    top_vars = df.sort_values('Mean Absolute Effects', ascending=False).head(3)
    for _, row in top_vars.iterrows():
        fig2.add_annotation(
            x=row['Mean Absolute Effects'],
            y=row['Standard Deviation'],
            text=row['Variable'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
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
                
                # Display plots
                st.subheader("Morris Mean Absolute Elementary Effects")
                st.plotly_chart(fig1, use_container_width=True)
                
                st.subheader("Morris Analysis: μ* vs σ")
                st.write("Variables in the top-right corner have high influence and non-linear effects or interactions.")
                st.write("Variables near the origin have low influence on the output.")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Identify non-influential variables
                non_influential = identify_non_influential_variables(results, threshold)
                
                if non_influential:
                    st.subheader("Non-influential Variables")
                    st.write(f"The following variables have effects below {threshold}% of the maximum effect and can potentially be fixed:")
                    
                    # Create a DataFrame for display
                    non_infl_df = pd.DataFrame(non_influential, columns=['Variable', 'Index', 'Effect'])
                    non_infl_df = non_infl_df.sort_values('Effect')
                    
                    # Display non-influential variables
                    st.dataframe(non_infl_df[['Variable', 'Effect']])
                    
                    # Get and display recommended fixed values
                    var_indices = [idx for _, idx, _ in non_influential]
                    recommendations = get_recommended_fixed_values(problem, var_indices)
                    
                    st.subheader("Recommended Fixed Values")
                    st.write("You can fix these variables at the following values in your model:")
                    st.dataframe(recommendations)
                    
                else:
                    st.info("No non-influential variables were identified. All variables appear to have significant effects on the output.")
            except Exception as e:
                st.error(f"Error in Morris analysis: {str(e)}")
                st.error("Please try with different parameters or check your model implementation.")
