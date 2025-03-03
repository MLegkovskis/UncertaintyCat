# modules/taylor_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
import re
from utils.core_utils import call_groq_api
from utils.markdown_utils import RETURN_INSTRUCTION
from utils.model_utils import get_ot_distribution, get_ot_model

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

    # Compute nominal value
    nominal_value = model(nominal_point)

    # Compute gradients
    gradients = np.zeros(dimension)
    for i in range(dimension):
        perturbed_point = nominal_point.copy()
        perturbed_point[i] += h
        gradients[i] = (model(perturbed_point) - nominal_value) / h

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
        'input_names': input_names
    }

    return results

def plot_taylor_indices(results, figsize=(10, 6)):
    """Plot Taylor-based sensitivity indices."""
    sensitivity_indices = results['sensitivity_indices']
    input_names = results['input_names']

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(input_names))
    ax.bar(x, sensitivity_indices)
    ax.set_xticks(x)
    ax.set_xticklabels(input_names, rotation=45, ha='right')
    ax.set_ylabel('Taylor-based Sensitivity Index')
    ax.set_title('Taylor-based Sensitivity Analysis')
    plt.tight_layout()

    return fig

def create_taylor_dataframe(results):
    """Create a DataFrame with Taylor analysis results."""
    df = pd.DataFrame({
        'Variable': results['input_names'],
        'Nominal_Point': results['nominal_point'],
        'Gradient': results['gradients'],
        'Variance': results['variances'],
        'Sensitivity_Index': results['sensitivity_indices']
    })
    return df

def taylor_analysis(model, problem, model_code_str, language_model='groq'):
    """Perform Taylor analysis on the model."""
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

    # Plot the Taylor importance factors as a pie chart
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.Paired(np.arange(len(variable_names)))

    # Create the pie chart with black borders
    wedges, texts = ax.pie(
        importance_values,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.3, edgecolor='black')
    )

    # Create the legend labels with percentages
    legend_labels = [f'{var}: {val:.1f}%' for var, val in zip(variable_names, importance_values)]

    # Add a legend to the pie chart, positioned to the right of the pie
    ax.legend(
        wedges,
        legend_labels,
        title="Variables",
        loc="center left",  # Align legend to the left, placing it outside the plot
        fontsize=12,
        title_fontsize=14,
        bbox_to_anchor=(1, 0.5)  # Position the legend to the right of the pie chart
    )

    # Set title and equal aspect ratio
    ax.set_title('Taylor Expansion Importance Factors', fontsize=16)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular

    plt.tight_layout()

    # Prepare the data for the API call
    taylor_md_table = taylor_df.to_markdown(index=False, floatfmt=".2f")

    # Use the provided model_code_str directly
    model_code = model_code_str

    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code.strip().split('\n')])

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

    # Prepare the prompt
    prompt = f"""
    {RETURN_INSTRUCTION}

    Given the following user-defined model defined in Python code:

    ```python
    {model_code_formatted}
    ```

    and the following uncertain input distributions:

    {inputs_description}

    The results of the Taylor expansion importance factors are given below (you must convert them to a table and show to the user):

    {taylor_df}

    Please:
      - Display the index values as a table. If the table is large (e.g., more than 10 rows), only show the top 8 rows corresponding to the highest-ranked inputs.
      - Explain the mathematical basis of the Taylor expansion method in sensitivity analysis.
      - Discuss how the importance factors are calculated using the derivatives of the model.
      - Interpret the results, focusing on which variables have the most significant impact on the output variance and why.
      - Provide insights into the physical or mathematical reasons behind the importance of these variables.
      - Reference the Taylor importance factors table in your discussion.
    """

    # Check if the results are already in session state
    if 'taylor_response_markdown' not in st.session_state:
        # Call the AI API
        response_markdown = call_groq_api(prompt, model_name=language_model)
        # Store the response in session state
        st.session_state.taylor_response_markdown = response_markdown
    else:
        response_markdown = st.session_state.taylor_response_markdown

    # Check if the figure is already in session state
    if 'taylor_fig' not in st.session_state:
        # Store the figure in session state
        st.session_state.taylor_fig = fig
    else:
        fig = st.session_state.taylor_fig

    # Display the results
    st.markdown(response_markdown)
    st.pyplot(fig)
