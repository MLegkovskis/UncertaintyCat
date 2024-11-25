# modules/taylor_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION

def taylor_analysis(model, problem, model_code_str, language_model='groq'):
    input_names = problem['names']

    # Create distributions
    marginals = []
    for dist_info in problem['distributions']:
        dist_type = dist_info['type']
        params = dist_info['params']
        if dist_type == 'Uniform':
            a, b = params
            marginals.append(ot.Uniform(a, b))
        elif dist_type == 'Normal':
            mu, sigma = params
            marginals.append(ot.Normal(mu, sigma))
        elif dist_type == 'Gumbel':
            beta_param, gamma_param = params
            marginals.append(ot.Gumbel(beta_param, gamma_param))
        elif dist_type == 'Triangular':
            a, m, b = params
            marginals.append(ot.Triangular(a, m, b))
        elif dist_type == 'Beta':
            alpha, beta_value, a, b = params
            marginals.append(ot.Beta(alpha, beta_value, a, b))
        elif dist_type == 'LogNormal':
            mu, sigma, gamma = params
            marginals.append(ot.LogNormal(mu, sigma, gamma))
        elif dist_type == 'LogNormalMuSigma':
            mu, sigma, gamma = params
            # Wrap the LogNormalMuSigma in ParametrizedDistribution
            marginals.append(ot.ParametrizedDistribution(ot.LogNormalMuSigma(mu, sigma, gamma)))
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    # Create the composed distribution
    distribution = ot.ComposedDistribution(marginals)
    distribution.setDescription(input_names)

    # Create OpenTURNS model
    ot_model = ot.PythonFunction(problem['num_vars'], 1, model)

    # Define the output as the result of the model applied to the input vector
    input_vector = ot.RandomVector(distribution)
    output_vector = ot.CompositeRandomVector(ot_model, input_vector)

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
    for name, dist_info in zip(problem['names'], problem['distributions']):
        input_parameters.append(f"- **{name}**: {dist_info['type']} distribution with parameters {dist_info['params']}")

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

    The results of the Taylor expansion importance factors are given below:

    {taylor_md_table}

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
