import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION

def hsic_analysis(model, problem, model_code_str, language_model='groq'):
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


    distribution = ot.ComposedDistribution(marginals)
    distribution.setDescription(problem['names'])

    # Create the OpenTURNS model
    ot_model = ot.PythonFunction(problem['num_vars'], 1, model)

    # Define sample size for HSIC analysis
    sample_size = 250
    input_design = distribution.getSample(sample_size)
    output_design = ot_model(input_design)

    # Define covariance models for HSIC analysis
    covariance_model_collection = []
    for i in range(problem['num_vars']):
        Xi = input_design.getMarginal(i)
        input_covariance = ot.SquaredExponential()
        sigma = Xi.computeStandardDeviation()[0]
        input_covariance.setScale([sigma])  # Wrap sigma in a list to create a Point
        covariance_model_collection.append(input_covariance)

    # Add output covariance model
    output_covariance = ot.SquaredExponential()
    sigma_Y = output_design.computeStandardDeviation()[0]
    output_covariance.setScale([sigma_Y])  # Wrap sigma_Y in a list to create a Point
    covariance_model_collection.append(output_covariance)

    # HSIC global sensitivity analysis
    estimator_type = ot.HSICUStat()
    hsic_algo = ot.HSICEstimatorGlobalSensitivity(covariance_model_collection, input_design, output_design, estimator_type)

    # Get HSIC results
    R2HSICIndices = hsic_algo.getR2HSICIndices()
    HSICIndices = hsic_algo.getHSICIndices()
    pvperm = hsic_algo.getPValuesPermutation()
    pvas = hsic_algo.getPValuesAsymptotic()

    hsic_results_list = []
    for i in range(len(problem['names'])):
        hsic_results_list.append({
            "Variable": problem['names'][i],
            "R2-HSIC Index": R2HSICIndices[i],
            "HSIC Index": HSICIndices[i],
            "p-value (permutation)": pvperm[i],
            "p-value (asymptotic)": pvas[i]
        })

    # Create DataFrame from the list of dictionaries
    hsic_results = pd.DataFrame(hsic_results_list)

    # Plot HSIC results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6), dpi=100)

    x = np.arange(len(problem['names']))
    ax1.bar(x, hsic_results['R2-HSIC Index'], color='orange', edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(problem['names'], rotation=45, ha='right', fontsize=8)
    ax1.set_title('R2-HSIC Indices', fontsize=14)
    ax1.set_ylabel('R2-HSIC', fontsize=12)

    ax2.bar(x, hsic_results['HSIC Index'], color='blue', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem['names'], rotation=45, ha='right', fontsize=8)
    ax2.set_title('HSIC Indices', fontsize=14)
    ax2.set_ylabel('HSIC', fontsize=12)

    width = 0.35
    ax3.bar(x - width/2, hsic_results['p-value (permutation)'], width, label='p-value (permutation)', color='red', edgecolor='black')
    ax3.bar(x + width/2, hsic_results['p-value (asymptotic)'], width, label='p-value (asymptotic)', color='green', edgecolor='black')
    ax3.set_xticks(x)
    ax3.set_xticklabels(problem['names'], rotation=45, ha='right', fontsize=8)
    ax3.set_title('HSIC p-values', fontsize=14)
    ax3.set_ylabel('p-value', fontsize=12)
    ax3.legend()

    plt.tight_layout()
    

    # Prepare data for the API call
    hsic_md_table = hsic_results.to_markdown(index=False, floatfmt=".4e")

    # Use the provided model_code_str directly
    model_code = model_code_str

    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code.strip().split('\n')])

    # Prepare the input parameters and their uncertainties
    input_parameters = []
    for name, dist_info in zip(problem['names'], problem['distributions']):
        input_parameters.append({
            'Variable': name,
            'Distribution': dist_info['type'],
            'Parameters': dist_info['params']
        })

    inputs_df = pd.DataFrame(input_parameters)
    inputs_md_table = inputs_df.to_markdown(index=False)

    # Prepare the prompt
    prompt = f"""
{RETURN_INSTRUCTION}

Given the following user-defined model defined in Python code:

```python
{model_code_formatted}
```

and the following uncertain input distributions:

{inputs_df}

The results of the HSIC analysis are given below:

{hsic_results}

Please:
  - Tabulate all the HSIC data and present it in an effecitve way.
  - Explain the mathematical basis of the HSIC method in sensitivity analysis.
  - Explain the Hilbert-Schmidt Independence Criterion (HSIC) and its use in measuring the dependence between input variables and the output.
  - Discuss the normalized R2-HSIC indices, indicating the proportion of the output variance explained by each input.
  - Explain the raw HSIC values, measuring the absolute dependence between each input and the output.
  - Interpret the p-values, determining the statistical significance of the HSIC indices.
  - Analyze the data and convey your observations, highlighting which variables are most influential and any notable findings.
  - Reference the HSIC results table in your discussion.
"""

    # Unique keys for session state
    response_key = 'hsic_response_markdown'
    fig_key = 'hsic_fig'

    # Check if the results are already in session state
    if response_key not in st.session_state:
        response_markdown = call_groq_api(prompt, model_name=language_model)
        st.session_state[response_key] = response_markdown
    else:
        response_markdown = st.session_state[response_key]

    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    # Display the results
    st.markdown(response_markdown)
    st.pyplot(fig)