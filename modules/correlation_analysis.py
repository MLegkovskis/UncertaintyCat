import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION

def correlation_analysis(model, problem, model_code_str, language_model='groq'):
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
            marginals.append(ot.ParametrizedDistribution(ot.LogNormalMuSigma(mu, sigma, gamma)))
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
    distribution = ot.ComposedDistribution(marginals)
    distribution.setDescription(problem['names'])

    # Create OpenTURNS model
    ot_model = ot.PythonFunction(problem['num_vars'], 1, model)

    # Generate samples
    n = 1000  # Number of samples for correlation analysis
    sampleX = distribution.getSample(n)
    sampleY = ot_model(sampleX)

    # Perform correlation analysis
    corr_analysis = ot.CorrelationAnalysis(sampleX, sampleY)
    methods = {
        "PCC": list(corr_analysis.computePCC()),
        "PRCC": list(corr_analysis.computePRCC()),
        "SRC": list(corr_analysis.computeSRC()),
        "SRRC": list(corr_analysis.computeSRRC()),
        "Pearson": list(corr_analysis.computePearsonCorrelation()),
        "Spearman": list(corr_analysis.computeSpearmanCorrelation()),
    }

    # Prepare DataFrame
    df = pd.DataFrame(methods, index=problem['names'])

    # Plot correlation coefficients
    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    df.plot(kind='bar', ax=ax)
    ax.set_title('Correlation Coefficients')
    ax.set_xlabel('Variable')
    ax.set_ylabel('Coefficient Value')
    ax.set_xticklabels(problem['names'], rotation=45, ha='right')
    ax.legend(title='Method')
    plt.tight_layout()


    # Prepare data for the prompt
    correlation_md_table = df.reset_index().rename(columns={'index': 'Variable'})

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

{inputs_md_table}

The results of the correlation analysis are given below (you must convert them to a table and show to the user):

{correlation_md_table}

Please:
  - Display the index values as a table. If the table is large (e.g., more than 10 rows), only show the top 8 rows corresponding to the highest-ranked inputs.
  - Include mathematical definitions for PCC, PRCC, Spearman, SRC, and SRRC, explaining what they represent.
  - Discuss key findings and consistency or inconsistency in the sensitivity predictions across the coefficients.
  - Provide insights into which variables are most influential according to the different methods.
  - Explain any discrepancies between the methods and what they might indicate about the relationships between the input variables and the output.
"""

    # Unique keys for session state
    response_key = 'correlation_response_markdown'
    fig_key = 'correlation_fig'

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

    # Optionally, return the DataFrame if needed
    return df