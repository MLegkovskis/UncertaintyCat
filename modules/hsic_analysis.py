import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
from utils.core_utils import call_groq_api
from utils.markdown_utils import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model, sample_inputs

def hsic_analysis(model, problem, model_code_str, language_model='groq'):
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Get input names from distribution
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        input_names.append(marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}")

    # Compute HSIC indices
    hsic_results = compute_hsic_indices(model, problem, N=1000, seed=42)

    # Create DataFrame from the list of dictionaries
    hsic_df = create_hsic_dataframe(hsic_results)

    # Plot HSIC results
    fig = plot_hsic_indices(hsic_results, figsize=(10, 6))

    # Prepare data for the API call
    hsic_md_table = hsic_df.to_markdown(index=False, floatfmt=".4e")

    # Use the provided model_code_str directly
    model_code = model_code_str

    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code.strip().split('\n')])

    # Prepare the input parameters and their uncertainties
    input_parameters = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        name = marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}"
        input_parameters.append({
            'Variable': name,
            'Distribution': marginal.__class__.__name__,
            'Parameters': list(marginal.getParameter())
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

{hsic_df}

Please:
  - Tabulate all the HSIC data and present it in an effective way.
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

def compute_hsic_indices(model, problem, N=1000, seed=42):
    """Compute HSIC-based sensitivity indices."""
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")

    # Get input names from distribution
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        input_names.append(marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}")

    # Generate samples
    X = sample_inputs(N, problem, seed)
    Y = np.array([model(x) for x in X])

    # Compute kernel matrices
    K = np.zeros((N, N))
    L = np.zeros((N, N))
    H = np.eye(N) - np.ones((N, N)) / N

    # Compute output kernel matrix
    for i in range(N):
        for j in range(N):
            L[i, j] = np.exp(-0.5 * (Y[i] - Y[j])**2)

    # Center kernel matrices
    L = H @ L @ H

    # Compute HSIC indices
    hsic_indices = np.zeros(dimension)
    for i in range(dimension):
        # Compute input kernel matrix for variable i
        for k in range(N):
            for l in range(N):
                K[k, l] = np.exp(-0.5 * (X[k, i] - X[l, i])**2)
        
        # Center kernel matrix
        K = H @ K @ H
        
        # Compute HSIC
        hsic_indices[i] = np.trace(K @ L) / (N - 1)**2

    # Normalize indices
    total_hsic = np.sum(hsic_indices)
    normalized_indices = hsic_indices / total_hsic if total_hsic > 0 else np.zeros_like(hsic_indices)

    # Create results dictionary
    results = {
        'hsic_indices': hsic_indices,
        'normalized_indices': normalized_indices,
        'input_names': input_names
    }

    return results

def plot_hsic_indices(results, figsize=(10, 6)):
    """Plot HSIC-based sensitivity indices."""
    normalized_indices = results['normalized_indices']
    input_names = results['input_names']

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(input_names))
    ax.bar(x, normalized_indices)
    ax.set_xticks(x)
    ax.set_xticklabels(input_names, rotation=45, ha='right')
    ax.set_ylabel('Normalized HSIC Index')
    ax.set_title('HSIC-based Sensitivity Analysis')
    plt.tight_layout()

    return fig

def create_hsic_dataframe(results):
    """Create a DataFrame with HSIC analysis results."""
    df = pd.DataFrame({
        'Variable': results['input_names'],
        'HSIC_Index': results['hsic_indices'],
        'Normalized_Index': results['normalized_indices']
    })
    return df
