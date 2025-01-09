import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st  # Added import statement
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION
from modules.statistical_utils import (
    plot_sobol_radial,
    describe_radial_plot
)
import openturns as ot
from modules.openturns_utils import get_ot_distribution, get_ot_model, ot_point_to_list


def sobol_sensitivity_analysis(
    N,
    model,
    problem,
    model_code_str,
    sensitivity_threshold=0.05,
    confidence_level=0.95,
    language_model="groq",
    verbose=True,
):
    # Get the input distribution and the model
    distribution = get_ot_distribution(problem)
    model_g = get_ot_model(model, problem)

    # Compute indices
    computeSecondOrder = True
    sie = ot.SobolIndicesExperiment(distribution, N, computeSecondOrder)
    inputDesignSobol = sie.generate()
    inputNames = distribution.getDescription()
    inputDesignSobol.setDescription(inputNames)
    inputDesignSobol.getSize()
    outputDesignSobol = model_g(inputDesignSobol)

    # %%
    sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(
        inputDesignSobol, outputDesignSobol, N
    )
    sensitivityAnalysis.setConfidenceLevel(confidence_level)
    S1 = sensitivityAnalysis.getFirstOrderIndices()
    ST = sensitivityAnalysis.getTotalOrderIndices()
    S2 = sensitivityAnalysis.getSecondOrderIndices()
    S1 = ot_point_to_list(S1)
    ST = ot_point_to_list(ST)
    S2 = np.array(S2)
    # Confidence intervals (supposed to be approximately symmetric)
    dimension = distribution.getDimension()
    S1_interval = sensitivityAnalysis.getFirstOrderIndicesInterval()
    lower_bound = S1_interval.getLowerBound()
    upper_bound = S1_interval.getUpperBound()
    S1_conf = [(upper_bound[i] - lower_bound[i]) / 2.0 for i in range(dimension)]
    ST_interval = sensitivityAnalysis.getTotalOrderIndicesInterval()
    lower_bound = ST_interval.getLowerBound()
    upper_bound = ST_interval.getUpperBound()
    ST_conf = [(upper_bound[i] - lower_bound[i]) / 2.0 for i in range(dimension)]

    # Perform Sobol analysis
    Si = {
        "S1": np.array(S1),
        "ST": np.array(ST),
        "S2": np.array(S2),
        "S1_conf": S1_conf,
        "ST_conf": ST_conf,
    }

    # Create DataFrame for indices
    Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
    Si_df = pd.DataFrame(Si_filter, index=problem["names"])

    # Plotting combined Sobol indices
    fig = plt.figure(figsize=(16, 6))

    # Bar plot of Sobol indices
    ax1 = fig.add_subplot(1, 2, 1)
    indices = Si_df[["S1", "ST"]]
    err = Si_df[["S1_conf", "ST_conf"]]
    indices.plot.bar(yerr=err.values.T, capsize=5, ax=ax1)
    ax1.set_title(f"Sobol Sensitivity Indices (N = {N})")
    ax1.set_ylabel("Sensitivity Index")
    ax1.set_xlabel("Input Variables")
    ax1.legend(["First-order", "Total-order"])

    # Radial plot
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    names = problem["names"]
    plot_sobol_radial(names, Si, ax2)

    plt.tight_layout()

    # Prepare data for the API call
    first_order_df = pd.DataFrame(
        {
            "Variable": problem["names"],
            "First-order Sobol Index": Si["S1"],
            "Confidence Interval": Si["S1_conf"],
        }
    )

    total_order_df = pd.DataFrame(
        {
            "Variable": problem["names"],
            "Total-order Sobol Index": Si["ST"],
            "Confidence Interval": Si["ST_conf"],
        }
    )

    # Process S2 indices
    S2_indices = []
    variable_names = problem["names"]
    for i in range(len(variable_names)):
        for j in range(i + 1, len(variable_names)):
            idx_i = i
            idx_j = j
            S2_value = Si["S2"][idx_i, idx_j]
            if not np.isnan(S2_value) and abs(S2_value) > 0.01:
                S2_indices.append(
                    {
                        "Variable 1": variable_names[idx_i],
                        "Variable 2": variable_names[idx_j],
                        "Second-order Sobol Index": S2_value,
                    }
                )

    S2_df = pd.DataFrame(S2_indices)

    # Convert DataFrames to Markdown tables
    if not S2_df.empty:
        second_order_md_table = S2_df
    else:
        second_order_md_table = "No significant second-order interactions detected."

    # Description of the radial plot with numerical data
    radial_plot_description = describe_radial_plot(Si, variable_names, sensitivity_threshold)

    # Use the provided model_code_str directly
    model_code = model_code_str

    # Format the model code for inclusion in the prompt
    model_code_formatted = "\n".join(
        ["    " + line for line in model_code.strip().split("\n")]
    )

    # Prepare the inputs description
    input_parameters = []
    for name, dist_info in zip(problem["names"], problem["distributions"]):
        input_parameters.append(
            f"- **{name}**: {dist_info['type']} distribution with parameters {dist_info['params']}"
        )

    inputs_description = "\n".join(input_parameters)

    # Prepare the prompt
    prompt = f"""
{RETURN_INSTRUCTION}

Given the following user-defined model defined in Python code:

```python
{model_code_formatted}
```

and the following uncertain input distributions:

{inputs_description}

Given the following first-order Sobol' indices and their confidence intervals:

{first_order_df}

And the following total-order Sobol' indices and their confidence intervals:

{total_order_df}

The following second-order Sobol' indices were identified:

{second_order_md_table}

Given a description of the Sobol Indices Radial Plot data:

{radial_plot_description}

Please:
  - There are three classes of Sobol' indices: 1) any index lower than 0.05 can be considered weak, 2) any index in [0.05, 0.2] can be considered moderate, 3) any index larger than 0.2 can be considered strong.
  - Display all the index values as separate tables (if the tables are big - feel free to show only top 10 ranked inputs).
  - Briefly explain the Sobol method and the difference between first-order and total-order indices in terms of their mathematics and what they represent.
  - Explain the significance of high-impact Sobol' indices and the importance of the corresponding input variables from both mathematical and physical perspectives.
  - Discuss the confidence intervals associated with the Sobol' indices and what they represent.
  - Provide an interpretation of the Sobol Indices Radial Plot based on the description and numerical data.
  - Reference the Sobol indices tables in your discussion.
"""
    if verbose:
        print("prompt")
        print(prompt)

    response_key = "sobol_response_markdown"
    fig_key = "sobol_fig"

    if response_key not in st.session_state:
        response_markdown = call_groq_api(prompt, model_name=language_model)
        st.session_state[response_key] = response_markdown
    else:
        response_markdown = st.session_state[response_key]

    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    st.markdown(response_markdown)
    st.pyplot(fig)
