import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
import streamlit as st  # Added import statement
from math import pi
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION
from modules.statistical_utils import get_bounds_for_salib



def plot_sobol_radial(Si, problem, ax):
    """Plot Sobol indices on a radial plot."""
    names = problem['names']
    n = len(names)
    ticklocs = np.linspace(0, 2 * pi, n, endpoint=False)
    locs = ticklocs

    # Get indices
    ST = np.abs(Si['ST'])
    S1 = np.abs(Si['S1'])
    S2 = Si['S2']
    names = np.array(names)

    # Filter out insignificant indices
    threshold = 0.01
    significant = ST > threshold
    filtered_names = names[significant]
    filtered_locs = locs[significant]
    ST = ST[significant]
    S1 = S1[significant]

    # Prepare S2 matrix
    S2_matrix = np.zeros((len(filtered_names), len(filtered_names)))
    for i in range(len(filtered_names)):
        for j in range(i+1, len(filtered_names)):
            idx_i = np.where(names == filtered_names[i])[0][0]
            idx_j = np.where(names == filtered_names[j])[0][0]
            S2_value = Si['S2'][idx_i, idx_j]
            if np.isnan(S2_value) or abs(S2_value) < threshold:
                S2_value = 0
            S2_matrix[i, j] = S2_value

    # Plotting
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks(filtered_locs)
    ax.set_xticklabels(filtered_names)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.5)

    # Plot ST and S1 using ax.scatter
    max_marker_size = 5000  # Adjust as needed
    smax = max(np.max(ST), np.max(S1))
    smin = min(np.min(ST), np.min(S1))

    def normalize(x, xmin, xmax):
        return (x - xmin) / (xmax - xmin) if xmax - xmin != 0 else 0

    # First plot the ST circles (outer circles)
    for loc, st_val in zip(filtered_locs, ST):
        size = 50 + max_marker_size * normalize(st_val, smin, smax)
        ax.scatter(loc, 1, s=size, c='white', edgecolors='black', zorder=2)

    # Then plot the S1 circles (inner circles)
    for loc, s1_val in zip(filtered_locs, S1):
        size = 50 + max_marker_size * normalize(s1_val, smin, smax)
        ax.scatter(loc, 1, s=size, c='black', edgecolors='black', zorder=3)

    # Plot S2 interactions
    s2max = np.max(S2_matrix)
    s2min = np.min(S2_matrix[S2_matrix > 0]) if np.any(S2_matrix > 0) else 0
    for i in range(len(filtered_names)):
        for j in range(i+1, len(filtered_names)):
            weight = S2_matrix[i, j]
            if weight > 0:
                lw = 0.5 + 5 * normalize(weight, s2min, s2max)
                ax.plot([filtered_locs[i], filtered_locs[j]], [1,1], c='darkgray', lw=lw, zorder=1)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='ST', markerfacecolor='white', markeredgecolor='black', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='S1', markerfacecolor='black', markeredgecolor='black', markersize=15),
        Line2D([0], [0], color='darkgray', lw=3, label='S2')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title("Sobol Indices Radial Plot")








def sobol_sensitivity_analysis(N, model, problem, model_code_str, language_model='groq'):
    # Ensure N is a power of 2
    N_power = int(np.ceil(np.log2(N)))
    N = int(2 ** N_power)

    # Generate bounds for SALib
    problem_for_salib = get_bounds_for_salib(problem)

    # Generate samples using Sobol' sequence
    param_values = sobol.sample(problem_for_salib, N, calc_second_order=True)

    # Evaluate the model
    Y = np.array([model(X) for X in param_values]).flatten()

    # Perform Sobol analysis
    Si = sobol_analyze.analyze(problem_for_salib, Y, calc_second_order=True, print_to_console=False)

    # Create DataFrame for indices
    Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
    Si_df = pd.DataFrame(Si_filter, index=problem['names'])

    # Plotting combined Sobol indices
    fig = plt.figure(figsize=(16,6))

    # Bar plot of Sobol indices
    ax1 = fig.add_subplot(1, 2, 1)
    indices = Si_df[['S1', 'ST']]
    err = Si_df[['S1_conf', 'ST_conf']]
    indices.plot.bar(yerr=err.values.T, capsize=5, ax=ax1)
    ax1.set_title(f"Sobol Sensitivity Indices (N = {N})")
    ax1.set_ylabel('Sensitivity Index')
    ax1.set_xlabel('Input Variables')
    ax1.legend(['First-order', 'Total-order'])

    # Radial plot
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    plot_sobol_radial(Si, problem, ax=ax2)

    plt.tight_layout()
    

    # Prepare data for the API call
    first_order_df = pd.DataFrame({
        'Variable': problem['names'],
        'First-order Sobol Index': Si['S1'],
        'Confidence Interval': Si['S1_conf']
    })

    total_order_df = pd.DataFrame({
        'Variable': problem['names'],
        'Total-order Sobol Index': Si['ST'],
        'Confidence Interval': Si['ST_conf']
    })

    # Process S2 indices
    S2_indices = []
    variable_names = problem['names']
    for i in range(len(variable_names)):
        for j in range(i+1, len(variable_names)):
            idx_i = i
            idx_j = j
            S2_value = Si['S2'][idx_i, idx_j]
            if not np.isnan(S2_value) and abs(S2_value) > 0.01:
                S2_indices.append({
                    'Variable 1': variable_names[idx_i],
                    'Variable 2': variable_names[idx_j],
                    'Second-order Sobol Index': S2_value
                })

    S2_df = pd.DataFrame(S2_indices)

    # Convert DataFrames to Markdown tables
    first_order_md_table = first_order_df.to_markdown(index=False, floatfmt=".4f")
    total_order_md_table = total_order_df.to_markdown(index=False, floatfmt=".4f")
    if not S2_df.empty:
        second_order_md_table = S2_df.to_markdown(index=False, floatfmt=".4f")
    else:
        second_order_md_table = "No significant second-order interactions detected."

    # Prepare data placeholders for radial plot interpretation
    radial_data = ""
    for idx, name in enumerate(variable_names):
        s1 = Si['S1'][idx]
        st_value = Si['ST'][idx]  # Renamed variable
        radial_data += f"- Variable **{name}**: S1 = {s1:.4f}, ST = {st_value:.4f}\n"

    if not S2_df.empty:
        radial_data += "\nSignificant second-order interactions:\n"
        for _, row in S2_df.iterrows():
            var1 = row['Variable 1']
            var2 = row['Variable 2']
            s2 = row['Second-order Sobol Index']
            radial_data += f"- Interaction between **{var1}** and **{var2}**: S2 = {s2:.4f}\n"
    else:
        radial_data += "\nNo significant second-order interactions detected."

    # Description of the radial plot with numerical data
    radial_plot_description = f"""
    The Sobol Indices Radial Plot is a polar plot where each input variable is placed at equal angular intervals around a circle. The elements of the plot are:

    - **Variables**: Each input variable is positioned at a specific angle on the circle, equally spaced from others.

    - **Circles**:
        - The **outer circle** (white fill) represents the **total-order Sobol' index (ST)** for each variable.
        - The **inner circle** (black fill) represents the **first-order Sobol' index (S1)**.
        - The **size of the circles** is proportional to the magnitude of the respective Sobol' indices.

    - **Lines**:
        - Lines connecting variables represent **second-order Sobol' indices (S2)**.
        - The **thickness of the lines** corresponds to the magnitude of the interaction between the two variables; thicker lines indicate stronger interactions.

    Numerical data for the plot:
    {radial_data}

    This plot visually conveys both the individual effects of variables and their interactions, aiding in understanding the model's sensitivity to input uncertainties.
    """

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

Given the following first-order Sobol' indices and their confidence intervals:

{first_order_md_table}

And the following total-order Sobol' indices and their confidence intervals:

{total_order_md_table}

The following second-order Sobol' indices were identified:

{second_order_md_table}

An interpretation of the Sobol Indices Radial Plot is provided:

{radial_plot_description}

Please:
  - Display the index values as a table. If the table is large (e.g., more than 10 rows), only show the top 8 rows corresponding to the highest-ranked inputs.
  - Explain the mathematical basis of the Sobol method in sensitivity analysis.
  - Briefly explain the difference between first-order and total-order indices in terms of their mathematics and what they represent.
  - Explain the significance of high-impact Sobol' indices and the importance of the corresponding input variables from both mathematical and physical perspectives.
  - Discuss the confidence intervals associated with the Sobol' indices and what they represent.
  - Provide an interpretation of the Sobol Indices Radial Plot based on the description and numerical data.
  - Reference the Sobol indices tables in your discussion.
"""

    response_key = 'sobol_response_markdown'
    fig_key = 'sobol_fig'

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



