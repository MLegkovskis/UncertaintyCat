# modules/exploratory_data_analysis.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION

def exploratory_data_analysis(data, N, model, problem, model_code_str, language_model='groq'):
    response_key = 'eda_response_markdown'
    fig_key = 'eda_fig'

    # Check if the results are already in session state
    if response_key in st.session_state and fig_key in st.session_state:
        # Retrieve from session state
        response_markdown = st.session_state[response_key]
        fig = st.session_state[fig_key]
    else:
        # --- Perform Computations Only If Necessary ---

        variables = data.columns.drop('Y')  # Exclude 'Y' from variables
        num_vars = len(variables)
        colors = sns.color_palette('husl', num_vars)

        # Create a figure with num_vars rows and 2 columns
        fig, axes = plt.subplots(num_vars, 2, figsize=(10, num_vars * 4), dpi=100)
        # Ensure axes is a 2D array even if num_vars == 1
        if num_vars == 1:
            axes = np.array([axes])

        correlations = {}

        for i, var in enumerate(variables):
            # Histogram in axes[i, 0]
            ax_hist = axes[i, 0]
            sns.histplot(
                data[var],
                kde=True,
                color=colors[i],
                bins=50,
                ax=ax_hist,
                edgecolor='black'
            )
            ax_hist.set_title(f'Distribution of {var}', fontsize=14)
            ax_hist.set_xlabel(var, fontsize=12)
            ax_hist.set_ylabel('Frequency', fontsize=12)
            ax_hist.tick_params(axis='both', labelsize=12)

            # Calculate statistics
            mean = data[var].mean()
            median = data[var].median()
            std_dev = data[var].std()
            skewness = data[var].skew()

            # Add vertical lines for mean and median
            ax_hist.axvline(mean, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {mean:.2e}')
            ax_hist.axvline(median, color='green', linestyle='-.', linewidth=1.5, label=f'Median: {median:.2e}')

            # Annotate skewness
            ax_hist.text(0.95, 0.95, f'Skewness: {skewness:.2f}',
                         transform=ax_hist.transAxes,
                         fontsize=12,
                         verticalalignment='top',
                         horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # Add legend
            ax_hist.legend(fontsize=10)

            # Scatter plot in axes[i, 1]
            ax_scatter = axes[i, 1]
            sns.scatterplot(
                x=data[var],
                y=data['Y'],
                ax=ax_scatter,
                alpha=0.6,
                s=30,
                color='teal',
                edgecolor='white'
            )
            # Calculate correlation coefficient
            corr_coef = np.corrcoef(data[var], data['Y'])[0, 1]
            correlations[var] = corr_coef
            # Annotate correlation coefficient
            ax_scatter.annotate(f'r = {corr_coef:.2f}', xy=(0.05, 0.9), xycoords='axes fraction',
                                fontsize=12, color='darkred')

            sns.regplot(
                x=data[var],
                y=data['Y'],
                ax=ax_scatter,
                scatter=False,
                color='orange',
                line_kws={'linewidth': 2}
            )
            ax_scatter.set_xlabel(var, fontsize=12, fontweight='bold')
            ax_scatter.tick_params(axis='x', labelsize=12)
            ax_scatter.tick_params(axis='y', labelsize=12)

            if i == 0:
                ax_scatter.set_ylabel('Y', fontsize=12, fontweight='bold')
            else:
                ax_scatter.set_ylabel("")

        # Adjust layout and spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        fig.suptitle(f'Exploratory Data Analysis (N = {N})', fontsize=16, fontweight='bold')

        # --- API Call for Interpretation ---

        # Prepare the data for the prompt
        corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with Y'])
        corr_df.index.name = 'Variable'

        # Convert the correlation DataFrame to markdown table
        corr_md_table = corr_df.reset_index().to_markdown(index=False, floatfmt=".4f")

        # Use the provided model_code_str directly
        model_code = model_code_str

        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code.strip().split('\n')])

        # Prepare the input parameters and their uncertainties
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

    Given the following correlation coefficients between the input variables and the model output:

    {corr_md_table}

    Please:
      - Show the coefficients values as a table(s).
      - Provide a detailed interpretation of the main findings from this analysis.
      - Focus on the relationships between the variables and the model output.
    """

        # Call the AI API
        response_markdown = call_groq_api(prompt, model_name=language_model)

        # Store results in session_state
        st.session_state[response_key] = response_markdown
        st.session_state[fig_key] = fig

    # --- Display the Results ---

    st.markdown(response_markdown)
    st.pyplot(fig)
