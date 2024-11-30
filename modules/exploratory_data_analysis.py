# modules/exploratory_data_analysis.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION
from matplotlib.gridspec import GridSpec

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

        # Ensure 'Y' column exists
        if 'Y' not in data.columns:
            raise KeyError("Column 'Y' is missing from the data.")

        variables = data.columns.drop('Y')  # Exclude 'Y' from variables
        num_vars = len(variables)
        colors = sns.color_palette('husl', num_vars)

        # Create a figure with num_vars + 1 rows and 2 columns using GridSpec
        fig = plt.figure(figsize=(12, (num_vars + 1) * 4), dpi=100)
        gs = GridSpec(num_vars + 1, 2, figure=fig)
        axes = []

        correlations = {}

        for i, var in enumerate(variables):
            # Histogram in axes[i, 0]
            ax_hist = fig.add_subplot(gs[i, 0])
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

            # Add legend with additional statistics
            stats_text = f"Std Dev: {std_dev:.2e}\nSkewness: {skewness:.2f}"
            ax_hist.legend(fontsize=10, loc='lower right', title=stats_text)

            # Scatter plot in axes[i, 1]
            ax_scatter = fig.add_subplot(gs[i, 1])
            sns.scatterplot(
                x=data[var],
                y=data['Y'],
                ax=ax_scatter,
                alpha=0.6,
                s=30,
                color='teal',
                edgecolor='white'
            )
            # Calculate correlation coefficient and p-value
            corr_coef, p_value = pearsonr(data[var], data['Y'])
            correlations[var] = corr_coef

            # Annotate correlation coefficient and significance
            significance = " (p < 0.05)" if p_value < 0.05 else ""
            ax_scatter.annotate(f'r = {corr_coef:.2f}{significance}',
                                xy=(0.65, 0.9), xycoords='axes fraction',
                                fontsize=12, color='darkred')

            # Add regression line
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

            # Highlight strong correlations
            if abs(corr_coef) > 0.5:
                ax_scatter.set_title(f'{var} vs Y (Strong Correlation)', fontsize=14, color='darkred')
            else:
                ax_scatter.set_title(f'{var} vs Y', fontsize=14)

            axes.append((ax_hist, ax_scatter))

        # Adjust layout and spacing
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle(f'Exploratory Data Analysis (N = {N})', fontsize=16, fontweight='bold')

        # Handle missing columns for correlation heatmap
        existing_columns = [col for col in variables if col in data.columns] + ['Y']
        if len(existing_columns) < len(variables) + 1:
            missing_columns = set(variables + ['Y']) - set(data.columns)
            print(f"Warning: The following columns are missing from the data: {missing_columns}")

        # Generate correlation matrix with existing columns
        correlation_matrix = data[existing_columns].corr()

        # Add correlation heatmap in the last row, spanning both columns
        heatmap_ax = fig.add_subplot(gs[num_vars, :])
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, ax=heatmap_ax)
        heatmap_ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        # Rotate x-axis labels if necessary
        heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), rotation=45, ha='right')

        # Adjust layout again after adding heatmap
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # --- API Call for Interpretation ---

        # Prepare the data for the prompt
        corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with Y'])
        corr_df.index.name = 'Variable'

        # Convert the correlation DataFrame to markdown table
        corr_md_table = corr_df.reset_index().to_markdown(index=False, floatfmt=".4f")

        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])

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
