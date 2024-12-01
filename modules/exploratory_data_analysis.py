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
    data_key = 'eda_data'

    # Check if the results are already in session state
    if response_key in st.session_state and data_key in st.session_state:
        # Retrieve from session state
        response_markdown = st.session_state[response_key]
        correlations = st.session_state[data_key]['correlations']
    else:
        # --- Perform Computations Only If Necessary ---

        # Ensure 'Y' column exists
        if 'Y' not in data.columns:
            raise KeyError("Column 'Y' is missing from the data.")

        variables = data.columns.drop('Y')  # Exclude 'Y' from variables
        correlations = {}

        # Calculate correlations
        for var in variables:
            corr_coef, p_value = pearsonr(data[var], data['Y'])
            correlations[var] = (corr_coef, p_value)

        # Prepare the data for the prompt
        corr_df = pd.DataFrame({
            'Variable': list(correlations.keys()),
            'Correlation with Y': [v[0] for v in correlations.values()],
            'P-value': [v[1] for v in correlations.values()]
        })

        # Convert the correlation DataFrame to markdown table
        corr_md_table = corr_df.to_markdown(index=False, floatfmt=".4f")

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
        st.session_state[data_key] = {
            'correlations': correlations,
            'variables': variables.tolist()
        }

    # Retrieve data from session_state
    variables = st.session_state[data_key]['variables']
    correlations = st.session_state[data_key]['correlations']

    # --- Generate Figures ---

    num_vars = len(variables)
    colors = sns.color_palette('husl', num_vars)

    # Set Seaborn theme for a professional look
    sns.set_theme(style="whitegrid")

    # Adjust figure size and grid layout
    fig_width = 15
    fig_height = num_vars * 4
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
    gs = GridSpec(num_vars, 2, figure=fig, width_ratios=[1, 1])

    for i, var in enumerate(variables):
        # Histogram
        ax_hist = fig.add_subplot(gs[i, 0])
        sns.histplot(
            data[var],
            kde=True,
            color=colors[i],
            bins=50,
            ax=ax_hist,
            edgecolor='black'
        )
        ax_hist.set_title(f'Distribution of {var}', fontsize=12)
        ax_hist.set_xlabel(var, fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        ax_hist.tick_params(axis='both', labelsize=10)

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
        ax_hist.legend(fontsize=8, loc='upper right', title=stats_text)

        # Scatter plot
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
        # Annotate correlation coefficient
        corr_coef, p_value = correlations[var]
        significance = " (p < 0.05)" if p_value < 0.05 else ""
        ax_scatter.annotate(f'r = {corr_coef:.2f}{significance}',
                            xy=(0.05, 0.9), xycoords='axes fraction',
                            fontsize=10, color='darkred')

        # Add regression line
        sns.regplot(
            x=data[var],
            y=data['Y'],
            ax=ax_scatter,
            scatter=False,
            color='orange',
            line_kws={'linewidth': 2}
        )
        ax_scatter.set_xlabel(var, fontsize=10, fontweight='bold')
        ax_scatter.tick_params(axis='x', labelsize=10)
        ax_scatter.tick_params(axis='y', labelsize=10)

        if i == 0:
            ax_scatter.set_ylabel('Y', fontsize=10, fontweight='bold')
        else:
            ax_scatter.set_ylabel("")

        # Highlight strong correlations
        if abs(corr_coef) > 0.5:
            ax_scatter.set_title(f'{var} vs Y (Strong Correlation)', fontsize=12, color='darkred')
        else:
            ax_scatter.set_title(f'{var} vs Y', fontsize=12)

    # Adjust layout and spacing
    plt.subplots_adjust(top=0.95, hspace=0.4)

    # Generate correlation matrix
    existing_columns = variables + ['Y']
    correlation_matrix = data[existing_columns].corr()

    # Correlation Clustermap
    clustermap_fig = sns.clustermap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        figsize=(12, 10),
        annot_kws={"size": 12},
        linewidths=.5,
        row_cluster=False,
        col_cluster=False
    )
    clustermap_fig.cax.set_visible(False)
    clustermap_fig.ax_heatmap.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.setp(clustermap_fig.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(clustermap_fig.ax_heatmap.get_yticklabels(), rotation=0, fontsize=12)

    # Remove dendrogram axes
    clustermap_fig.ax_row_dendrogram.set_visible(False)
    clustermap_fig.ax_col_dendrogram.set_visible(False)

    # --- Display the Results ---
    st.markdown(response_markdown)
    st.pyplot(fig)
    st.pyplot(clustermap_fig.fig)
