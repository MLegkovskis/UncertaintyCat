# modules/exploratory_data_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr
import openturns as ot
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION

def exploratory_data_analysis(data, N, model, problem, model_code_str, language_model='groq'):
    """
    Perform exploratory data analysis on Monte Carlo simulation results.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing input variables and response
    N : int
        Number of samples
    model : callable
        Model function
    problem : dict or openturns.Distribution
        Problem definition containing input variable information
    model_code_str : str
        String containing the model code
    language_model : str, optional
        Language model to use for analysis
    """
    response_key = 'eda_response_markdown'
    
    if response_key in st.session_state:
        st.markdown(st.session_state[response_key])
    else:
        # Ensure 'Y' column exists
        if 'Y' not in data.columns:
            raise KeyError("Column 'Y' is missing from the data.")

        # Get input variable names based on problem type
        if isinstance(problem, (ot.Distribution, ot.JointDistribution)):
            dimension = problem.getDimension()
            variables = [problem.getMarginal(i).getDescription()[0] for i in range(dimension)]
            distributions = []
            for i in range(dimension):
                marginal = problem.getMarginal(i)
                distributions.append({
                    'type': marginal.__class__.__name__,
                    'params': list(marginal.getParameter())
                })
        else:
            variables = problem['names']
            distributions = problem['distributions']

        # Calculate correlations
        correlations = {}
        for var in variables:
            corr_coef, p_value = pearsonr(data[var], data['Y'])
            correlations[var] = (corr_coef, p_value)

        # Prepare correlation data
        correlations_df = pd.DataFrame({
            'Variable': list(correlations.keys()),
            'Correlation with Y': [v[0] for v in correlations.values()],
            'P-value': [v[1] for v in correlations.values()]
        })

        # Sort by absolute correlation
        correlations_df['Abs_Correlation'] = abs(correlations_df['Correlation with Y'])
        correlations_df = correlations_df.sort_values('Abs_Correlation', ascending=False)
        correlations_df = correlations_df.drop('Abs_Correlation', axis=1)

        # Basic statistics of the response
        response_stats = data['Y'].describe()
        st.write("### Y Variable Statistics")
        st.write(response_stats)

        # Distribution plot of the response
        st.write("### Y Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, x='Y', kde=True, ax=ax)
        plt.title('Distribution of Y Variable')
        st.pyplot(fig)
        plt.close()

        # Correlation analysis
        st.write("### Correlation Analysis")
        st.write(correlations_df)

        # Plot correlation bars
        fig, ax = plt.subplots(figsize=(10, 6))
        correlations_df.plot(x='Variable', y='Correlation with Y', kind='bar', ax=ax)
        plt.title('Correlation with Y Variable')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Scatter plots for top correlated variables
        st.write("### Scatter Plots")
        top_vars = correlations_df['Variable'].head(3).tolist()
        
        for var in top_vars:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=data, x=var, y='Y', ax=ax)
            plt.title(f'{var} vs Y')
            st.pyplot(fig)
            plt.close()

        # Box plots for categorical variables (if any)
        categorical_vars = data[variables].select_dtypes(include=['object', 'category']).columns
        if len(categorical_vars) > 0:
            st.write("### Box Plots for Categorical Variables")
            for var in categorical_vars:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(data=data, x=var, y='Y', ax=ax)
                plt.title(f'Y Distribution by {var}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close()

        # Pair plot for selected variables
        st.write("### Pair Plot")
        vars_for_pair = top_vars + ['Y']
        fig = sns.pairplot(data[vars_for_pair], diag_kind='kde')
        st.pyplot(fig)
        plt.close()

        # Additional insights
        st.write("### Additional Insights")
        st.write(f"- The Y variable has a mean of {response_stats['mean']:.4f}")
        st.write(f"- The standard deviation is {response_stats['std']:.4f}")
        st.write(f"- The range is from {response_stats['min']:.4f} to {response_stats['max']:.4f}")
        
        # Identify most influential variables
        st.write("### Most Influential Variables")
        st.write("Variables with strongest correlation to the Y:")
        for _, row in correlations_df.head(3).iterrows():
            st.write(f"- {row['Variable']}: correlation = {row['Correlation with Y']:.4f} (p-value = {row['P-value']:.4f})")

        # Generate prompt for GPT
        distributions_info = []
        for name, dist_info in zip(variables, distributions):
            distributions_info.append(f"- **{name}**: {dist_info['type']} distribution with parameters {dist_info['params']}")

        inputs_description = '\n'.join(distributions_info)

        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])

        prompt = f"""
Analyze this Monte Carlo simulation with {N} samples:

```python
{model_code_formatted}
```

The model has the following uncertain input parameters:

{inputs_description}

Based on the exploratory data analysis:

1. The Y variable has:
   - Mean: {response_stats['mean']:.4f}
   - Standard deviation: {response_stats['std']:.4f}
   - Range: [{response_stats['min']:.4f}, {response_stats['max']:.4f}]

2. Correlation analysis shows:
{correlations_df.to_string()}

Please provide:
1. A technical interpretation of these results
2. Insights about which input parameters most strongly influence the output
3. Any potential concerns or limitations of the analysis
4. Recommendations for further investigation or model improvement

{RETURN_INSTRUCTION}
"""

        # Call GPT API and store response
        response = call_groq_api(prompt, language_model)
        st.session_state[response_key] = response

        # Display the response
        st.markdown(response)