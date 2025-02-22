# modules/ml_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st  # Import Streamlit
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION
import openturns as ot

# Import necessary libraries for ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
from scipy.stats import pearsonr

def ml_analysis(data, problem, model_code_str, language_model='groq'):
    # Handle OpenTURNS distribution
    if isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        dimension = problem.getDimension()
        feature_names = []
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            feature_names.append(marginal.getDescription()[0])
    else:
        feature_names = problem['names']
    
    # Model Fitting
    rf_model, X_test, y_test, scaler = model_fitting(data, problem)
    
    # SHAP Analysis
    shap_values, shap_summary_df = shap_analysis(rf_model, X_test, feature_names)
    
    # Prepare the prompt and call the API for interpretation
    prepare_and_call_prompt(
        data, problem, rf_model, feature_names, model_code_str, shap_summary_df, language_model=language_model
    )
    
    # Plotting SHAP summary plot
    plot_shap_summary(shap_values, X_test, feature_names)
    
    # Plotting dependence plots
    plot_dependence_plots(shap_values, shap_summary_df, X_test, feature_names)

def model_fitting(data, problem):
    # Check if the model is already in session_state
    if ('rf_model' in st.session_state and 'X_test' in st.session_state and
        'y_test' in st.session_state and 'scaler' in st.session_state):
        rf_model = st.session_state['rf_model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        scaler = st.session_state['scaler']
    else:
        # Handle OpenTURNS distribution
        if isinstance(problem, (ot.Distribution, ot.JointDistribution)):
            dimension = problem.getDimension()
            feature_names = []
            for i in range(dimension):
                marginal = problem.getMarginal(i)
                feature_names.append(marginal.getDescription()[0])
            X = data[feature_names]
        else:
            X = data[problem['names']]
        y = data['Y']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Fit Random Forest Regressor
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)

        # Store in session_state
        st.session_state['rf_model'] = rf_model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler

    return rf_model, X_test, y_test, scaler

def shap_analysis(model, X_test, feature_names):
    if ('shap_values' in st.session_state and 'shap_summary_df' in st.session_state):
        shap_values = st.session_state['shap_values']
        shap_summary_df = st.session_state['shap_summary_df']
    else:
        shap.initjs()
        
        # Ensure that X_test is a DataFrame with feature names
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create the explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
        shap_summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_SHAP_Value': mean_abs_shap_values
        }).sort_values(by='Mean_SHAP_Value', ascending=False).reset_index(drop=True)
        
        # Store in session_state
        st.session_state['shap_values'] = shap_values
        st.session_state['shap_summary_df'] = shap_summary_df

    return shap_values, shap_summary_df

def prepare_and_call_prompt(data, problem, rf_model, feature_names, model_code_str, shap_summary_df, language_model='groq'):
    # Check if the response is already in session_state
    response_key = 'ml_response_markdown'
    if response_key in st.session_state:
        response_markdown = st.session_state[response_key]
    else:
        # Extract performance metrics
        X = data[feature_names]
        y = data['Y']
        scaler = st.session_state['scaler']  # Access scaler from session state
        X_scaled = scaler.transform(X)

        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores_rf = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='r2')
        rf_cv_mean = scores_rf.mean()
        rf_cv_std = scores_rf.std()

        # Prepare DataFrame for model performance
        performance_df = pd.DataFrame({
            'Model': ['Random Forest'],
            'Cross-Validation R² Mean': [rf_cv_mean],
            'Cross-Validation R² Std': [rf_cv_std]
        })

        # Convert DataFrames to Markdown tables
        performance_md_table = performance_df.to_markdown(index=False, floatfmt=".4f")
        shap_md_table = shap_summary_df.to_markdown(index=False, floatfmt=".4f")

        # Use the provided model_code_str directly
        model_code = model_code_str

        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code.strip().split('\n')])

        # Prepare the inputs description
        input_parameters = []
        if isinstance(problem, (ot.Distribution, ot.JointDistribution)):
            dimension = problem.getDimension()
            for i in range(dimension):
                marginal = problem.getMarginal(i)
                input_parameters.append(f"- **{marginal.getDescription()[0]}**: {marginal.getName()}")
        else:
            for name, dist_info in zip(problem['names'], problem['distributions']):
                input_parameters.append(f"- **{name}**: {dist_info['type']} distribution with parameters {dist_info['params']}")

        inputs_description = '\n'.join(input_parameters)

        # Prepare the prompt
        prompt = f"""
        {RETURN_INSTRUCTION}

        ## SHAP Analysis

        Given the following user-defined model defined in Python code:

        ```python
        {model_code_formatted}
        ```

        and the following uncertain input distributions:

        {inputs_description}

        The performance metrics of the Random Forest model are as follows:

        {performance_md_table}

        The mean absolute SHAP values for each feature are provided in the table below:

        {shap_md_table}

        Please:
        - Explain the purpose of SHAP analysis and its significance in interpreting machine learning models.
        - Discuss the performance of the Random Forest model based on the cross-validation R² scores.
        - Interpret the SHAP summary table, highlighting which input variables are most influential in predicting the output.
        - Provide insights into how the top features contribute to the model's predictions, including whether the relationship is positive or negative.
        - Reference the performance metrics and SHAP values in your discussion.  
        """

        # Call the API
        response_markdown = call_groq_api(prompt, model_name=language_model)
        st.session_state[response_key] = response_markdown

    st.markdown(response_markdown)

def plot_shap_summary(shap_values, X_test, feature_names):
    summary_fig_key = 'ml_shap_summary_fig'
    if summary_fig_key not in st.session_state:
        summary_fig = plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        st.session_state[summary_fig_key] = summary_fig
    else:
        summary_fig = st.session_state[summary_fig_key]
    st.pyplot(summary_fig)

def plot_dependence_plots(shap_values, shap_summary_df, X_test, feature_names):
    num_top_features = 4
    top_features = shap_summary_df['Feature'].values[:num_top_features]
    top_indices = [feature_names.index(feat) for feat in top_features]
    
    dependence_fig_key = 'ml_dependence_fig'
    if dependence_fig_key not in st.session_state:
        # Generate dependence plots
        cols = 2  # Number of columns in the grid
        rows = int(np.ceil(num_top_features / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), dpi=100)
        axes = axes.flatten()
        
        for i, idx in enumerate(top_indices):
            feature = feature_names[idx]
            shap.dependence_plot(
                idx, shap_values, X_test,
                feature_names=feature_names, show=False, ax=axes[i]
            )
            axes[i].set_title(f'Dependence of {feature}', fontsize=14)
            axes[i].tick_params(axis='both', labelsize=12)
        
        # Remove any unused subplots
        if len(axes) > num_top_features:
            for j in range(num_top_features, len(axes)):
                fig.delaxes(axes[j])
        
        plt.tight_layout()
        st.session_state[dependence_fig_key] = fig
    else:
        fig = st.session_state[dependence_fig_key]
    
    st.pyplot(fig)