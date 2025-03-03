# modules/ml_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import shap
from scipy.stats import pearsonr
from utils.core_utils import call_groq_api
from utils.markdown_utils import RETURN_INSTRUCTION

def ml_analysis(data, problem, model_code_str, language_model='groq'):
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    dimension = problem.getDimension()
    feature_names = []
    
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        name = marginal.getDescription()[0]
        feature_names.append(name if name != "" else f"X{i+1}")
    
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
        # Ensure problem is an OpenTURNS distribution
        if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
            raise ValueError("Problem must be an OpenTURNS distribution")
        
        dimension = problem.getDimension()
        feature_names = []
        
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = marginal.getDescription()[0]
            feature_names.append(name if name != "" else f"X{i+1}")
            
        X = data[feature_names]
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
    """Perform SHAP analysis on the trained model."""
    if ('shap_values' in st.session_state and 'shap_summary_df' in st.session_state):
        shap_values = st.session_state['shap_values']
        shap_summary_df = st.session_state['shap_summary_df']
    else:
        # Ensure X_test is a DataFrame with feature names
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create the explainer
        explainer = shap.TreeExplainer(model)
        st.session_state['shap_explainer'] = explainer
        
        # Calculate SHAP values
        shap_values = explainer(X_test_df)
        st.session_state['shap_values'] = shap_values
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
        
        # Create a DataFrame with feature names and mean absolute SHAP values
        shap_summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': mean_abs_shap_values
        })
        
        # Sort by mean absolute SHAP values in descending order
        shap_summary_df = shap_summary_df.sort_values('Mean |SHAP|', ascending=False)
        
        # Store in session_state
        st.session_state['shap_summary_df'] = shap_summary_df
    
    return shap_values, shap_summary_df

def prepare_and_call_prompt(data, problem, rf_model, feature_names, model_code_str, shap_summary_df, language_model='groq'):
    """Prepare the prompt and call the API for interpretation."""
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Check if the interpretation is already in session_state
    if 'ml_interpretation' in st.session_state:
        st.markdown(st.session_state['ml_interpretation'])
    else:
        # Format the model code for inclusion in the prompt
        model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])
        
        # Calculate model performance metrics
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        y_pred = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate feature importances
        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Calculate correlations
        correlations = []
        for feature in feature_names:
            corr, _ = pearsonr(data[feature], data['Y'])
            correlations.append((feature, corr))
        
        corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Format the dataframes as markdown tables
        importance_table = importance_df.to_markdown(index=False)
        corr_table = corr_df.to_markdown(index=False)
        shap_table = shap_summary_df.to_markdown(index=False)
        
        # Prepare the inputs description
        input_parameters = []
        dimension = problem.getDimension()
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = marginal.getDescription()[0]
            if name == "":
                name = f"X{i+1}"
            input_parameters.append(f"- **{name}**: {marginal.getName()}")
            
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

        R² Score: {r2:.4f}
        Mean Squared Error: {mse:.4f}

        The feature importances are provided in the table below:

        {importance_table}

        The correlations between the input variables and the output are provided in the table below:

        {corr_table}

        The mean absolute SHAP values for each feature are provided in the table below:

        {shap_table}

        Please:
        - Explain the purpose of SHAP analysis and its significance in interpreting machine learning models.
        - Discuss the performance of the Random Forest model based on the R² score and mean squared error.
        - Interpret the SHAP summary table, highlighting which input variables are most influential in predicting the output.
        - Provide insights into how the top features contribute to the model's predictions, including whether the relationship is positive or negative.
        - Reference the performance metrics and SHAP values in your discussion.  
        """

        # Call the API
        response_markdown = call_groq_api(prompt, model_name=language_model)
        st.session_state['ml_interpretation'] = response_markdown

    st.markdown(st.session_state['ml_interpretation'])

def plot_shap_summary(shap_values, X_test, feature_names):
    """Create SHAP summary plot showing feature importance."""
    summary_fig_key = 'ml_shap_summary_fig'
    
    if summary_fig_key not in st.session_state:
        # Create a new figure using the newer API
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.title("SHAP Feature Importance", fontsize=14)
        plt.tight_layout()
        
        # Save the current figure
        fig = plt.gcf()
        st.session_state[summary_fig_key] = fig
        
        # Close the figure to avoid displaying it twice
        plt.close()
    
    # Display the saved figure
    st.pyplot(st.session_state[summary_fig_key])

def plot_dependence_plots(shap_values, shap_summary_df, X_test, feature_names):
    """Create SHAP dependence plots for top features."""
    # Ensure X_test is a DataFrame with feature names
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Get top 3 features based on mean absolute SHAP values
    top_features = shap_summary_df['Feature'].head(3).tolist()
    
    # Create dependence plots for top features
    st.subheader("SHAP Dependence Plots for Top Features")
    st.write("These plots show how the SHAP value (impact on model output) changes with the feature value.")
    
    for feature in top_features:
        dependence_fig_key = f'ml_shap_dependence_{feature}'
        
        if dependence_fig_key not in st.session_state:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Use the newer shap.plots API for dependence plots
            shap.plots.scatter(shap_values[:, feature], show=False)
            plt.title(f"SHAP Dependence Plot for {feature}", fontsize=14)
            plt.tight_layout()
            
            # Save the current figure
            fig = plt.gcf()
            st.session_state[dependence_fig_key] = fig
            
            # Close the figure to avoid displaying it twice
            plt.close()
        
        # Display the saved figure
        st.pyplot(st.session_state[dependence_fig_key])
        
        # Add interpretation
        st.write(f"""
        **Interpretation for {feature}:**
        - The x-axis shows the value of {feature}.
        - The y-axis shows the SHAP value (impact on model prediction).
        - Each point represents a sample from the test set.
        - The color represents the value of the most interactive feature.
        - A positive SHAP value means the feature increases the prediction, while a negative value decreases it.
        """)