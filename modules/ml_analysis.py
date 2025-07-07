# modules/ml_analysis.py

import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import matplotlib.pyplot as plt
import shap
import traceback
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import pearsonr
from utils.core_utils import call_groq_api
from utils.constants import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model, sample_inputs
from modules.monte_carlo import monte_carlo_simulation, create_monte_carlo_dataframe

def compute_ml_analysis(model, problem, size=1000, model_code_str=None, language_model='groq'):
    """
    Compute machine learning-based sensitivity analysis using SHAP values without UI components.
    
    This function builds a surrogate machine learning model (Random Forest) and uses SHAP (SHapley Additive exPlanations)
    to analyze the importance and impact of each input variable on the model output.
    
    Parameters
    ----------
    model : callable or pd.DataFrame
        Either the model function or DataFrame containing input samples and model outputs
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    size : int, optional
        Number of samples for Monte Carlo simulation, by default 1000
    model_code_str : str, optional
        String representation of the model code for documentation, by default None
    language_model : str, optional
        Language model to use for analysis, by default "groq"
        
    Returns
    -------
    dict
        Dictionary containing Shapley Analysis results
    """
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Get input names from distribution
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        input_names.append(marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}")
    
    # If model is a function, generate data by running Monte Carlo simulation
    if callable(model) and not isinstance(model, pd.DataFrame):
        # Generate Monte Carlo samples
        results = monte_carlo_simulation(model, problem, N=size, seed=42)
        # Convert to DataFrame
        data = create_monte_carlo_dataframe(results)
    else:
        # Assume model is already a DataFrame
        data = model
        
    # Check if data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame but got {type(data)}. The model must be either a callable function or a pandas DataFrame.")
    
    # Train the surrogate model
    rf_model, X_test, y_test, scaler, performance_metrics = train_surrogate_model(data, input_names)
    
    # Calculate SHAP values
    shap_results = calculate_shap_values(rf_model, X_test, input_names)
    
    # Create visualizations
    fig_importance = create_importance_plot(shap_results)
    fig_dependence = create_dependence_plots(shap_results, X_test, input_names)
    fig_validation = create_validation_plot(performance_metrics['y_test'], performance_metrics['y_pred'])
    
    # Find most influential variables
    top_feature = shap_results["shap_summary_df"]['Feature'].iloc[0]
    significant_features_count = sum(shap_results["importance_df"]['Importance'] > 0.05)
    
    # Create model metrics dataframe
    model_metrics_df = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE', 'Cross-Val R² (mean)', 'Cross-Val R² (std)'],
        'Value': [
            f"{performance_metrics['r2']:.4f}",
            f"{performance_metrics['rmse']:.4f}",
            f"{performance_metrics['cv_r2_mean']:.4f}",
            f"{performance_metrics['cv_r2_std']:.4f}"
        ]
    })
    
    # Generate expert analysis if a language model is provided
    ai_insights = None
    if language_model and model_code_str:
        ai_insights = generate_expert_analysis(
            data, problem, rf_model, input_names, model_code_str, 
            shap_results, performance_metrics, language_model
        )
    
    # Return all results in a dictionary with the correct keys expected by UncertaintyCat.py
    return {
        "input_names": input_names,
        "surrogate_model": rf_model,
        "shap_results": shap_results,
        "performance_metrics": performance_metrics,
        "top_feature": top_feature,
        "significant_features_count": significant_features_count,
        "fig_importance": fig_importance,
        "fig_dependence": fig_dependence,
        "fig_validation": fig_validation,
        "ai_insights": ai_insights,
        # Keys expected by UncertaintyCat.py
        "shap_summary_plot": fig_dependence,
        "shap_bar_plot": fig_importance,
        "validation_plot": fig_validation,
        "feature_importance": shap_results["importance_df"],
        "model_metrics": model_metrics_df
    }

def display_ml_results(analysis_results, model_code_str=None, language_model='groq'):
    """
    Display machine learning analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing Shapley Analysis results
    model_code_str : str, optional
        String representation of the model code for documentation, by default None
    language_model : str, optional
        Language model to use for analysis, by default "groq"
    """
    try:
        # Extract results from the analysis_results dictionary
        input_names = analysis_results["input_names"]
        rf_model = analysis_results["surrogate_model"]
        performance_metrics = analysis_results["performance_metrics"]
        top_feature = analysis_results["top_feature"]
        significant_features_count = analysis_results["significant_features_count"]
        
        # Get the figures with the correct names
        shap_bar_plot = analysis_results["shap_bar_plot"]
        shap_summary_plot = analysis_results["shap_summary_plot"]
        validation_plot = analysis_results["validation_plot"]
        
        # Display SHAP Analysis Results header
        st.subheader("SHAP Analysis Results")
        st.markdown("""
        This analysis uses machine learning techniques to understand the relationship between input variables and model outputs.
        By training a surrogate model (Random Forest) and applying SHAP (SHapley Additive exPlanations), we can identify
        which variables have the most influence on the model output and how they affect predictions.
        """)
        
        # Create metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Most Influential Variable", top_feature)
        with col2:
            st.metric("Significant Variables", f"{significant_features_count}")
        with col3:
            st.metric("Model Accuracy", f"{performance_metrics['r2']:.4f} R²")
        
        # Display model performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{performance_metrics['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{performance_metrics['rmse']:.4f}")
        with col3:
            st.metric("Cross-Val R²", f"{performance_metrics['cv_r2_mean']:.4f} ± {performance_metrics['cv_r2_std']:.4f}")
        
        # Display validation plot
        st.subheader("Validation Plot")
        st.plotly_chart(validation_plot, use_container_width=True)
        
        # Display importance plot
        st.subheader("Feature Importance")
        st.plotly_chart(shap_bar_plot, use_container_width=True)
        
        # Display dependence plots
        st.subheader("Feature Dependence")
        st.plotly_chart(shap_summary_plot, use_container_width=True)
        
        # AI Insights section
        if "ai_insights" in analysis_results and analysis_results["ai_insights"]:
            # Store the insights in session state for reuse in the global chat
            if 'ml_analysis_response_markdown' not in st.session_state:
                st.session_state['ml_analysis_response_markdown'] = analysis_results["ai_insights"]
            
            st.subheader("AI-Generated Expert Analysis")
            st.markdown(analysis_results["ai_insights"])
    
    except Exception as e:
        st.error(f"Error in Shapley Analysis display: {str(e)}")
        st.code(traceback.format_exc())

def ml_analysis(model, problem, size=1000, model_code_str=None, language_model='groq', display_results=True):
    """
    Perform machine learning-based sensitivity analysis using SHAP values with enterprise-grade visualizations.
    
    This function builds a surrogate machine learning model (Random Forest) and uses SHAP (SHapley Additive exPlanations)
    to analyze the importance and impact of each input variable on the model output.
    
    Parameters
    ----------
    model : callable or pd.DataFrame
        Either the model function or DataFrame containing input samples and model outputs
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    size : int, optional
        Number of samples for Monte Carlo simulation, by default 1000
    model_code_str : str, optional
        String representation of the model code for documentation, by default None
    language_model : str, optional
        Language model to use for analysis, by default "groq"
    display_results : bool, optional
        Whether to display results in the UI (default is True)
        
    Returns
    -------
    dict
        Dictionary containing Shapley Analysis results
    """
    try:
        # Use a context manager to conditionally display results
        display_context = st.container() if display_results else nullcontext()
        
        with display_context:
            # Compute the analysis
            analysis_results = compute_ml_analysis(model, problem, size, model_code_str, language_model)
            
            # Display the results if requested
            if display_results:
                display_ml_results(analysis_results, model_code_str, language_model)
        
        return analysis_results
    
    except Exception as e:
        if display_results:
            st.error(f"Error in Shapley Analysis: {str(e)}")
            st.code(traceback.format_exc(), language="python")
        return None

# Define a nullcontext class to use when display_results is False
def get_ml_context_for_chat(ml_results):
    """
    Generate a formatted string containing Shapley Analysis results for the global chat context.
    
    Parameters
    ----------
    ml_results : dict
        Dictionary containing the results of the Shapley Analysis
        
    Returns
    -------
    str
        Formatted string with Shapley Analysis results for chat context
    """
    context = ""
    
    # SHAP summary (feature importance)
    shap_summary_df = None
    if "shap_results" in ml_results and "shap_summary_df" in ml_results["shap_results"]:
        shap_summary_df = ml_results["shap_results"]["shap_summary_df"]
    elif "feature_importance" in ml_results:
        shap_summary_df = ml_results["feature_importance"]
    
    if shap_summary_df is not None:
        context += "\n\n### Shapley Analysis: SHAP Feature Importance\n"
        context += shap_summary_df.to_markdown(index=False)
    
    # Model metrics
    model_metrics_df = ml_results.get("model_metrics")
    if model_metrics_df is not None:
        context += "\n\n### ML Model Performance Metrics\n"
        context += model_metrics_df.to_markdown(index=False)
    
    return context

class nullcontext:
    def __init__(self, enter_result=None):
        self.enter_result = enter_result
    
    def __enter__(self):
        return self.enter_result
    
    def __exit__(self, *excinfo):
        pass

def train_surrogate_model(data, input_names):
    """
    Train a Random Forest surrogate model on the data.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing input samples and model outputs
    input_names : list
        List of input variable names
        
    Returns
    -------
    tuple
        (rf_model, X_test, y_test, scaler, performance_metrics)
    """
    # Separate inputs and outputs
    X = data[input_names].values
    y = data['Y'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    cv_r2_mean = np.mean(cv_scores)
    cv_r2_std = np.std(cv_scores)
    
    # Create a dictionary of performance metrics
    performance_metrics = {
        'r2': r2,
        'rmse': rmse,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return rf_model, X_test_scaled, y_test, scaler, performance_metrics

def calculate_shap_values(model, X_test, feature_names):
    """
    Calculate SHAP values for the trained model.
    
    Parameters
    ----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    X_test : numpy.ndarray
        Test data features
    feature_names : list
        List of feature names
        
    Returns
    -------
    dict
        Dictionary containing SHAP analysis results
    """
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X_test)
    
    # Calculate mean absolute SHAP values for feature importance
    mean_abs_shap = np.abs(shap_values.values).mean(0)
    
    # Create a DataFrame with feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap / np.sum(mean_abs_shap)  # Normalize to sum to 1
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # Create a summary DataFrame for the SHAP summary plot
    shap_summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    })
    
    # Sort by mean absolute SHAP value
    shap_summary_df = shap_summary_df.sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
    
    # Return all SHAP-related results
    return {
        'shap_values': shap_values,
        'importance_df': importance_df,
        'shap_summary_df': shap_summary_df,
        'feature_names': feature_names
    }

def create_importance_plot(shap_results):
    """
    Create an interactive feature importance plot based on SHAP values.
    
    Parameters
    ----------
    shap_results : dict
        Dictionary containing SHAP analysis results
        
    Returns
    -------
    go.Figure
        Plotly figure with feature importance plot
    """
    # Extract data
    shap_summary_df = shap_results['shap_summary_df']
    
    # Create a horizontal bar chart
    fig = px.bar(
        shap_summary_df,
        y='Feature',
        x='Mean |SHAP|',
        orientation='h',
        title='Feature Importance (SHAP Values)',
        color='Mean |SHAP|',
        color_continuous_scale='viridis'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Mean |SHAP| Value',
        yaxis_title='Feature',
        yaxis=dict(autorange="reversed"),  # Put highest importance at the top
        coloraxis_showscale=False,
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_dependence_plots(shap_results, X_test, feature_names, max_features=3):
    """
    Create interactive dependence plots for the top features.
    
    Parameters
    ----------
    shap_results : dict
        Dictionary containing SHAP analysis results
    X_test : numpy.ndarray
        Test data features
    feature_names : list
        List of feature names
    max_features : int, optional
        Maximum number of features to show, by default 3
        
    Returns
    -------
    go.Figure
        Plotly figure with dependence plots
    """
    # Extract data
    shap_values = shap_results['shap_values']
    shap_summary_df = shap_results['shap_summary_df']
    
    # Get the top features
    top_features = shap_summary_df['Feature'].head(max_features).tolist()
    
    # Create a subplot grid
    rows = (len(top_features) + 1) // 2  # Ceiling division
    cols = min(2, len(top_features))
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{feature} Impact" for feature in top_features],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # Create a DataFrame from X_test for easier indexing
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Add dependence plots for each top feature
    for i, feature in enumerate(top_features):
        # Calculate row and column for this subplot
        row = i // cols + 1
        col = i % cols + 1
        
        # Get feature values and SHAP values
        feature_values = X_test_df[feature].values
        feature_shap_values = shap_values.values[:, feature_names.index(feature)]
        
        # Create scatter plot
        fig.add_trace(
            go.Scatter(
                x=feature_values,
                y=feature_shap_values,
                mode='markers',
                marker=dict(
                    color=feature_shap_values,
                    colorscale='RdBu_r',
                    size=8,
                    colorbar=dict(title="SHAP Value") if i == 0 else None,
                    showscale=i == 0
                ),
                name=feature,
                hovertemplate=f"{feature}: %{{x:.4f}}<br>SHAP value: %{{y:.4f}}<extra></extra>"
            ),
            row=row, col=col
        )
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=min(feature_values),
            y0=0,
            x1=max(feature_values),
            y1=0,
            line=dict(color="black", width=1, dash="dash"),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text="SHAP Value", row=row, col=col)
    
    # Update layout
    fig.update_layout(
        height=300 * rows,
        title_text="Feature Dependence Plots",
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_validation_plot(y_true, y_pred):
    """
    Create an interactive validation plot showing actual vs. predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns
    -------
    go.Figure
        Plotly figure with validation plot
    """
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate correlation
    correlation, _ = pearsonr(y_true, y_pred)
    
    # Create a DataFrame for the scatter plot
    validation_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Residual': y_true - y_pred
    })
    
    # Calculate min and max for the diagonal line
    min_val = min(validation_df['Actual'].min(), validation_df['Predicted'].min())
    max_val = max(validation_df['Actual'].max(), validation_df['Predicted'].max())
    
    # Add some padding to the limits
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    
    # Create a scatter plot
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=validation_df['Actual'],
            y=validation_df['Predicted'],
            mode='markers',
            marker=dict(
                color=validation_df['Residual'],
                colorscale='RdBu_r',
                size=8,
                colorbar=dict(title="Residual")
            ),
            name='Predictions',
            hovertemplate="Actual: %{x:.4f}<br>Predicted: %{y:.4f}<br>Residual: %{marker.color:.4f}<extra></extra>"
        )
    )
    
    # Add diagonal line (perfect prediction)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color="black", width=2, dash="dash"),
            name='Perfect Prediction'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"Model Validation (R² = {r2:.4f}, RMSE = {rmse:.4f})",
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        height=400,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set equal axis ranges
    fig.update_xaxes(range=[min_val, max_val])
    fig.update_yaxes(range=[min_val, max_val])
    
    return fig

def generate_expert_analysis(data, problem, rf_model, feature_names, model_code_str, 
                           shap_results, performance_metrics, language_model='groq'):
    """
    Generate expert analysis of SHAP results using AI.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing input samples and model outputs
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    rf_model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    feature_names : list
        List of feature names
    model_code_str : str
        String representation of the model code
    shap_results : dict
        Dictionary containing SHAP analysis results
    performance_metrics : dict
        Dictionary containing model performance metrics
    language_model : str, optional
        Language model to use for analysis, by default "groq"
        
    Returns
    -------
    str
        The generated AI analysis text
    """
    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')]) if model_code_str else ""
    
    # Prepare the inputs description
    input_parameters = []
    dimension = problem.getDimension()
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        name = feature_names[i]
        dist_type = marginal.__class__.__name__
        params = marginal.getParameter()
        input_parameters.append(f"- **{name}**: {dist_type} distribution with parameters {list(params)}")
    
    inputs_description = '\n'.join(input_parameters)
    
    # Format the feature importance table
    importance_table = shap_results['importance_df'].to_markdown(index=False, floatfmt=".4f")
    
    # Format the performance metrics
    performance_text = f"""
    - R² Score: {performance_metrics['r2']:.4f}
    - RMSE: {performance_metrics['rmse']:.4f}
    - Cross-Validation R² (mean ± std): {performance_metrics['cv_r2_mean']:.4f} ± {performance_metrics['cv_r2_std']:.4f}
    """
    
    # Prepare the prompt
    prompt = f"""
    {RETURN_INSTRUCTION}
    
    Given the following user-defined model defined in Python code:
    
    ```python
    {model_code_formatted}
    ```
    
    and the following uncertain input distributions:
    
    {inputs_description}
    
    A machine learning sensitivity analysis has been performed using a Random Forest surrogate model and SHAP values.
    
    The feature importance results are:
    
    {importance_table}
    
    The surrogate model performance metrics are:
    {performance_text}
    
    Please provide an expert analysis of these results:
    
    1. **Surrogate Model Performance**
       - Evaluate the quality of the Random Forest surrogate model
       - Discuss what the R² score, RMSE, and cross-validation results tell us about the model's predictive power
       - Explain any limitations or caveats in interpreting these results
    
    2. **Feature Importance Analysis**
       - Identify which variables have the strongest influence on the model output based on SHAP values
       - Explain what these importance values mean in the context of sensitivity analysis
       - Discuss how SHAP values provide insights beyond traditional sensitivity measures
    
    3. **Physical Interpretation**
       - Relate the identified important variables to the physical behavior of the model
       - Explain why certain variables might be more influential than others
       - Discuss any surprising or counter-intuitive findings
    
    4. **Recommendations**
       - Suggest which variables should be prioritized for uncertainty reduction
       - Recommend additional analyses that might be valuable given these ML patterns
       - Provide guidance on how these results can inform decision-making or model refinement
    
    Format your response with clear section headings and bullet points. Focus on actionable insights and quantitative recommendations.
    """
    
    # Call the AI API
    try:
        response_markdown = call_groq_api(prompt, model_name=language_model)
        return response_markdown
    except Exception as e:
        print(f"Error generating expert analysis: {str(e)}")
        return f"Error generating expert analysis: {str(e)}"