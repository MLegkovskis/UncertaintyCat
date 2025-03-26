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
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model, sample_inputs
from modules.monte_carlo import monte_carlo_simulation, create_monte_carlo_dataframe

def ml_analysis(model, problem, model_code_str=None, language_model='groq'):
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
    model_code_str : str, optional
        String representation of the model code for documentation, by default None
    language_model : str, optional
        Language model to use for analysis, by default "groq"
        
    Returns
    -------
    dict
        Dictionary containing ML analysis results
    """
    try:
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
            st.info("Generating data for ML analysis using Monte Carlo simulation (1000 samples)...")
            # Generate Monte Carlo samples
            results = monte_carlo_simulation(model, problem, N=1000, seed=42)
            # Convert to DataFrame
            data = create_monte_carlo_dataframe(results)
        else:
            # Assume model is already a DataFrame
            data = model
            
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame but got {type(data)}. The model must be either a callable function or a pandas DataFrame.")
        
        # Results section
        with st.expander("Results", expanded=True):
            st.subheader("SHAP Analysis Results")
            st.markdown("""
            ### SHAP Analysis Results
            
            This analysis uses machine learning techniques to understand the relationship between input variables and model outputs.
            By training a surrogate model (Random Forest) and applying SHAP (SHapley Additive exPlanations), we can identify
            which variables have the most influence on the model output and how they affect predictions.
            
            **Key Concepts:**
            - **SHAP (SHapley Additive exPlanations)**: A unified approach to explain the output of any machine learning model
            - **Feature Importance**: Ranking of variables by their overall impact on predictions
            - **Dependence Plots**: Show how the effect of a variable changes across its range
            """)
            
            # Train the surrogate model
            with st.spinner("Training surrogate model..."):
                rf_model, X_test, y_test, scaler, performance_metrics = train_surrogate_model(data, input_names)
            
            # Calculate SHAP values
            with st.spinner("Calculating SHAP values..."):
                shap_results = calculate_shap_values(rf_model, X_test, input_names)
            
            # Display key metrics
            top_feature = shap_results["shap_summary_df"]['Feature'].iloc[0]
            significant_features_count = sum(shap_results["importance_df"]['Importance'] > 0.05)
            
            # Create metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Influential Variable", top_feature)
            with col2:
                st.metric("Significant Variables", f"{significant_features_count}")
            with col3:
                st.metric("Model Accuracy", f"{performance_metrics['r2']:.4f} R²")
            
            # Display model performance
            st.subheader("Surrogate Model Performance")
            
            # Create metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{performance_metrics['r2']:.4f}")
            with col2:
                st.metric("RMSE", f"{performance_metrics['rmse']:.4f}")
            with col3:
                st.metric("Cross-Val R²", f"{performance_metrics['cv_r2_mean']:.4f} ± {performance_metrics['cv_r2_std']:.4f}")
            
            # Add validation plot
            create_validation_plot(performance_metrics['y_test'], performance_metrics['y_pred'])
            
            # Create interactive visualizations
            with st.spinner("Creating visualizations..."):
                create_interactive_visualizations(shap_results, X_test, input_names, performance_metrics)
            
            # Add model performance metrics explanation as a regular section instead of an expander
            st.subheader("Understanding Model Performance Metrics")
            st.markdown("""
            - **R² Score**: Coefficient of determination (0-1)
              - Measures the proportion of variance in the output that is predictable from the inputs
              - Higher values indicate better fit (1.0 is perfect prediction)
              - Values below 0.5 suggest poor model performance
            
            - **RMSE (Root Mean Square Error)**:
              - Measures the average magnitude of prediction errors
              - Lower values indicate better fit
              - In the same units as the output variable
            
            - **Cross-Validation R²**:
              - Average R² score across 5 different data splits
              - Measures how well the model generalizes to unseen data
              - The ± value shows the standard deviation across folds
            
            - **Validation Plot**:
              - Shows actual vs. predicted values
              - Points should fall close to the diagonal line (perfect prediction)
              - Scatter indicates prediction uncertainty
              - Systematic deviations suggest model bias
            """)
        
        # AI Insights section (only if language_model is provided)
        if language_model:
            with st.expander("AI Insights", expanded=True):
                st.markdown("### AI-Generated Expert Analysis")
                generate_expert_analysis(data, problem, rf_model, input_names, model_code_str, 
                                        shap_results, performance_metrics, language_model)
        
        return {
            "surrogate_model": rf_model,
            "shap_values": shap_results["shap_values"],
            "feature_importance": shap_results["importance_df"],
            "performance_metrics": performance_metrics
        }
    
    except Exception as e:
        st.error(f"Error in ML analysis: {str(e)}")
        st.code(traceback.format_exc(), language="python")
        return None

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
    # Check if the model is already in session_state
    if ('rf_model' in st.session_state and 'X_test' in st.session_state and
        'y_test' in st.session_state and 'scaler' in st.session_state and
        'rf_performance' in st.session_state):
        return (
            st.session_state['rf_model'],
            st.session_state['X_test'],
            st.session_state['y_test'],
            st.session_state['scaler'],
            st.session_state['rf_performance']
        )
    else:
        X = data[input_names]
        y = data['Y']

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Fit Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Calculate performance metrics
        y_pred = rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2')
        
        performance_metrics = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores),
            'y_test': y_test,
            'y_pred': y_pred
        }

        # Store in session_state
        st.session_state['rf_model'] = rf_model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler
        st.session_state['rf_performance'] = performance_metrics
    
        return rf_model, X_test, y_test, scaler, performance_metrics

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
    if ('shap_values' in st.session_state and 'shap_summary_df' in st.session_state and
        'importance_df' in st.session_state):
        shap_values = st.session_state['shap_values']
        shap_summary_df = st.session_state['shap_summary_df']
        importance_df = st.session_state['importance_df']
    else:
        # Ensure X_test is a DataFrame with feature names
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        
        # Create the explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(X_test_df)
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
        
        # Create a DataFrame with feature names and mean absolute SHAP values
        shap_summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean |SHAP|': mean_abs_shap_values
        })
        
        # Sort by mean absolute SHAP values in descending order
        shap_summary_df = shap_summary_df.sort_values('Mean |SHAP|', ascending=False)
        
        # Calculate feature importances from the model
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Store in session_state
        st.session_state['shap_values'] = shap_values
        st.session_state['shap_summary_df'] = shap_summary_df
        st.session_state['importance_df'] = importance_df
    
    return {
        "shap_values": shap_values,
        "shap_summary_df": shap_summary_df,
        "importance_df": importance_df
    }

def create_interactive_visualizations(shap_results, X_test, feature_names, performance_metrics):
    """
    Create interactive Plotly visualizations for SHAP analysis results.
    
    Parameters
    ----------
    shap_results : dict
        Dictionary containing SHAP analysis results
    X_test : numpy.ndarray
        Test data features
    feature_names : list
        List of feature names
    performance_metrics : dict
        Dictionary containing model performance metrics
    """
    shap_values = shap_results["shap_values"]
    shap_summary_df = shap_results["shap_summary_df"]
    importance_df = shap_results["importance_df"]
    
    # Ensure X_test is a DataFrame with feature names
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Create a 2x2 subplot figure
    st.subheader("SHAP Analysis Visualizations")
    
    # Create a figure with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "SHAP Feature Importance", 
            "Random Forest Feature Importance",
            "SHAP Values Distribution", 
            "Feature Correlation with SHAP Values"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "box"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.3,  # Increased spacing to prevent overlap
        horizontal_spacing=0.1
    )
    
    # 1. SHAP Feature Importance (Bar Chart) - Top left
    fig.add_trace(
        go.Bar(
            y=shap_summary_df['Feature'],
            x=shap_summary_df['Mean |SHAP|'],
            orientation='h',
            marker_color='#1f77b4',
            name='Mean |SHAP|'
        ),
        row=1, col=1
    )
    
    # 2. Random Forest Feature Importance (Bar Chart) - Top right
    fig.add_trace(
        go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Importance'],
            orientation='h',
            marker_color='#ff7f0e',
            name='RF Importance'
        ),
        row=1, col=2
    )
    
    # 3. SHAP Values Distribution (Box Plot) - Bottom left
    # Prepare data for box plot
    box_data = []
    for feature in shap_summary_df['Feature'].tolist():
        box_data.append(
            go.Box(
                y=shap_values[:, feature].values,
                name=feature,
                boxmean=True,
                jitter=0.3,
                pointpos=-1.8,
                boxpoints='outliers'
            )
        )
    
    # Add all box plots to the figure
    for box in box_data[:5]:  # Limit to top 5 features for clarity
        fig.add_trace(box, row=2, col=1)
    
    # 4. Feature Correlation with SHAP Values (Scatter Plot) - Bottom right
    # Get top feature for scatter plot
    top_feature = shap_summary_df['Feature'].iloc[0]
    
    # Create scatter plot for top feature
    fig.add_trace(
        go.Scatter(
            x=X_test_df[top_feature],
            y=shap_values[:, top_feature].values,
            mode='markers',
            marker=dict(
                size=8,
                color=X_test_df[top_feature],
                colorscale='Viridis',
                colorbar=dict(
                    title=top_feature,
                    thickness=15,
                    len=0.5,
                    y=0.25
                )
            ),
            name=f'SHAP for {top_feature}'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="SHAP Analysis Dashboard",
        showlegend=False,
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Mean |SHAP| Value", row=1, col=1)
    fig.update_xaxes(title_text="Random Forest Importance", row=1, col=2)
    fig.update_xaxes(title_text=f"{top_feature} Value", row=2, col=2)
    
    fig.update_yaxes(title_text="Feature", row=1, col=1)
    fig.update_yaxes(title_text="Feature", row=1, col=2)
    fig.update_yaxes(title_text="SHAP Value", row=2, col=1)
    fig.update_yaxes(title_text=f"SHAP Value for {top_feature}", row=2, col=2)
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Create dependence plots for top features
    st.subheader("SHAP Dependence Plots for Top Features")
    st.markdown("""
    These plots show how the SHAP value (impact on model output) changes with the feature value.
    A positive SHAP value means the feature increases the prediction, while a negative value decreases it.
    """)
    
    # Get top 3 features
    top_features = shap_summary_df['Feature'].head(3).tolist()
    
    # Create a row of 3 columns for the top features
    cols = st.columns(len(top_features))
    
    # Create a dependence plot for each top feature
    for i, feature in enumerate(top_features):
        # Find most correlated feature for coloring
        corr_matrix = X_test_df.corr().abs()
        corr_feature = corr_matrix[feature].drop(feature).idxmax()
        
        # Create scatter plot
        scatter_fig = px.scatter(
            x=X_test_df[feature],
            y=shap_values[:, feature].values,
            color=X_test_df[corr_feature],
            labels={
                'x': feature,
                'y': f'SHAP value for {feature}',
                'color': corr_feature
            },
            title=f"SHAP Dependence Plot: {feature}",
            color_continuous_scale='Viridis',
            height=400
        )
        
        # Add trend line
        scatter_fig.add_trace(
            go.Scatter(
                x=X_test_df[feature].sort_values(),
                y=np.poly1d(np.polyfit(X_test_df[feature], shap_values[:, feature].values, 1))(X_test_df[feature].sort_values()),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Update layout
        scatter_fig.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        
        # Display in the appropriate column
        with cols[i]:
            st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Add interpretation
            mean_shap = np.mean(shap_values[:, feature].values)
            direction = "positive" if mean_shap > 0 else "negative"
            strength = abs(mean_shap) / shap_summary_df['Mean |SHAP|'].max()
            strength_desc = "strong" if strength > 0.66 else "moderate" if strength > 0.33 else "weak"
            
            st.markdown(f"""
            **Key Insights for {feature}:**
            - Overall {direction} impact on predictions ({strength_desc} effect)
            - {'Higher values generally increase predictions' if mean_shap > 0 else 'Higher values generally decrease predictions'}
            - Interacts with {corr_feature} (color scale)
            """)

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
    """
    # Check if the interpretation is already in session_state
    if 'ml_interpretation' in st.session_state:
        st.markdown(st.session_state['ml_interpretation'])
    else:
        # Format the model code for inclusion in the prompt
        if model_code_str:
            model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])
        else:
            model_code_formatted = "No model code provided"
        
        # Calculate correlations
        correlations = []
        for feature in feature_names:
            corr, _ = pearsonr(data[feature], data['Y'])
            correlations.append((feature, corr))
        
        corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        # Format the dataframes as markdown tables
        importance_table = shap_results["importance_df"].to_markdown(index=False)
        corr_table = corr_df.to_markdown(index=False)
        shap_table = shap_results["shap_summary_df"].to_markdown(index=False)
        
        # Prepare the inputs description
        input_parameters = []
        dimension = problem.getDimension()
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = marginal.getDescription()[0]
            if name == "":
                name = f"X{i+1}"
            dist_type = marginal.__class__.__name__
            params = marginal.getParameter()
            input_parameters.append(f"- **{name}**: {dist_type} distribution with parameters {list(params)}")
        
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

        The performance metrics of the Random Forest surrogate model are as follows:

        - R² Score: {performance_metrics['r2']:.4f}
        - Mean Squared Error: {performance_metrics['mse']:.4f}
        - RMSE: {performance_metrics['rmse']:.4f}
        - Cross-Validation R² (mean ± std): {performance_metrics['cv_r2_mean']:.4f} ± {performance_metrics['cv_r2_std']:.4f}

        The Random Forest feature importances are provided in the table below:

        {importance_table}

        The correlations between the input variables and the output are provided in the table below:

        {corr_table}

        The mean absolute SHAP values for each feature are provided in the table below:

        {shap_table}

        Please provide a comprehensive analysis of the results:
        
        1. Explain what SHAP values represent and how they help interpret the model
        2. Analyze the surrogate model performance and what it tells us about the system
        3. Compare SHAP-based importance with traditional Random Forest importance
        4. Identify the most influential variables and explain their impact on the output
        5. Discuss any interesting patterns or relationships revealed by the SHAP analysis
        6. Provide actionable insights based on the SHAP results
        
        Your analysis should be detailed yet accessible, with clear explanations of technical concepts.
        """

        # Call the API
        with st.spinner("Generating expert analysis..."):
            response_markdown = call_groq_api(prompt, model_name=language_model)
            st.session_state['ml_interpretation'] = response_markdown

        st.markdown(st.session_state['ml_interpretation'])
    
    # Display a disclaimer about the prompt
    disclaimer_text = """
    **Note:** The AI assistant has been provided with the model code, input distributions, 
    and the SHAP analysis results above. You can ask questions to clarify any aspects of the analysis.
    """
    
    # Define context generator function
    def generate_context(prompt):
        # Format the model code for inclusion in the context
        if model_code_str:
            model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])
        else:
            model_code_formatted = "Model code not available"
        
        # Recreate the inputs description
        input_parameters = []
        dimension = problem.getDimension()
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = marginal.getDescription()[0]
            if name == "":
                name = f"X{i+1}"
            dist_type = marginal.__class__.__name__
            params = marginal.getParameter()
            input_parameters.append(f"- **{name}**: {dist_type} distribution with parameters {list(params)}")
        
        inputs_description = '\n'.join(input_parameters)
        
        # Get the SHAP summary table
        shap_table = shap_results["shap_summary_df"].to_markdown(index=False)
            
        return f"""
        You are an expert assistant helping users understand SHAP-based machine learning analysis results. 
        
        Here is the model code:
        ```python
        {model_code_formatted}
        ```
        
        Here is information about the input distributions:
        {inputs_description}
        
        The performance metrics of the Random Forest surrogate model are as follows:
        - R² Score: {performance_metrics['r2']:.4f}
        - Mean Squared Error: {performance_metrics['mse']:.4f}
        - RMSE: {performance_metrics['rmse']:.4f}
        - Cross-Validation R² (mean ± std): {performance_metrics['cv_r2_mean']:.4f} ± {performance_metrics['cv_r2_std']:.4f}
        
        Here is the SHAP analysis summary:
        {shap_table}
        
        Here is the explanation that was previously generated:
        {st.session_state['ml_interpretation']}
        
        Answer the user's question based on this information. Be concise but thorough.
        If you're not sure about something, acknowledge the limitations of your knowledge.
        Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
        Explain the mathematical basis of SHAP values and how they differ from other feature importance methods if asked.
        """
    
    # Create the chat interface
    create_chat_interface(
        session_key="ml_analysis",
        context_generator=generate_context,
        input_placeholder="Ask a question about the SHAP analysis...",
        disclaimer_text=disclaimer_text,
        language_model=language_model
    )

def create_validation_plot(y_true, y_pred):
    """
    Create an interactive validation plot showing actual vs. predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    """
    # Create a DataFrame for the validation plot
    validation_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': np.abs(y_true - y_pred)
    })
    
    # Calculate min and max for plot limits
    min_val = min(validation_df['Actual'].min(), validation_df['Predicted'].min())
    max_val = max(validation_df['Actual'].max(), validation_df['Predicted'].max())
    
    # Add some padding to the limits
    range_val = max_val - min_val
    min_val -= range_val * 0.05
    max_val += range_val * 0.05
    
    # Create the scatter plot
    fig = go.Figure()
    
    # Add the scatter plot of actual vs predicted
    fig.add_trace(
        go.Scatter(
            x=validation_df['Actual'],
            y=validation_df['Predicted'],
            mode='markers',
            marker=dict(
                size=8,
                color=validation_df['Error'],
                colorscale='Viridis',
                colorbar=dict(title='Absolute Error'),
                showscale=True
            ),
            name='Test Samples'
        )
    )
    
    # Add the perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Perfect Prediction'
        )
    )
    
    # Add confidence intervals (±RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    x_range = np.linspace(min_val, max_val, 100)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range + rmse,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dot'),
            name=f'+RMSE ({rmse:.4f})'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=x_range - rmse,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dot'),
            name=f'-RMSE ({rmse:.4f})'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Surrogate Model Validation: Actual vs. Predicted Values',
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    # Set equal axis ranges
    fig.update_xaxes(range=[min_val, max_val])
    fig.update_yaxes(range=[min_val, max_val])
    
    # Add a grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Add interpretation
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate percentage of points within RMSE
        within_rmse = np.sum(np.abs(y_true - y_pred) <= rmse) / len(y_true) * 100
        st.metric("Points within ±RMSE", f"{within_rmse:.1f}%")
        
    with col2:
        # Calculate max error
        max_error = np.max(np.abs(y_true - y_pred))
        st.metric("Maximum Error", f"{max_error:.4f}")
    
    # Add a brief explanation
    st.markdown("""
    **Validation Plot Interpretation:**
    - **Points on diagonal line**: Perfect predictions
    - **Color intensity**: Magnitude of prediction error
    - **Dotted lines**: ±RMSE confidence interval
    - **Clustering pattern**: Indicates model's prediction accuracy across the output range
    """)