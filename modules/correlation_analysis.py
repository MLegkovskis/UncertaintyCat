import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model

def compute_correlation_analysis(model, problem, model_code_str=None):
    """
    Perform comprehensive correlation analysis calculations without UI display.
    
    This function calculates various correlation coefficients between input variables
    and model outputs, including Pearson, Spearman, PCC, PRCC, SRC, and SRRC.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    model_code_str : str, optional
        String representation of the model code for documentation
        
    Returns
    -------
    dict
        Dictionary containing correlation analysis results
    """
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Get input names and dimension
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        name = marginal.getDescription()[0]
        input_names.append(name if name != "" else f"X{i+1}")
    
    # Generate samples
    n_samples = 1000  # Number of samples for correlation analysis
    # Generate samples
    sample_X = problem.getSample(n_samples)
    sample_Y = model(sample_X)
    
    # Check if output is multivariate
    if isinstance(sample_Y, ot.Sample) and sample_Y.getDimension() > 1:
        is_multivariate = True
        output_dimension = sample_Y.getDimension()
        output_names = sample_Y.getDescription()
        # If output names are empty, create default names
        if not output_names or all(name == "" for name in output_names):
            output_names = [f"Y{i+1}" for i in range(output_dimension)]
    else:
        is_multivariate = False
        output_dimension = 1
        output_names = ["Y"]
    
    # Process each output
    all_correlation_results = {}
    
    for output_idx in range(output_dimension):
        # Get the current output name
        output_name = output_names[output_idx]
        
        # Extract current output data
        if is_multivariate:
            output_data = np.array([sample_Y[i, output_idx] for i in range(n_samples)])
        else:
            output_data = np.array([float(sample_Y[i][0]) for i in range(n_samples)])
        
        # Convert OpenTURNS sample to numpy array for easier manipulation
        input_data = np.array([[sample_X[i, j] for j in range(dimension)] for i in range(n_samples)])
        
        # Create pandas DataFrames
        df_X = pd.DataFrame(input_data, columns=input_names)
        df_Y = pd.Series(output_data, name=output_name)
        
        # Calculate Pearson correlation coefficients
        pearson_corr = [np.corrcoef(df_X[var], df_Y)[0, 1] for var in input_names]
        
        # Calculate Spearman rank correlation coefficients
        spearman_corr = [df_X[var].corr(df_Y, method='spearman') for var in input_names]
        
        # Calculate Partial Correlation Coefficients (PCC)
        pcc_corr = []
        for var in input_names:
            # Linear regression for output vs all inputs
            y_model = np.linalg.lstsq(np.column_stack([np.ones(n_samples), df_X.drop(columns=[var])]), df_Y, rcond=None)[0]
            y_residuals = df_Y - (np.ones(n_samples) * y_model[0] + df_X.drop(columns=[var]).dot(y_model[1:]))
            
            # Linear regression for current input vs all other inputs
            x_model = np.linalg.lstsq(np.column_stack([np.ones(n_samples), df_X.drop(columns=[var])]), df_X[var], rcond=None)[0]
            x_residuals = df_X[var] - (np.ones(n_samples) * x_model[0] + df_X.drop(columns=[var]).dot(x_model[1:]))
            
            # Correlation between residuals
            pcc = np.corrcoef(x_residuals, y_residuals)[0, 1]
            pcc_corr.append(pcc)
        
        # Calculate Partial Rank Correlation Coefficients (PRCC)
        # Convert data to ranks first
        df_X_ranks = df_X.rank()
        df_Y_ranks = df_Y.rank()
        
        prcc_corr = []
        for var in input_names:
            # Linear regression for output ranks vs all input ranks
            y_model = np.linalg.lstsq(np.column_stack([np.ones(n_samples), df_X_ranks.drop(columns=[var])]), df_Y_ranks, rcond=None)[0]
            y_residuals = df_Y_ranks - (np.ones(n_samples) * y_model[0] + df_X_ranks.drop(columns=[var]).dot(y_model[1:]))
            
            # Linear regression for current input ranks vs all other input ranks
            x_model = np.linalg.lstsq(np.column_stack([np.ones(n_samples), df_X_ranks.drop(columns=[var])]), df_X_ranks[var], rcond=None)[0]
            x_residuals = df_X_ranks[var] - (np.ones(n_samples) * x_model[0] + df_X_ranks.drop(columns=[var]).dot(x_model[1:]))
            
            # Correlation between residuals
            prcc = np.corrcoef(x_residuals, y_residuals)[0, 1]
            prcc_corr.append(prcc)
        
        # Calculate Standardized Regression Coefficients (SRC)
        X_standardized = (df_X - df_X.mean()) / df_X.std()
        Y_standardized = (df_Y - df_Y.mean()) / df_Y.std()
        regression_model = np.linalg.lstsq(np.column_stack([np.ones(n_samples), X_standardized]), Y_standardized, rcond=None)[0]
        src_corr = regression_model[1:]  # Skip the intercept
        
        # Calculate Standardized Rank Regression Coefficients (SRRC)
        X_rank_standardized = (df_X_ranks - df_X_ranks.mean()) / df_X_ranks.std()
        Y_rank_standardized = (df_Y_ranks - df_Y_ranks.mean()) / df_Y_ranks.std()
        rank_regression_model = np.linalg.lstsq(np.column_stack([np.ones(n_samples), X_rank_standardized]), Y_rank_standardized, rcond=None)[0]
        srrc_corr = rank_regression_model[1:]  # Skip the intercept
        
        # Combine all correlation results into a DataFrame
        corr_data = {
            'Pearson': pearson_corr,
            'Spearman': spearman_corr,
            'PCC': pcc_corr,
            'PRCC': prcc_corr,
            'SRC': src_corr,
            'SRRC': srrc_corr
        }
        
        corr_df = pd.DataFrame(corr_data, index=input_names)
        
        # Store results for this output
        result_item = {
            'corr_df': corr_df,
            'input_names': input_names,
            'output_name': output_name,
            'sample_X': sample_X,
            'sample_Y': sample_Y,
            'output_idx': output_idx,
            'n_samples': n_samples,
            'is_multivariate': is_multivariate
        }
        
        all_correlation_results[output_name] = result_item
    
    # Return all results
    return {
        'all_correlation_results': all_correlation_results,
        'input_names': input_names,
        'output_names': output_names,
        'output_dimension': output_dimension,
        'is_multivariate': is_multivariate,
        'model_code_str': model_code_str,
        'sample_X': sample_X,
        'sample_Y': sample_Y,
        'n_samples': n_samples
    }

def correlation_analysis(model, problem, model_code_str=None, language_model="groq", display_results=True):
    """
    Perform comprehensive correlation analysis with enterprise-grade visualizations.
    
    This function calculates various correlation coefficients between input variables
    and model outputs, including Pearson, Spearman, PCC, PRCC, SRC, and SRRC.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis, by default "groq"
    display_results : bool, optional
        Whether to display results using Streamlit UI (default: True)
        Set to False when running in batch mode or "Run All Analyses"
        
    Returns
    -------
    dict
        Dictionary containing correlation analysis results
    """
    try:
        # Compute all correlation analysis results
        results = compute_correlation_analysis(model, problem, model_code_str)
        
        # Save results to session state for later access
        if 'correlation_results' not in st.session_state:
            st.session_state.correlation_results = results
        
        # If not displaying results, just return the computed data
        if not display_results:
            return results
            
        # Create a two-column layout for the main content and chat interface
        main_col, chat_col = st.columns([2, 1])
        
        with main_col:
            # Extract data from results
            all_correlation_results = results['all_correlation_results']
            input_names = results['input_names']
            output_names = results['output_names']
            output_dimension = results['output_dimension']
            is_multivariate = results['is_multivariate']
            sample_X = results['sample_X']
            sample_Y = results['sample_Y']
            n_samples = results['n_samples']
            
            # Results Section
            with st.expander("Results", expanded=True):
                st.subheader("Correlation Analysis")
                st.markdown("""
                Correlation analysis quantifies the statistical relationship between input variables and model outputs.
                This analysis helps identify which inputs have the strongest linear and monotonic relationships with the output.
                
                Multiple correlation metrics are calculated to provide a comprehensive understanding of variable relationships:
                
                - **Pearson Correlation**: Measures linear correlation between variables
                - **Spearman Correlation**: Measures monotonic relationships using ranks
                - **Partial Correlation (PCC)**: Measures linear correlation while controlling for other variables
                - **Partial Rank Correlation (PRCC)**: Rank-based version of PCC for nonlinear relationships
                - **Standardized Regression Coefficient (SRC)**: Standardized coefficients from linear regression
                - **Standardized Rank Regression Coefficient (SRRC)**: Rank-based version of SRC
                """)
                
                # Create tabs for multiple outputs if needed
                if is_multivariate and output_dimension > 1:
                    tabs = st.tabs(output_names)
                
                for output_idx in range(output_dimension):
                    # Get the current output name
                    output_name = output_names[output_idx]
                    
                    # Get results for this output
                    result_item = all_correlation_results[output_name]
                    corr_df = result_item['corr_df']
                    
                    # Display in the appropriate container
                    if is_multivariate and output_dimension > 1:
                        with tabs[output_idx]:
                            display_correlation_results(
                                {
                                    'corr_df': corr_df,
                                    'input_names': input_names,
                                    'output_names': output_names,
                                    'sample_X': sample_X,
                                    'sample_Y': sample_Y,
                                    'n_samples': n_samples,
                                    'is_multivariate': is_multivariate
                                },
                                language_model
                            )
                    else:
                        display_correlation_results(
                            {
                                'corr_df': corr_df,
                                'input_names': input_names,
                                'output_names': output_names,
                                'sample_X': sample_X,
                                'sample_Y': sample_Y,
                                'n_samples': n_samples,
                                'is_multivariate': is_multivariate
                            },
                            language_model
                        )
            
            # AI Insights Section
            if language_model:
                with st.expander("AI Insights", expanded=True):
                    # Generate prompt based on correlation results
                    main_output = output_names[0]  # Use first output for AI prompt in multivariate case
                    main_results = all_correlation_results[main_output]
                    main_corr_df = main_results['corr_df']
                    
                    # Create a string representation of the correlation table
                    corr_table_text = "Correlation Coefficients:\n"
                    for var in main_corr_df.index:
                        corr_table_text += f"- {var}:\n"
                        for method in main_corr_df.columns:
                            corr_table_text += f"  {method}: {main_corr_df.loc[var, method]:.4f}\n"
                    
                    # Find strongest correlations
                    strongest_pearson_var = main_corr_df['Pearson'].abs().idxmax()
                    strongest_pearson = main_corr_df.loc[strongest_pearson_var, 'Pearson']
                    strongest_spearman_var = main_corr_df['Spearman'].abs().idxmax()
                    strongest_spearman = main_corr_df.loc[strongest_spearman_var, 'Spearman']
                    
                    # Generate prompt
                    prompt = f"""
{RETURN_INSTRUCTION}

Analyze these correlation results for an enterprise-grade engineering model:

```python
{model_code_str if model_code_str else "Model code not provided"}
```

{corr_table_text}

Key observations:
- Strongest Pearson correlation: {strongest_pearson_var} ({strongest_pearson:.4f})
- Strongest Spearman correlation: {strongest_spearman_var} ({strongest_spearman:.4f})
- Number of samples: {n_samples}
- Input variables: {", ".join(input_names)}
- Output: {main_output}

Please provide a comprehensive enterprise-grade analysis of these correlation results. Your analysis should include:

1. Executive Summary
   - Key findings and their business/engineering implications
   - Most influential variables and their significance
   - Overall assessment of input-output relationships

2. Technical Analysis
   - Explanation of correlation patterns and their statistical significance
   - Analysis of linear vs. nonlinear relationships (comparing Pearson vs. Spearman)
   - Interpretation of partial correlations and their implications
   - Assessment of collinearity among inputs (if relevant)

3. Risk Assessment
   - Identification of critical variables for risk management
   - Potential impact of input uncertainties on output variability
   - Robustness of the identified relationships

4. Recommendations
   - Guidance for system optimization based on correlation patterns
   - Variables that should be closely monitored or controlled
   - Suggestions for further analysis or model refinement

Focus on actionable insights that would be valuable for executive decision-makers in an engineering context.
"""
                    
                    with st.spinner("Generating expert analysis..."):
                        response_key = f'correlation_analysis_response_{output_name}'
                        
                        if response_key not in st.session_state:
                            response = call_groq_api(prompt, model_name=language_model)
                            st.session_state[response_key] = response
                        else:
                            response = st.session_state[response_key]
                            
                        st.markdown(response)
        
        # CHAT INTERFACE in the right column
        with chat_col:
            st.markdown("### Ask Questions About This Analysis")
            
            # Display a disclaimer about the prompt
            disclaimer_text = """
            **Note:** The AI assistant has been provided with the model code and the 
            correlation analysis results. You can ask questions to clarify any aspects of the analysis.
            """
            st.info(disclaimer_text)
            
            # Initialize session state for chat messages if not already done
            if "correlation_analysis_chat_messages" not in st.session_state:
                st.session_state.correlation_analysis_chat_messages = []
            
            # Create chat interface
            create_chat_interface(
                "correlation_analysis_chat",
                lambda prompt: f"""
                You are an expert assistant helping users understand correlation analysis results. 
                
                Here is the model code:
                ```python
                {model_code_str if model_code_str else "Model code not provided"}
                ```
                
                Here is the correlation analysis summary:
                {corr_table_text}
                
                Here is the explanation that was previously generated:
                {st.session_state.get(f'correlation_analysis_response_{output_names[0]}', 'No analysis available yet.')}
                
                Answer the user's question: {prompt}
                
                Be concise but thorough. Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                """,
                input_placeholder="Ask a question about the correlation analysis...",
                disclaimer_text="Ask questions about the correlation analysis results.",
                language_model=language_model
            )
        
        return results
        
    except Exception as e:
        if display_results:
            st.error(f"Error in correlation analysis: {str(e)}")
        raise

def display_correlation_results(analysis_results, language_model='groq'):
    """
    Display correlation analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing correlation analysis results
    language_model : str, optional
        Language model to use for AI insights, by default 'groq'
    """
    # Extract results
    corr_df = analysis_results.get('corr_df')
    input_names = analysis_results.get('input_names')
    output_names = analysis_results.get('output_names')
    sample_X = analysis_results.get('sample_X')
    sample_Y = analysis_results.get('sample_Y')
    n_samples = analysis_results.get('n_samples')
    is_multivariate = analysis_results.get('is_multivariate')
    correlation_interpretation = analysis_results.get('correlation_interpretation')
    
    # If multivariate, we need to handle each output separately
    for output_idx, output_name in enumerate(output_names):
        # Display correlation results for this output
        st.subheader(f"Correlation Analysis for {output_name}")
        
        # Create a stacked bar chart for correlation coefficients
        corr_values = []
        for method in ['Pearson', 'Spearman', 'Kendall']:
            for input_name in input_names:
                corr_values.append({
                    'Input': input_name,
                    'Method': method,
                    'Correlation': corr_df.loc[(input_name, output_name), method]
                })
        
        corr_bar_df = pd.DataFrame(corr_values)
        
        # Create a bar chart
        fig_bar = px.bar(
            corr_bar_df,
            x='Input',
            y='Correlation',
            color='Method',
            barmode='group',
            title=f'Correlation Coefficients for {output_name}',
            labels={'Correlation': 'Correlation Coefficient', 'Input': 'Input Variable'},
            color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'],
            height=500
        )
        
        fig_bar.update_layout(
            xaxis_title='Input Variable',
            yaxis_title='Correlation Coefficient',
            legend_title='Method',
            yaxis=dict(range=[-1, 1]),
            hovermode='closest'
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Create a heatmap for correlation matrix
        corr_heatmap_data = []
        for input_name in input_names:
            corr_heatmap_data.append({
                'Input': input_name,
                'Pearson': corr_df.loc[(input_name, output_name), 'Pearson'],
                'Spearman': corr_df.loc[(input_name, output_name), 'Spearman'],
                'Kendall': corr_df.loc[(input_name, output_name), 'Kendall']
            })
        
        corr_heatmap_df = pd.DataFrame(corr_heatmap_data)
        corr_heatmap_df = corr_heatmap_df.set_index('Input')
        
        # Create a heatmap
        fig_heatmap = px.imshow(
            corr_heatmap_df,
            labels=dict(x="Method", y="Input Variable", color="Correlation"),
            x=['Pearson', 'Spearman', 'Kendall'],
            y=corr_heatmap_df.index,
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1],
            title=f'Correlation Heatmap for {output_name}',
            height=500
        )
        
        fig_heatmap.update_layout(
            xaxis_title='Method',
            yaxis_title='Input Variable',
            coloraxis_colorbar=dict(
                title='Correlation',
                thicknessmode="pixels", thickness=20,
                lenmode="pixels", len=300,
                yanchor="top", y=1,
                ticks="outside"
            )
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Display AI interpretation if available
    if correlation_interpretation:
        st.subheader("AI Insights")
        st.markdown(correlation_interpretation)
