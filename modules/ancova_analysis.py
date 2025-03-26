import numpy as np
import pandas as pd
import openturns as ot
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def ancova_sensitivity_analysis(model, problem, size=2000, model_code_str=None, language_model=None):
    """Perform enterprise-grade ANCOVA sensitivity analysis.
    
    This module provides comprehensive global sensitivity analysis using the ANCOVA method,
    which is particularly useful for models with correlated inputs. ANCOVA separates the 
    variance explained by individual variables from that explained by correlations with 
    other inputs.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    size : int, optional
        Number of samples for ANCOVA analysis (default is 2000)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
        
    Returns
    -------
    dict
        Dictionary containing the results of the ANCOVA analysis
    """
    try:
        # Verify input types
        if not isinstance(model, ot.Function):
            raise TypeError("Model must be an OpenTURNS Function")
        if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
            raise TypeError("Problem must be an OpenTURNS Distribution")
            
        # Get dimension from the model's input dimension
        dimension = model.getInputDimension()
        
        # Get variable names
        variable_names = []
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            name = marginal.getDescription()[0]
            variable_names.append(name if name != "" else f"X{i+1}")
        
        # Create independent distribution for functional chaos
        marginals = [problem.getMarginal(i) for i in range(dimension)]
        independent_dist = ot.JointDistribution(marginals)
        
        # Create the orthogonal basis
        enumerateFunction = ot.LinearEnumerateFunction(dimension)
        
        # Create appropriate polynomial basis based on marginals
        polynomialCollection = []
        for i in range(dimension):
            marginal = independent_dist.getMarginal(i)
            
            # Select appropriate polynomial family based on distribution type
            if isinstance(marginal, ot.Normal):
                polynomialCollection.append(ot.HermiteFactory())
            elif isinstance(marginal, ot.Uniform):
                polynomialCollection.append(ot.LegendreFactory())
            elif isinstance(marginal, ot.Beta):
                polynomialCollection.append(ot.JacobiFactory())
            elif isinstance(marginal, ot.Gamma):
                polynomialCollection.append(ot.LaguerreFactory())
            else:
                # Default to Legendre for other distributions
                polynomialCollection.append(ot.LegendreFactory())
        
        productBasis = ot.OrthogonalProductPolynomialFactory(
            polynomialCollection, enumerateFunction
        )
        
        # Create adaptive strategy
        chaos_degree = 4  # Default degree
        adaptiveStrategy = ot.FixedStrategy(
            productBasis, enumerateFunction.getStrataCumulatedCardinal(chaos_degree)
        )
        
        # Generate sample for functional chaos with independent distribution
        experiment = ot.MonteCarloExperiment(independent_dist, size)
        X_indep = experiment.generate()
        Y_indep = model(X_indep)
        
        # Run functional chaos algorithm
        algo = ot.FunctionalChaosAlgorithm(X_indep, Y_indep, independent_dist, adaptiveStrategy)
        algo.run()
        result = algo.getResult()
        
        # Generate correlated sample for ANCOVA
        X_corr = problem.getSample(size)
        
        # Check if inputs are correlated
        has_copula = False
        try:
            copula = problem.getCopula()
            has_copula = not isinstance(copula, ot.IndependentCopula)
        except:
            has_copula = False
        
        indices_data = []
        
        if has_copula:
            # For correlated inputs, use the OpenTURNS ANCOVA class
            try:
                # Perform ANCOVA decomposition
                ancova = ot.ANCOVA(result, X_corr)
                
                # Get ANCOVA indices
                indices = ancova.getIndices()
                uncorrelatedIndices = ancova.getUncorrelatedIndices()
                correlatedIndices = indices - uncorrelatedIndices
                
                # Create data for display
                for i, name in enumerate(variable_names):
                    total_index = float(indices[i])
                    uncorrelated_index = float(uncorrelatedIndices[i])
                    correlated_index = float(correlatedIndices[i])
                    
                    # Calculate correlation percentage
                    corr_percent = (correlated_index / total_index) * 100 if total_index > 0 else 0.0
                    
                    indices_data.append({
                        'Variable': name,
                        'ANCOVA Index': total_index,
                        'Uncorrelated Index': uncorrelated_index,
                        'Correlated Index': correlated_index,
                        'Correlation %': corr_percent
                    })
            except Exception as e:
                # If ANCOVA fails, use Sobol indices from functional chaos
                sobol_indices = result.getSobolIndices()
                
                for i, name in enumerate(variable_names):
                    # Get first order Sobol index
                    first_order = sobol_indices.getSobolIndex(i)
                    
                    indices_data.append({
                        'Variable': name,
                        'ANCOVA Index': float(first_order),
                        'Uncorrelated Index': float(first_order),
                        'Correlated Index': 0.0,
                        'Correlation %': 0.0
                    })
        else:
            # For independent inputs, ANCOVA indices are equivalent to Sobol indices
            sobol_indices = result.getSobolIndices()
            
            for i, name in enumerate(variable_names):
                # Get first order Sobol index
                first_order = sobol_indices.getSobolIndex(i)
                
                indices_data.append({
                    'Variable': name,
                    'ANCOVA Index': float(first_order),
                    'Uncorrelated Index': float(first_order),
                    'Correlated Index': 0.0,
                    'Correlation %': 0.0
                })
        
        # Create DataFrame for display
        indices_df = pd.DataFrame(indices_data)
        
        # Sort by ANCOVA index for better visualization
        indices_df = indices_df.sort_values('ANCOVA Index', ascending=False)
        
        # Create Plotly bar chart for ANCOVA indices
        fig_bar = go.Figure()
        
        # Add ANCOVA indices
        fig_bar.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['ANCOVA Index'],
            name='ANCOVA Indices',
            marker_color='rgba(31, 119, 180, 0.8)'
        ))
        
        # Update layout
        fig_bar.update_layout(
            title='ANCOVA Sensitivity Indices',
            xaxis_title='Input Variables',
            yaxis_title='Sensitivity Index',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create stacked bar chart for uncorrelated and correlated contributions
        fig_stacked = go.Figure()
        
        # Add uncorrelated contribution
        fig_stacked.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['Uncorrelated Index'],
            name='Uncorrelated Effect',
            marker_color='rgba(31, 119, 180, 0.8)'
        ))
        
        # Add correlated contribution
        fig_stacked.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['Correlated Index'],
            name='Correlated Effect',
            marker_color='rgba(214, 39, 40, 0.8)'
        ))
        
        # Update layout
        fig_stacked.update_layout(
            title='Decomposition of ANCOVA Indices',
            xaxis_title='Input Variables',
            yaxis_title='Sensitivity Index',
            barmode='stack',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create a pie chart for ANCOVA indices
        fig_pie = px.pie(
            indices_df, 
            values='ANCOVA Index', 
            names='Variable',
            title='ANCOVA Indices Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # Create correlation heatmap
        correlation_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                input_i = X_corr.getMarginal(i).asPoint()
                input_j = X_corr.getMarginal(j).asPoint()
                correlation_matrix[i, j] = np.corrcoef(input_i, input_j)[0, 1]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=variable_names,
            y=variable_names,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='Correlation')
        ))
        fig_heatmap.update_layout(
            title='Input Correlation Structure',
            template='plotly_white',
            height=500
        )
        
        # Create explanatory text
        ancova_explanation = """
        ### ANCOVA Sensitivity Analysis
        
        The Analysis of Covariance (ANCOVA) method is a global sensitivity analysis approach that 
        specifically accounts for correlations between input variables. It decomposes the variance 
        of the model output into contributions from individual variables and their correlations.
        
        #### Interpreting the Results:
        
        - **ANCOVA Index**: Total sensitivity of the output to each input variable
        - **Uncorrelated Index**: Portion of sensitivity due to the variable's independent effect
        - **Correlated Index**: Portion of sensitivity due to correlations with other variables
        - **Correlation %**: Percentage of the total sensitivity that comes from correlations
        
        Unlike other methods like FAST, ANCOVA specifically accounts for input correlations, making it 
        more appropriate for models with dependent inputs.
        """
        
        # Generate LLM insights if requested
        llm_insights = None
        if language_model and model_code_str:
            # Prepare the prompt
            prompt = f"""
            I've performed an ANCOVA sensitivity analysis on the following model:
            ```python
            {model_code_str}
            ```
            
            The results show these ANCOVA indices:
            {', '.join([f"{row['Variable']}: {row['ANCOVA Index']:.4f} (Uncorrelated: {row['Uncorrelated Index']:.4f}, Correlated: {row['Correlated Index']:.4f})" for _, row in indices_df.iterrows()])}
            
            The correlation matrix between inputs shows:
            {', '.join([f"{variable_names[i]}-{variable_names[j]}: {correlation_matrix[i, j]:.4f}" for i in range(dimension) for j in range(i+1, dimension) if abs(correlation_matrix[i, j]) > 0.1])}
            
            Please provide 2-3 paragraphs of insights about:
            1. Which variables have the most influence on the model output and why
            2. How much of the variance is explained by correlations between inputs
            3. How these results could inform model simplification or further analysis
            
            {RETURN_INSTRUCTION}
            """
            
            # Call the LLM
            llm_insights = call_groq_api(prompt, model_name=language_model)
        
        # Return all results
        return {
            'indices_df': indices_df,
            'fig_bar': fig_bar,
            'fig_stacked': fig_stacked,
            'fig_pie': fig_pie,
            'fig_heatmap': fig_heatmap,
            'explanation': ancova_explanation,
            'llm_insights': llm_insights
        }
    except Exception as e:
        st.error(f"Error in ANCOVA sensitivity analysis: {str(e)}")
        raise e

def display_ancova_results(ancova_results, language_model=None, model_code_str=None):
    """
    Display ANCOVA sensitivity analysis results in the Streamlit interface.
    
    Parameters
    ----------
    ancova_results : dict
        Dictionary containing the results of the ANCOVA analysis
    language_model : str, optional
        Language model used for analysis, by default None
    model_code_str : str, optional
        String representation of the model code, by default None
    """
    # Results Section
    with st.expander("Results", expanded=True):
        # Overview
        st.subheader("ANCOVA Sensitivity Analysis Overview")
        st.markdown("""
        ANCOVA (Analysis of Covariance) sensitivity analysis is particularly useful for models with correlated inputs. 
        It separates the variance explained by individual variables from that explained by correlations with other inputs.
        
        #### Interpreting the Results:
        
        - **ANCOVA Index**: Total sensitivity of the output to each input variable
        - **Uncorrelated Index**: Portion of sensitivity due to the variable's independent effect
        - **Correlated Index**: Portion of sensitivity due to correlations with other variables
        - **Correlation %**: Percentage of the total sensitivity that comes from correlations
        
        Unlike FAST analysis, ANCOVA specifically accounts for input correlations, making it more appropriate 
        for models with dependent inputs.
        """)
        
        # Display the indices table
        st.subheader("Sensitivity Indices")
        
        # Get most influential variable
        most_influential = ancova_results['indices_df'].iloc[0]['Variable']
        most_influential_index = ancova_results['indices_df'].iloc[0]['ANCOVA Index']
        
        # Calculate sums
        sum_ancova = ancova_results['indices_df']['ANCOVA Index'].sum()
        sum_uncorrelated = ancova_results['indices_df']['Uncorrelated Index'].sum()
        sum_correlated = ancova_results['indices_df']['Correlated Index'].sum()
        correlation_effect = sum_correlated / sum_ancova if sum_ancova > 0 else 0.0
        
        # Create summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Most Influential Variable", 
                most_influential,
                f"ANCOVA Index: {most_influential_index:.4f}"
            )
        with col2:
            st.metric("Sum of ANCOVA Indices", f"{sum_ancova:.4f}")
        with col3:
            st.metric(
                "Correlation Effect", 
                f"{correlation_effect:.2%}",
                f"Sum of Correlated: {sum_correlated:.4f}"
            )
        
        # Display the indices table
        st.subheader("Detailed Numerical Results")
        display_df = ancova_results['indices_df'][['Variable', 'ANCOVA Index', 'Uncorrelated Index', 'Correlated Index', 'Correlation %']]
        display_df['Correlation %'] = display_df['Correlation %'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(display_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Sensitivity Visualizations")
        
        # Display the bar chart
        st.markdown("#### ANCOVA Sensitivity Indices")
        st.markdown("""
        This bar chart shows the total ANCOVA sensitivity index for each variable, 
        indicating their overall importance to the model output.
        """)
        st.plotly_chart(ancova_results['fig_bar'], use_container_width=True)
        
        # Display stacked bar chart
        st.markdown("#### Correlation Decomposition")
        st.markdown("""
        This stacked bar chart breaks down each variable's sensitivity into:
        - **Uncorrelated Effect**: The variable's direct influence on the output (blue)
        - **Correlated Effect**: The influence due to correlations with other variables (red)
        
        Variables with large red portions are strongly affected by correlations in the model.
        """)
        st.plotly_chart(ancova_results['fig_stacked'], use_container_width=True)
        
        # Display pie chart and heatmap in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ANCOVA Indices Distribution")
            st.markdown("""
            This pie chart shows the relative contribution of each variable to the total output variance.
            Larger slices indicate variables with stronger influence on the model output.
            """)
            st.plotly_chart(ancova_results['fig_pie'], use_container_width=True)
        with col2:
            st.markdown("#### Input Correlation Structure")
            st.markdown("""
            This heatmap visualizes the correlation structure between input variables:
            - Red cells indicate positive correlations
            - Blue cells indicate negative correlations
            - Darker colors represent stronger correlations
            
            Strong correlations explain why some variables have high correlated effects.
            """)
            st.plotly_chart(ancova_results['fig_heatmap'], use_container_width=True)
        
        # Add interpretation based on results
        if correlation_effect > 0.3:
            st.info("""
            **Significant correlation effects detected.** A substantial portion of the output variance 
            is explained by correlations between input variables. This means the model behavior is strongly 
            influenced by the joint distribution of inputs, not just their individual distributions.
            """)
        elif correlation_effect < 0.1:
            st.success("""
            **Minimal correlation effects detected.** The input correlations have little impact on the output variance.
            The model behavior can be understood primarily by studying the effect of each variable separately.
            """)
    
    # AI Insights Section
    if ancova_results['llm_insights'] and language_model:
        with st.expander("AI Insights", expanded=True):
            # Store the insights in session state for reuse
            if 'ancova_analysis_response_markdown' not in st.session_state:
                st.session_state['ancova_analysis_response_markdown'] = ancova_results['llm_insights']
            
            st.markdown(ancova_results['llm_insights'])
            
            # Display a disclaimer about the prompt
            disclaimer_text = """
            **Note:** The AI assistant has been provided with the model code, sensitivity indices, 
            and the analysis results above. You can ask questions to clarify any aspects of the ANCOVA analysis.
            """
            
            # Define context generator function
            def generate_context(prompt):
                # Get variable names and indices from the results
                variable_names = ancova_results['indices_df']['Variable'].tolist()
                uncorrelated = ancova_results['indices_df']['Uncorrelated Index'].tolist()
                correlated = ancova_results['indices_df']['Correlated Index'].tolist()
                total = ancova_results['indices_df']['ANCOVA Index'].tolist()
                
                indices_summary = ', '.join([f"{name}: Uncorrelated={unc:.4f}, Correlated={cor:.4f}, Total={tot:.4f}" 
                                           for name, unc, cor, tot in zip(variable_names, uncorrelated, correlated, total)])
                
                return f"""
                You are an expert assistant helping users understand ANCOVA sensitivity analysis results. 
                
                Here is the model code:
                ```python
                {model_code_str if model_code_str else "Model code not available"}
                ```
                
                Here is the sensitivity analysis summary:
                {indices_summary}
                
                Here is the explanation that was previously generated:
                {ancova_results['llm_insights']}
                
                Answer the user's question based on this information. Be concise but thorough.
                If you're not sure about something, acknowledge the limitations of your knowledge.
                Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
                Explain the difference between uncorrelated and correlated contributions if asked.
                """
            
            # Create the chat interface
            create_chat_interface(
                session_key="ancova_analysis",
                context_generator=generate_context,
                input_placeholder="Ask a question about the ANCOVA sensitivity analysis...",
                disclaimer_text=disclaimer_text,
                language_model=language_model
            )

def ancova_analysis(model, problem, size=2000, model_code_str=None, language_model=None):
    """
    Perform and display ANCOVA sensitivity analysis.
    
    This function serves as the main entry point for ANCOVA analysis, handling both
    the calculation and visualization of results.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    size : int, optional
        Number of samples for ANCOVA analysis (default is 2000)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    """
    with st.spinner("Running ANCOVA Sensitivity Analysis..."):
        ancova_results = ancova_sensitivity_analysis(
            model, problem, size=size, model_code_str=model_code_str,
            language_model=language_model
        )
        
        display_ancova_results(ancova_results, language_model, model_code_str)
