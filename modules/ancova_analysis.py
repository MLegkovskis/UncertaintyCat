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
    
    ANCOVA (Analysis of Covariance) is a variance-based method that generalizes the ANOVA 
    (Analysis of Variance) decomposition for models with correlated input parameters. It 
    decomposes the total variance into uncorrelated (physical) effects and correlated effects.
    
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
        
        # Calculate correlation matrix
        correlation_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                input_i = X_corr.getMarginal(i).asPoint()
                input_j = X_corr.getMarginal(j).asPoint()
                correlation_matrix[i, j] = np.corrcoef(input_i, input_j)[0, 1]
        
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
        
        # Determine if there are correlation effects
        has_correlation_effects = any(indices_df['Correlated Index'].abs() > 0.001)
        
        # Create a combined bar chart showing both total ANCOVA indices and their decomposition
        fig_combined = go.Figure()
        
        # Add total ANCOVA indices (S_i)
        fig_combined.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['ANCOVA Index'],
            name='Total ANCOVA Index (S<sub>i</sub>)',
            marker_color='rgba(55, 83, 109, 0.8)',
            hovertemplate='%{x}: S<sub>i</sub> = %{y:.4f}<extra></extra>'
        ))
        
        # Add uncorrelated contribution (S_i^U)
        fig_combined.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['Uncorrelated Index'],
            name='Uncorrelated Effect (S<sub>i</sub><sup>U</sup>)',
            marker_color='rgba(31, 119, 180, 0.8)',
            hovertemplate='%{x}: S<sub>i</sub><sup>U</sup> = %{y:.4f}<extra></extra>'
        ))
        
        # Add correlated contribution (S_i^C)
        fig_combined.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['Correlated Index'],
            name='Correlated Effect (S<sub>i</sub><sup>C</sup>)',
            marker_color='rgba(214, 39, 40, 0.8)',
            hovertemplate='%{x}: S<sub>i</sub><sup>C</sup> = %{y:.4f}<extra></extra>'
        ))
        
        # Add annotations to explain the relationship
        fig_combined.add_annotation(
            x=0.02,
            y=1.12,
            xref="paper",
            yref="paper",
            text="ANCOVA Decomposition: S<sub>i</sub> = S<sub>i</sub><sup>U</sup> + S<sub>i</sub><sup>C</sup>",
            showarrow=False,
            font=dict(size=14),
            align="left"
        )
        
        # Update layout
        fig_combined.update_layout(
            title='ANCOVA Sensitivity Analysis',
            xaxis_title='Input Variables',
            yaxis_title='Sensitivity Index',
            template='plotly_white',
            height=600,
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=120)  # Add more top margin for the annotation
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
        
        # Create improved correlation heatmap
        # First, let's create a mask for the diagonal to exclude it
        mask = np.ones((dimension, dimension))
        np.fill_diagonal(mask, 0)
        
        # Apply colorscale only to non-diagonal elements
        colorscale_values = []
        for i in range(dimension):
            for j in range(dimension):
                if i != j:  # Skip diagonal
                    colorscale_values.append(correlation_matrix[i, j])
        
        # Determine colorscale range based on actual data
        if colorscale_values:
            abs_max = max(abs(min(colorscale_values)), abs(max(colorscale_values)))
            abs_max = max(abs_max, 0.1)  # Ensure we have at least some range
        else:
            abs_max = 1.0
            
        # Create heatmap with improved visualization
        fig_heatmap = go.Figure()
        
        # Add heatmap trace
        heatmap_trace = go.Heatmap(
            z=correlation_matrix * mask,  # Apply mask to hide diagonal
            x=variable_names,
            y=variable_names,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(
                title='Correlation'
            ),
            hovertemplate='%{x} - %{y}: %{z:.4f}<extra></extra>'
        )
        fig_heatmap.add_trace(heatmap_trace)
        
        # Add text annotations for correlation values
        for i in range(dimension):
            for j in range(dimension):
                if i != j:  # Skip diagonal
                    color = 'white' if abs(correlation_matrix[i, j]) > 0.4 else 'black'
                    fig_heatmap.add_annotation(
                        x=variable_names[j],
                        y=variable_names[i],
                        text=f"{correlation_matrix[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color=color, size=10)
                    )
        
        fig_heatmap.update_layout(
            title='Input Correlation Structure',
            template='plotly_white',
            height=500,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        
        # Create explanatory text
        ancova_explanation = """
        ### ANCOVA Sensitivity Analysis
        
        The Analysis of Covariance (ANCOVA) method is a variance-based sensitivity analysis approach that 
        specifically accounts for correlations between input variables. It decomposes the variance 
        of the model output into contributions from individual variables and their correlations.
        
        #### Mathematical Formulation:
        
        ANCOVA decomposes the total variance as:
        
        $S_i = S_i^U + S_i^C$
        
        Where:
        - $S_i$ is the total ANCOVA index for variable $i$
        - $S_i^U$ is the uncorrelated (physical) part of the variance due to variable $i$
        - $S_i^C$ is the correlated part of the variance due to correlations between variable $i$ and other variables
        
        #### Interpreting the Results:
        
        - **ANCOVA Index ($S_i$)**: Total sensitivity of the output to each input variable
        - **Uncorrelated Index ($S_i^U$)**: Portion of sensitivity due to the variable's independent effect
        - **Correlated Index ($S_i^C$)**: Portion of sensitivity due to correlations with other variables
        - **Correlation %**: Percentage of the total sensitivity that comes from correlations
        
        Unlike other methods like FAST or standard Sobol indices, ANCOVA specifically accounts for input correlations, 
        making it more appropriate for models with dependent inputs.
        """
        
        # Generate LLM insights if requested
        llm_insights = None
        if model_code_str:
             # Prepare the prompt
             prompt = f"""
             I've performed an ANCOVA (Analysis of Covariance) sensitivity analysis on the following mathematical model:
             ```python
             {model_code_str}
             ```
             
             ANCOVA is a specialized variance-based sensitivity analysis method that explicitly accounts for correlations between input variables. It decomposes the total variance into uncorrelated (physical) and correlated components according to:
             
             S_i = S_i^U + S_i^C
             
             Where:
             - S_i is the total ANCOVA index for variable i
             - S_i^U is the uncorrelated part representing the variable's direct influence
             - S_i^C is the correlated part representing influence due to correlations with other variables
             
             The ANCOVA indices from my analysis are:
             {', '.join([f"{row['Variable']}: S_i = {row['ANCOVA Index']:.4f} (S_i^U = {row['Uncorrelated Index']:.4f}, S_i^C = {row['Correlated Index']:.4f})" for _, row in indices_df.iterrows()])}
             
             The correlation structure between inputs is:
             {', '.join([f"{variable_names[i]}-{variable_names[j]}: {correlation_matrix[i, j]:.4f}" for i in range(dimension) for j in range(i+1, dimension) if abs(correlation_matrix[i, j]) > 0.1])}
             
             Please provide a rigorous scientific analysis addressing:
             
             1. Variable influence hierarchy: Identify the dominant variables and quantify their relative contributions to output uncertainty. Explain how the ANCOVA indices reveal the mechanistic relationships in the model.
             
             2. Correlation effects: Analyze how much of each variable's influence is due to correlations with other inputs. For variables with significant S_i^C values, explain the implications for uncertainty propagation.
             
             3. Uncertainty reduction strategies: Based on the ANCOVA decomposition, recommend specific approaches for reducing output uncertainty. Discuss whether focusing on reducing individual parameter uncertainties or addressing correlation structures would be more effective.
             
             4. Model simplification potential: Evaluate whether any variables could be fixed at nominal values without significantly affecting output uncertainty, based on their ANCOVA indices.
             
             Use precise mathematical language and quantitative statements. Include specific numerical values from the analysis to support your conclusions.
             
             {RETURN_INSTRUCTION}
             """
             # Determine model name (use default if unspecified or 'groq')
             model_name = language_model
             if not language_model or language_model == 'groq':
                 model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
             # Call the LLM with retry logic
             max_attempts = 3
             attempts = 0
             while attempts < max_attempts:
                 try:
                     llm_insights = call_groq_api(prompt, model_name=model_name)
                     break
                 except Exception as e:
                     attempts += 1
                     if attempts >= max_attempts:
                         llm_insights = f"Error generating insights: {str(e)}"
                     import time; time.sleep(2)
        
        # Return all results
        return {
            'indices_df': indices_df,
            'fig_combined': fig_combined,
            'fig_pie': fig_pie,
            'fig_heatmap': fig_heatmap,
            'explanation': ancova_explanation,
            'llm_insights': llm_insights,
            'correlation_matrix': correlation_matrix,
            'variable_names': variable_names,
            'has_copula': has_copula,
            'has_correlation_effects': has_correlation_effects,
            'functional_chaos_result': result
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
    with st.container():
        # Overview
        st.subheader("ANCOVA Sensitivity Analysis Overview")
        st.markdown("""
        ANCOVA (Analysis of Covariance) sensitivity analysis is particularly useful for models with correlated inputs. 
        It separates the variance explained by individual variables from that explained by correlations with other inputs.
        
        #### Mathematical Formulation:
        
        ANCOVA decomposes the total variance as:
        
        $S_i = S_i^U + S_i^C$
        
        Where:
        - $S_i$ is the total ANCOVA index for variable $i$
        - $S_i^U$ is the uncorrelated (physical) part of the variance due to variable $i$
        - $S_i^C$ is the correlated part of the variance due to correlations between variable $i$ and other variables
        
        #### Interpreting the Results:
        
        - **ANCOVA Index ($S_i$)**: Total sensitivity of the output to each input variable
        - **Uncorrelated Index ($S_i^U$)**: Portion of sensitivity due to the variable's independent effect
        - **Correlated Index ($S_i^C$)**: Portion of sensitivity due to correlations with other variables
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
        
        # Display the combined bar chart
        st.markdown("#### ANCOVA Sensitivity Analysis")
        st.markdown("""
        This grouped bar chart shows both the total ANCOVA sensitivity index ($S_i$) and its decomposition into 
        uncorrelated ($S_i^U$) and correlated ($S_i^C$) parts for each variable.
        """)
        st.plotly_chart(ancova_results['fig_combined'], use_container_width=True)
        
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
    if ancova_results['llm_insights']:
        st.subheader("AI-Generated Expert Analysis")
        st.markdown(ancova_results['llm_insights'])

def get_ancova_context_for_chat(ancova_results):
    """
    Generate a formatted string containing ANCOVA analysis results for the global chat context.
    
    Parameters
    ----------
    ancova_results : dict
        Dictionary containing the results of the ANCOVA analysis
        
    Returns
    -------
    str
        Formatted string with ANCOVA analysis results for chat context
    """
    context = ""
    
    # Extract key information from the results
    ancova_indices_df = ancova_results.get("indices_df")
    
    if ancova_indices_df is not None:
        context += "\n\n### ANCOVA Sensitivity Analysis Results\n"
        context += ancova_indices_df.to_markdown(index=False)
    
    return context

def ancova_analysis(model, problem, size=2000, model_code_str=None, language_model=None, display_results=True):
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
    display_results : bool, optional
        Whether to display results using Streamlit UI (default: True)
        Set to False when running in batch mode or "Run All Analyses"
    
    Returns
    -------
    dict
        Dictionary containing the results of the ANCOVA analysis
    """
    try:
        with st.spinner("Running ANCOVA Sensitivity Analysis..."):
            ancova_results = ancova_sensitivity_analysis(
                model, problem, size=size, model_code_str=model_code_str,
                language_model=language_model
            )
            
            # Save results to session state for later access and global chat
            if 'ancova_results' not in st.session_state:
                st.session_state.ancova_results = ancova_results
            
            # Display results if requested
            if display_results:
                display_ancova_results(ancova_results, language_model, model_code_str)
            
            return ancova_results
    except Exception as e:
        if display_results:
            st.error(f"Error in ANCOVA analysis: {str(e)}")
        raise
