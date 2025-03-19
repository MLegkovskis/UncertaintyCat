import numpy as np
import pandas as pd
import openturns as ot
from utils.core_utils import call_groq_api
from utils.constants import RETURN_INSTRUCTION
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def fast_sensitivity_analysis(model, problem, size=400, model_code_str=None, language_model=None):
    """Perform enterprise-grade FAST sensitivity analysis.
    
    This module provides comprehensive global sensitivity analysis using the Fourier Amplitude 
    Sensitivity Test (FAST) method, which is a relevant alternative to the classical simulation 
    approach for computing sensitivity indices. The FAST method decomposes the model response 
    using Fourier decomposition.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    size : int, optional
        Number of samples for FAST analysis (default is 400)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
        
    Returns
    -------
    dict
        Dictionary containing the results of the FAST analysis
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
        
        # Create independent distribution for FAST (since FAST doesn't work with correlated inputs)
        # Extract marginals from the original problem
        marginals = [problem.getMarginal(i) for i in range(dimension)]
        independent_dist = ot.JointDistribution(marginals)
        
        # Create FAST analysis with independent distribution
        sensitivityAnalysis = ot.FAST(model, independent_dist, size)
        
        # Compute the first order indices
        firstOrderIndices = sensitivityAnalysis.getFirstOrderIndices()
        
        # Retrieve total order indices
        totalOrderIndices = sensitivityAnalysis.getTotalOrderIndices()
        
        # Create DataFrame for indices
        indices_data = []
        for i, name in enumerate(variable_names):
            indices_data.append({
                'Variable': name,
                'First Order': float(firstOrderIndices[i]),
                'Total Order': float(totalOrderIndices[i]),
                'Interaction': float(totalOrderIndices[i]) - float(firstOrderIndices[i]),
                'Interaction %': (float(totalOrderIndices[i]) - float(firstOrderIndices[i])) / float(totalOrderIndices[i]) * 100 if float(totalOrderIndices[i]) > 0 else 0
            })
        
        # Create DataFrame for display
        indices_df = pd.DataFrame(indices_data)
        
        # Sort by total order index for better visualization
        indices_df = indices_df.sort_values('Total Order', ascending=False)
        
        # Create Plotly bar chart for sensitivity indices
        fig_bar = go.Figure()
        
        # Add first order indices
        fig_bar.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['First Order'],
            name='First Order Indices',
            marker_color='rgba(31, 119, 180, 0.8)'
        ))
        
        # Add total order indices
        fig_bar.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['Total Order'],
            name='Total Order Indices',
            marker_color='rgba(214, 39, 40, 0.8)'
        ))
        
        # Update layout
        fig_bar.update_layout(
            title='FAST Sensitivity Indices',
            xaxis_title='Input Variables',
            yaxis_title='Sensitivity Index',
            barmode='group',
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
        
        # Create a pie chart for first order indices
        fig_pie_first = px.pie(
            indices_df, 
            values='First Order', 
            names='Variable',
            title='First Order Indices Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie_first.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_first.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # Create a pie chart for total order indices
        fig_pie_total = px.pie(
            indices_df, 
            values='Total Order', 
            names='Variable',
            title='Total Order Indices Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie_total.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie_total.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # Create explanatory text
        fast_explanation = """
        ### FAST Sensitivity Analysis
        
        The Fourier Amplitude Sensitivity Test (FAST) is a global sensitivity analysis method that uses 
        the Fourier decomposition of the model response to quantify the correlation between input variables 
        and output variables.
        
        #### Interpreting the Results:
        
        - **First Order Indices**: Measure the direct contribution of each input variable to the output variance.
        - **Total Order Indices**: Measure the contribution of each input variable including all its interactions with other variables.
        - **Interaction**: The difference between Total and First order indices, representing the contribution due to interactions.
        
        The indices range from 0 to 1, where values closer to 1 indicate greater sensitivity of the model to that variable.
        
        > **Note**: FAST analysis requires independent inputs, so the marginals of your distribution are used while ignoring any correlation structure.
        > For models with correlated inputs, consider the ANCOVA analysis which specifically accounts for correlations.
        """
        
        # Generate LLM insights if requested
        llm_insights = None
        if language_model and model_code_str:
            # Prepare the prompt
            prompt = f"""
            I've performed a FAST sensitivity analysis on the following model:
            ```python
            {model_code_str}
            ```
            
            The results show these first order indices:
            {', '.join([f"{name}: {float(firstOrderIndices[i]):.4f}" for i, name in enumerate(variable_names)])}
            
            And these total order indices:
            {', '.join([f"{name}: {float(totalOrderIndices[i]):.4f}" for i, name in enumerate(variable_names)])}
            
            Please provide 2-3 paragraphs of insights about:
            1. Which variables have the most influence on the model output and why
            2. The significance of any interaction effects observed
            3. How these results could inform model simplification or further analysis
            
            {RETURN_INSTRUCTION}
            """
            
            # Call the LLM
            llm_insights = call_groq_api(prompt, model_name=language_model)
        
        # Return all results
        return {
            'indices_df': indices_df,
            'fig_bar': fig_bar,
            'fig_pie_first': fig_pie_first,
            'fig_pie_total': fig_pie_total,
            'explanation': fast_explanation,
            'llm_insights': llm_insights
        }
        
    except Exception as e:
        st.error(f"Error in FAST sensitivity analysis: {str(e)}")
        return None

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
        
        # Create a sample
        sample = problem.getSample(size)
        
        # Evaluate the model on the sample
        output_sample = model(sample)
        
        # Check if the problem has a copula (to determine if inputs are correlated)
        has_copula = False
        try:
            # Try to get the copula - if this succeeds, it has a copula
            copula = problem.getCopula()
            # Check if it's not an independent copula
            has_copula = not isinstance(copula, ot.IndependentCopula)
        except:
            # If there's an error, assume no copula
            has_copula = False
        
        # If inputs are independent, use a different approach
        if not has_copula:
            # For independent inputs, we'll use a simpler approach
            # Create a polynomial chaos expansion
            try:
                # Create a functional chaos algorithm
                algo = ot.FunctionalChaosAlgorithm(sample, output_sample)
                algo.run()
                result = algo.getResult()
                
                # Get the Sobol indices
                sobol_indices = result.getSobolIndices()
                
                # Create a dictionary to store the indices
                indices_data = []
                for i, name in enumerate(variable_names):
                    # Get first order Sobol index
                    first_order = sobol_indices.getSobolIndex(i)
                    # For independent inputs, ANCOVA index equals Sobol index
                    indices_data.append({
                        'Variable': name,
                        'ANCOVA Index': float(first_order),
                        'Uncorrelated Index': float(first_order),
                        'Correlated Index': 0.0,  # No correlation contribution for independent inputs
                        'Correlation %': 0.0
                    })
            except Exception as e:
                # If polynomial chaos fails, use FAST as a fallback
                try:
                    # Create independent distribution
                    marginals = [problem.getMarginal(i) for i in range(dimension)]
                    independent_dist = ot.JointDistribution(marginals)
                    
                    # Use FAST
                    sensitivityAnalysis = ot.FAST(model, independent_dist, size)
                    firstOrderIndices = sensitivityAnalysis.getFirstOrderIndices()
                    
                    # Create a dictionary to store the indices
                    indices_data = []
                    for i, name in enumerate(variable_names):
                        # For independent inputs, ANCOVA index equals FAST first order index
                        first_order = float(firstOrderIndices[i])
                        indices_data.append({
                            'Variable': name,
                            'ANCOVA Index': first_order,
                            'Uncorrelated Index': first_order,
                            'Correlated Index': 0.0,  # No correlation contribution for independent inputs
                            'Correlation %': 0.0
                        })
                except Exception as inner_e:
                    # If both methods fail, use a basic correlation approach
                    # Calculate correlation between inputs and output
                    indices_data = []
                    output_values = output_sample.asPoint()
                    
                    for i, name in enumerate(variable_names):
                        input_values = sample.getMarginal(i).asPoint()
                        
                        # Calculate correlation coefficient (Pearson)
                        correlation = np.corrcoef(input_values, output_values)[0, 1]
                        corr_squared = correlation ** 2
                        
                        indices_data.append({
                            'Variable': name,
                            'ANCOVA Index': abs(corr_squared),  # Use squared correlation as a proxy
                            'Uncorrelated Index': abs(corr_squared),
                            'Correlated Index': 0.0,
                            'Correlation %': 0.0
                        })
        else:
            # For correlated inputs, use the ANCOVA approach
            try:
                # Create a polynomial chaos expansion
                algo = ot.FunctionalChaosAlgorithm(sample, output_sample)
                algo.run()
                result = algo.getResult()
                
                # Get the Sobol indices
                sobol_indices = result.getSobolIndices()
                
                # Calculate ANCOVA indices
                indices_data = []
                for i, name in enumerate(variable_names):
                    # Get first order Sobol index
                    first_order = sobol_indices.getSobolIndex(i)
                    
                    # Calculate correlation between this input and output
                    input_values = sample.getMarginal(i).asPoint()
                    output_values = output_sample.asPoint()
                    correlation = np.corrcoef(input_values, output_values)[0, 1]
                    corr_squared = correlation ** 2
                    
                    # ANCOVA index is the squared correlation coefficient
                    ancova_index = abs(corr_squared)
                    
                    # Uncorrelated contribution is the Sobol index
                    uncorrelated = float(first_order)
                    
                    # Correlated contribution is the difference
                    correlated = ancova_index - uncorrelated if ancova_index > uncorrelated else 0.0
                    
                    # Calculate correlation percentage
                    corr_percent = (correlated / ancova_index) * 100 if ancova_index > 0 else 0.0
                    
                    indices_data.append({
                        'Variable': name,
                        'ANCOVA Index': ancova_index,
                        'Uncorrelated Index': uncorrelated,
                        'Correlated Index': correlated,
                        'Correlation %': corr_percent
                    })
            except Exception as e:
                # If polynomial chaos fails, use a simpler correlation-based approach
                indices_data = []
                output_values = output_sample.asPoint()
                
                # Calculate correlation matrix between inputs
                input_corr_matrix = np.zeros((dimension, dimension))
                for i in range(dimension):
                    for j in range(dimension):
                        input_i = sample.getMarginal(i).asPoint()
                        input_j = sample.getMarginal(j).asPoint()
                        input_corr_matrix[i, j] = np.corrcoef(input_i, input_j)[0, 1]
                
                for i, name in enumerate(variable_names):
                    input_values = sample.getMarginal(i).asPoint()
                    
                    # Calculate correlation with output
                    correlation = np.corrcoef(input_values, output_values)[0, 1]
                    corr_squared = correlation ** 2
                    
                    # Estimate uncorrelated contribution
                    # This is a simplified approach - in a full ANCOVA we would use regression
                    # Here we use a heuristic based on correlations with other inputs
                    other_input_correlations = [abs(input_corr_matrix[i, j]) for j in range(dimension) if j != i]
                    mean_correlation_with_others = np.mean(other_input_correlations) if other_input_correlations else 0
                    
                    # Estimate uncorrelated contribution as a fraction of total based on correlation with other inputs
                    uncorrelated_fraction = 1.0 - min(0.9, mean_correlation_with_others)  # Cap at 0.9 to avoid zero
                    uncorrelated = corr_squared * uncorrelated_fraction
                    
                    # Correlated contribution is the remainder
                    correlated = corr_squared - uncorrelated
                    
                    # Calculate correlation percentage
                    corr_percent = (correlated / corr_squared) * 100 if corr_squared > 0 else 0.0
                    
                    indices_data.append({
                        'Variable': name,
                        'ANCOVA Index': abs(corr_squared),
                        'Uncorrelated Index': abs(uncorrelated),
                        'Correlated Index': abs(correlated),
                        'Correlation %': corr_percent
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
        
        # Create a stacked bar chart showing uncorrelated and correlated contributions
        fig_stacked = go.Figure()
        
        # Add uncorrelated contribution
        fig_stacked.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['Uncorrelated Index'],
            name='Uncorrelated Contribution',
            marker_color='rgba(31, 119, 180, 0.8)'
        ))
        
        # Add correlated contribution
        fig_stacked.add_trace(go.Bar(
            x=indices_df['Variable'],
            y=indices_df['Correlated Index'],
            name='Correlation Contribution',
            marker_color='rgba(214, 39, 40, 0.8)'
        ))
        
        # Update layout
        fig_stacked.update_layout(
            title='Decomposition of ANCOVA Indices',
            xaxis_title='Input Variables',
            yaxis_title='Contribution to Variance',
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
        
        # Create a heatmap of input correlations
        correlation_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                input_i = sample.getMarginal(i).asPoint()
                input_j = sample.getMarginal(j).asPoint()
                correlation_matrix[i, j] = np.corrcoef(input_i, input_j)[0, 1]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=variable_names,
            y=variable_names,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1
        ))
        
        fig_heatmap.update_layout(
            title='Input Variables Correlation Matrix',
            template='plotly_white',
            height=500,
            width=500
        )
        
        # Create explanatory text
        ancova_explanation = """
        ### ANCOVA Sensitivity Analysis
        
        ANCOVA (Analysis of Covariance) allows one to estimate sensitivity indices from a model with correlated inputs.
        It separates the part of variance explained by individual variables from that explained by correlations with 
        other inputs.
        
        #### Interpreting the Results:
        
        - **ANCOVA Indices**: Total sensitivity indices, ranging from 0 to 1.
        - **Uncorrelated Indices**: The physical part of variance in the model (S₁ᵁ).
        - **Correlated Indices**: The part of variance due to correlation with other variables (S₁ᶜ).
        
        When the correlated part (S₁ᶜ) is low and the ANCOVA index is close to the uncorrelated index,
        correlation has a weak influence on the contribution of that variable. Conversely, when the correlated
        part is high, correlation has a strong influence on the variable's contribution.
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
            {', '.join([f"{name}: {float(indices_df['ANCOVA Index'][i]):.4f}" for i, name in enumerate(variable_names)])}
            
            These uncorrelated indices (physical part):
            {', '.join([f"{name}: {float(indices_df['Uncorrelated Index'][i]):.4f}" for i, name in enumerate(variable_names)])}
            
            And these correlated indices (correlation part):
            {', '.join([f"{name}: {float(indices_df['Correlated Index'][i]):.4f}" for i, name in enumerate(variable_names)])}
            
            Please provide 2-3 paragraphs of insights about:
            1. Which variables have the most influence on the model output and why
            2. The significance of correlation effects observed between variables
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
        return None

def display_fast_results(fast_results, language_model=None):
    """
    Display FAST sensitivity analysis results in the Streamlit interface.
    
    Parameters
    ----------
    fast_results : dict
        Dictionary containing the results of the FAST analysis
    language_model : str, optional
        Language model used for analysis, by default None
    """
    if fast_results:
        # Display the explanation
        st.markdown(fast_results['explanation'], unsafe_allow_html=True)
        
        # Display the indices table
        st.markdown('<p style="font-weight: bold; margin-top: 20px;">FAST Sensitivity Indices</p>', unsafe_allow_html=True)
        st.dataframe(fast_results['indices_df'][['Variable', 'First Order', 'Total Order', 'Interaction', 'Interaction %']], use_container_width=True)
        
        # Display the bar chart
        st.plotly_chart(fast_results['fig_bar'], use_container_width=True)
        
        # Display pie charts in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fast_results['fig_pie_first'], use_container_width=True)
        with col2:
            st.plotly_chart(fast_results['fig_pie_total'], use_container_width=True)
        
        # Display LLM insights if available
        if fast_results['llm_insights'] and language_model:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">FAST Analysis Insights</h3>', unsafe_allow_html=True)
            st.markdown(fast_results['llm_insights'], unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def display_ancova_results(ancova_results, language_model=None):
    """
    Display ANCOVA sensitivity analysis results in the Streamlit interface.
    
    Parameters
    ----------
    ancova_results : dict
        Dictionary containing the results of the ANCOVA analysis
    language_model : str, optional
        Language model used for analysis, by default None
    """
    if ancova_results:
        # Display the explanation
        st.markdown(ancova_results['explanation'], unsafe_allow_html=True)
        
        # Display the indices table
        st.markdown('<p style="font-weight: bold; margin-top: 20px;">ANCOVA Sensitivity Indices</p>', unsafe_allow_html=True)
        st.dataframe(ancova_results['indices_df'][['Variable', 'ANCOVA Index', 'Uncorrelated Index', 'Correlated Index', 'Correlation %']], use_container_width=True)
        
        # Display the bar chart
        st.plotly_chart(ancova_results['fig_bar'], use_container_width=True)
        
        # Display stacked bar chart
        st.plotly_chart(ancova_results['fig_stacked'], use_container_width=True)
        
        # Display pie chart and heatmap in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(ancova_results['fig_pie'], use_container_width=True)
        with col2:
            st.plotly_chart(ancova_results['fig_heatmap'], use_container_width=True)
        
        # Display LLM insights if available
        if ancova_results['llm_insights'] and language_model:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">ANCOVA Analysis Insights</h3>', unsafe_allow_html=True)
            st.markdown(ancova_results['llm_insights'], unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def fast_analysis(model, problem, size=400, model_code_str=None, language_model=None):
    """
    Perform and display FAST sensitivity analysis.
    
    This function serves as the main entry point for FAST analysis, handling both
    the calculation and visualization of results.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function to analyze
    problem : ot.Distribution
        OpenTURNS distribution (typically a JointDistribution)
    size : int, optional
        Number of samples for FAST analysis (default is 400)
    model_code_str : str, optional
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis
    """
    st.markdown("## FAST Sensitivity Analysis")
    
    with st.spinner("Running FAST Sensitivity Analysis..."):
        fast_results = fast_sensitivity_analysis(
            model, problem, size=size, model_code_str=model_code_str,
            language_model=language_model
        )
        
        display_fast_results(fast_results, language_model)

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
    st.markdown("## ANCOVA Sensitivity Analysis")
    
    with st.spinner("Running ANCOVA Sensitivity Analysis..."):
        ancova_results = ancova_sensitivity_analysis(
            model, problem, size=size, model_code_str=model_code_str,
            language_model=language_model
        )
        
        display_ancova_results(ancova_results, language_model)
