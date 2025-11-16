import numpy as np
import pandas as pd
import openturns as ot
from utils.core_utils import call_groq_api, create_chat_interface # Assuming these are correctly defined elsewhere
from utils.constants import RETURN_INSTRUCTION # Assuming this is correctly defined elsewhere
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
        OpenTURNS distribution (typically a JointDistribution defining marginals and correlations)
    size : int, optional
        Number of samples used for the functional chaos experiment and for the ANCOVA evaluation 
        (default is 2000). Consider if different sizes are needed for optimal performance.
    model_code_str : str, optional
        String representation of the model code for documentation and LLM analysis.
    language_model : str, optional
        Language model to use for analysis (e.g., specific Groq model name).
        
    Returns
    -------
    dict
        Dictionary containing the results of the ANCOVA analysis.
    """
    try:
        # Verify input types
        if not isinstance(model, ot.Function):
            raise TypeError("Model must be an OpenTURNS Function")
        if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
            raise TypeError("Problem must be an OpenTURNS Distribution, JointDistribution, or ComposedDistribution")
            
        # Get dimension from the model's input dimension
        dimension = model.getInputDimension()
        
        # Get variable names
        variable_names = []
        for i in range(dimension):
            try:
                marginal = problem.getMarginal(i)
                name = marginal.getDescription()[0] if marginal.getDescription() else f"X{i+1}"
            except Exception: # Fallback if getMarginal or getDescription fails for some custom distribution
                name = f"X{i+1}"
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
                # For Beta, JacobiFactory requires parameters, get them from the distribution
                # Ensure the Beta distribution is correctly parameterized for Jacobi
                alpha = marginal.getAlpha() 
                beta = marginal.getBeta()
                # OpenTURNS Jacobi polynomials are defined on [-1, 1]
                # Beta distribution parameters for standard Jacobi: alpha_jacobi = beta - 1, beta_jacobi = alpha - 1
                polynomialCollection.append(ot.JacobiFactory(beta - 1.0, alpha - 1.0))
            elif isinstance(marginal, ot.Gamma):
                # For Gamma, LaguerreFactory requires parameter k (shape)
                # Ensure the Gamma distribution is correctly parameterized for Laguerre
                # Standard Laguerre polynomials are for k=1 (Exponential). Generalized Laguerre uses k.
                # ot.LaguerreFactory() defaults to k=1. For general Gamma, it's ot.LaguerreFactory(k, 1.0)
                # Assuming standard Laguerre is sufficient or k=1 is implied by ot.LaguerreFactory() default
                # If using specific k from Gamma dist: polynomialCollection.append(ot.LaguerreFactory(marginal.getK()))
                polynomialCollection.append(ot.LaguerreFactory()) # Uses k=1 by default
            else:
                # Default to Legendre for other distributions or if parameters are complex to map
                polynomialCollection.append(ot.LegendreFactory())
        
        productBasis = ot.OrthogonalProductPolynomialFactory(
            polynomialCollection, enumerateFunction
        )
        
        # Create adaptive strategy
        chaos_degree = 4  # Default degree, could be a parameter for advanced users
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
        # This sample comes from the true 'problem' distribution which may have correlations
        X_corr = problem.getSample(size)
        
        # Calculate correlation matrix from the correlated sample
        X_corr_np = np.array(X_corr)
        correlation_matrix = np.identity(dimension) # Initialize with identity
        if dimension > 1:
            correlation_matrix = np.corrcoef(X_corr_np, rowvar=False)
        # Ensure it's a square matrix if dimension is 1 (np.corrcoef might return scalar)
        if dimension == 1 and np.isscalar(correlation_matrix):
            correlation_matrix = np.array([[1.0]])


        # Check if inputs are actually correlated by inspecting the copula
        has_copula_defining_correlation = False
        if dimension > 1:
            try:
                copula = problem.getCopula()
                has_copula_defining_correlation = not isinstance(copula, ot.IndependentCopula)
            except AttributeError: # If problem type doesn't have getCopula (e.g., basic ot.Distribution)
                has_copula_defining_correlation = False
            except Exception: # Other potential errors with getCopula
                 has_copula_defining_correlation = False
        
        indices_data = []
        ancova_calculation_failed_flag = False # Flag to track if ANCOVA specific calculation failed
        
        if has_copula_defining_correlation and dimension > 1:
            # For correlated inputs, use the OpenTURNS ANCOVA class
            try:
                # Perform ANCOVA decomposition
                # Uses chaos result (metamodel from independent inputs) and a sample from the true correlated distribution
                ancova = ot.ANCOVA(result, X_corr)
                
                # Get ANCOVA indices
                ancova_total_indices = ancova.getIndices()      # S_i
                ancova_uncorrelated_indices = ancova.getUncorrelatedIndices() # S_i^U
                
                # Calculate correlated indices: S_i^C = S_i - S_i^U
                # Ensure they are ot.Point for subtraction if not already
                if not isinstance(ancova_total_indices, ot.Point):
                    ancova_total_indices = ot.Point(ancova_total_indices)
                if not isinstance(ancova_uncorrelated_indices, ot.Point):
                    ancova_uncorrelated_indices = ot.Point(ancova_uncorrelated_indices)

                ancova_correlated_indices = ancova_total_indices - ancova_uncorrelated_indices # S_i^C
                
                # Create data for display
                for i, name in enumerate(variable_names):
                    total_index = float(ancova_total_indices[i])
                    uncorrelated_index = float(ancova_uncorrelated_indices[i])
                    correlated_index = float(ancova_correlated_indices[i])
                    
                    corr_percent = (correlated_index / total_index) * 100 if total_index != 0 else 0.0
                    
                    indices_data.append({
                        'Variable': name,
                        'ANCOVA Index': total_index,
                        'Uncorrelated Index': uncorrelated_index,
                        'Correlated Index': correlated_index,
                        'Correlation %': corr_percent
                    })
            except Exception as e:
                # If ANCOVA fails (e.g. numerical issues, unexpected structure), fallback to Sobol indices from functional chaos
                ancova_calculation_failed_flag = True 
                st.warning(f"ANCOVA calculation failed: {str(e)}. Falling back to Sobol' indices from functional chaos (interpreting them as uncorrelated effects).")
                sobol_indices_result = result.getSobolIndices() # These are first-order Sobol' indices
                
                for i, name in enumerate(variable_names):
                    first_order_sobol = sobol_indices_result.getSobolIndex(i)
                    indices_data.append({
                        'Variable': name,
                        'ANCOVA Index': float(first_order_sobol),      # Effectively S_i (Sobol)
                        'Uncorrelated Index': float(first_order_sobol),# S_i^U = S_i (Sobol)
                        'Correlated Index': 0.0,                        # S_i^C = 0
                        'Correlation %': 0.0
                    })
        else:
            # For independent inputs (or single input), ANCOVA indices are equivalent to first-order Sobol indices
            # and correlated effects are zero.
            if dimension > 1 and not has_copula_defining_correlation:
                 st.info("Inputs are treated as independent (no non-Independent copula found or dimension < 2). ANCOVA indices will reflect Sobol first-order effects, and correlated indices will be zero.")

            sobol_indices_result = result.getSobolIndices()
            for i, name in enumerate(variable_names):
                first_order_sobol = sobol_indices_result.getSobolIndex(i)
                indices_data.append({
                    'Variable': name,
                    'ANCOVA Index': float(first_order_sobol),
                    'Uncorrelated Index': float(first_order_sobol),
                    'Correlated Index': 0.0,
                    'Correlation %': 0.0
                })
        
        # Create DataFrame for display
        indices_df = pd.DataFrame(indices_data)
        
        # Sort by ANCOVA index for better visualization
        if not indices_df.empty:
            indices_df = indices_df.sort_values('ANCOVA Index', ascending=False)
        
        # Determine if there are significant correlation effects based on calculated correlated indices
        has_significant_correlation_effects = False
        if not indices_df.empty:
             has_significant_correlation_effects = any(abs(indices_df['Correlated Index']) > 1e-3) # Check for non-negligible correlated indices

        # Create a combined bar chart showing total ANCOVA indices and their decomposition
        fig_combined = go.Figure()
        
        if not indices_df.empty:
            fig_combined.add_trace(go.Bar(
                x=indices_df['Variable'], y=indices_df['ANCOVA Index'],
                name='Total ANCOVA Index (S<sub>i</sub>)',
                marker_color='rgba(55, 83, 109, 0.8)',
                hovertemplate='%{x}: S<sub>i</sub> = %{y:.4f}<extra></extra>'
            ))
            fig_combined.add_trace(go.Bar(
                x=indices_df['Variable'], y=indices_df['Uncorrelated Index'],
                name='Uncorrelated Effect (S<sub>i</sub><sup>U</sup>)',
                marker_color='rgba(31, 119, 180, 0.8)',
                hovertemplate='%{x}: S<sub>i</sub><sup>U</sup> = %{y:.4f}<extra></extra>'
            ))
            fig_combined.add_trace(go.Bar(
                x=indices_df['Variable'], y=indices_df['Correlated Index'],
                name='Correlated Effect (S<sub>i</sub><sup>C</sup>)',
                marker_color='rgba(214, 39, 40, 0.8)', # Red for correlated part
                hovertemplate='%{x}: S<sub>i</sub><sup>C</sup> = %{y:.4f}<extra></extra>'
            ))
        
        fig_combined.add_annotation(
            x=0.02, y=1.12, xref="paper", yref="paper",
            text="ANCOVA Decomposition: S<sub>i</sub> = S<sub>i</sub><sup>U</sup> + S<sub>i</sub><sup>C</sup>",
            showarrow=False, font=dict(size=14), align="left"
        )
        
        fig_combined.update_layout(
            title='ANCOVA Sensitivity Indices: Total, Uncorrelated, and Correlated Effects',
            xaxis_title='Input Variables', yaxis_title='Sensitivity Index Value',
            template='plotly_white', height=600, barmode='group',
            bargap=0.15, bargroupgap=0.1,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=120) 
        )
        
        # Create a pie chart for total ANCOVA indices
        fig_pie = go.Figure()
        if not indices_df.empty and indices_df['ANCOVA Index'].sum() > 1e-6 : # Check for non-zero sum for meaningful pie
             fig_pie = px.pie(
                indices_df, values='ANCOVA Index', names='Variable',
                title='Share of Total ANCOVA Indices (S<sub>i</sub>)',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
             fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        else: # Handle empty or all-zero data for pie chart
            fig_pie.update_layout(title='Share of Total ANCOVA Indices (S<sub>i</sub>) - No data or all zero indices')


        fig_pie.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        # Create improved correlation heatmap
        fig_heatmap = go.Figure()
        if dimension > 0 :
            # Mask for the diagonal
            mask = np.ones_like(correlation_matrix, dtype=bool)
            np.fill_diagonal(mask, False)
            
            # Use actual correlation values for z, but apply mask for display (or make diagonal NaN/transparent)
            # Plotly's heatmap handles NaN by not coloring those cells
            z_display = correlation_matrix.copy()
            if dimension > 1 : # Only make sense to hide diagonal if dim > 1
                 np.fill_diagonal(z_display, np.nan) # Use NaN for diagonal so it's not colored

            abs_max_corr = 1.0
            if dimension > 1:
                valid_corrs = correlation_matrix[mask] # Get off-diagonal elements
                if len(valid_corrs) > 0:
                    abs_max_corr = max(np.max(np.abs(valid_corrs)), 0.1) # Ensure some range

            heatmap_trace = go.Heatmap(
                z=z_display, x=variable_names, y=variable_names,
                colorscale='RdBu_r', zmid=0,
                zmin=-abs_max_corr, zmax=abs_max_corr,
                colorbar=dict(title='Pairwise Correlation'),
                hovertemplate='Corr(%{x}, %{y}): %{z:.4f}<extra></extra>'
            )
            fig_heatmap.add_trace(heatmap_trace)
            
            # Add text annotations for correlation values (off-diagonal)
            if dimension > 1:
                for i in range(dimension):
                    for j in range(dimension):
                        if i != j: 
                            color = 'white' if abs(correlation_matrix[i, j]) > 0.5 * abs_max_corr else 'black'
                            fig_heatmap.add_annotation(
                                x=variable_names[j], y=variable_names[i],
                                text=f"{correlation_matrix[i, j]:.2f}",
                                showarrow=False, font=dict(color=color, size=10)
                            )
        
        fig_heatmap.update_layout(
            title='Input Variable Correlation Matrix',
            template='plotly_white', height=500 if dimension > 1 else 200,
            xaxis=dict(side='bottom'), yaxis=dict(autorange='reversed')
        )
        
        # Create explanatory text
        ancova_explanation = """
        ### ANCOVA Sensitivity Analysis Method
        
        The Analysis of Covariance (ANCOVA) method is a variance-based global sensitivity analysis approach 
        designed for models with **correlated input variables**. It extends the traditional ANOVA/Sobol' 
        decomposition by explicitly accounting for these correlations.
        
        #### Mathematical Formulation:
        
        ANCOVA decomposes the total sensitivity index $S_i$ for each input variable $X_i$ into two parts:
        
        $S_i = S_i^U + S_i^C$
        
        Where:
        - $S_i$: The **total ANCOVA index** for variable $X_i$. It represents the total contribution of $X_i$ to the output variance, including effects due to its correlations with other variables.
        - $S_i^U$: The **uncorrelated (or physical) index**. This part quantifies the direct contribution of $X_i$ to the output variance if $X_i$ were independent of other inputs. It reflects the intrinsic importance of $X_i$.
        - $S_i^C$: The **correlated index**. This part quantifies the contribution to the output variance that arises specifically from the correlations between $X_i$ and other input variables. 
            - A positive $S_i^C$ indicates that correlations involving $X_i$ amplify its overall importance.
            - A negative $S_i^C$ suggests that correlations involving $X_i$ reduce its apparent total importance compared to its physical effect (i.e., $S_i < S_i^U$). This can happen if the correlated effects counteract the direct physical effect.
        
        #### Interpretation:
        
        - **High $S_i$**: Variable $X_i$ is influential on the model output.
        - **High $S_i^U$**: Variable $X_i$ has a strong direct impact, irrespective of correlations.
        - **High absolute $S_i^C$**: The correlations involving $X_i$ significantly alter its influence on the output. Understanding the input correlation structure (see heatmap) is crucial in these cases.
        - **Correlation % ($S_i^C / S_i$)**: The proportion of a variable's total influence that is mediated by its correlations.
        
        ANCOVA provides a more nuanced understanding than methods assuming input independence (like standard Sobol' indices if correlations are present but ignored) by distinguishing between a variable's intrinsic effect and effects due to its dependencies with other inputs. This is achieved by using a polynomial chaos expansion (PCE) built on an independent version of inputs as a surrogate model, and then evaluating ANCOVA indices using samples from the true, correlated input distribution.
        """
        
        # Generate LLM insights if requested
        llm_insights = None
        if model_code_str and language_model: # Ensure language_model is also provided
            # Prepare the prompt
            prompt_header = (
                "You are an expert in sensitivity analysis and uncertainty quantification. "
                "Analyze the following ANCOVA (Analysis of Covariance) results for a mathematical model."
            )

            ancova_context = f"""
            The model code under analysis is:
            ```python
            {model_code_str}
            ```
            
            ANCOVA is a variance-based sensitivity analysis method for models with correlated inputs. It decomposes the total sensitivity index ($S_i$) for each variable into an uncorrelated part ($S_i^U$) and a correlated part ($S_i^C$):
            $S_i = S_i^U + S_i^C$
            - $S_i$: Total ANCOVA index (total influence).
            - $S_i^U$: Uncorrelated/Physical index (direct influence, independent effect).
            - $S_i^C$: Correlated index (influence due to correlations with other inputs).
            """

            if ancova_calculation_failed_flag:
                ancova_context += (
                    "\nIMPORTANT ADVISORY: The primary ANCOVA calculation encountered an issue. "
                    "The reported indices are therefore First-Order Sobol' indices derived from a "
                    "functional chaos expansion (which assumes independent inputs for its construction). "
                    "In this fallback scenario: 'ANCOVA Index' ($S_i$) should be interpreted as the First-Order Sobol' index, "
                    "'Uncorrelated Index' ($S_i^U$) is equal to this $S_i$, and 'Correlated Index' ($S_i^C$) is 0.0. "
                    "Your analysis of correlation effects should primarily rely on the provided input correlation matrix "
                    "and the problem description, rather than the $S_i^C$ values in the table below, which are zero due to the fallback.\n"
                )
            
            indices_summary_str = "No indices data available."
            if not indices_df.empty:
                indices_summary_list = [
                    f"- {row['Variable']}: Total $S_i = {row['ANCOVA Index']:.4f}$ (Uncorrelated $S_i^U = {row['Uncorrelated Index']:.4f}$, Correlated $S_i^C = {row['Correlated Index']:.4f}$, Correlation % = {row['Correlation %']:.2f}%)" 
                    for _, row in indices_df.iterrows()
                ]
                indices_summary_str = "\n".join(indices_summary_list)

            correlation_summary_str = "No significant input correlations reported (or dimension < 2)."
            if dimension > 1:
                significant_corrs = []
                for r in range(dimension):
                    for c in range(r + 1, dimension):
                        if abs(correlation_matrix[r, c]) > 0.1: # Threshold for "significant"
                            significant_corrs.append(f"{variable_names[r]} and {variable_names[c]}: {correlation_matrix[r, c]:.3f}")
                if significant_corrs:
                    correlation_summary_str = "Key input correlations (absolute value > 0.1) are:\n" + "\n".join(significant_corrs)
                else:
                    correlation_summary_str = "No input correlations with absolute value > 0.1 were found."


            prompt = f"""{prompt_header}
            {ancova_context}
            
            ANCOVA Sensitivity Indices:
            {indices_summary_str}
            
            Input Correlation Structure:
            {correlation_summary_str}
            
            Please provide a rigorous scientific analysis, focusing on:
            
            1.  **Dominant Variables & Mechanistic Insights**:
                * Identify the most influential input variables based on their total ANCOVA indices ($S_i$).
                * Discuss the relative importance hierarchy.
                * If possible, relate these sensitivities to potential mechanistic behaviors suggested by the model code or its general form.
            
            2.  **Impact of Correlations ($S_i^C$)**:
                * For variables with notable $S_i^C$ values (both positive and negative), explain how input correlations are modifying their overall influence on the output variance. 
                * Discuss the implications of these correlation-driven effects for understanding uncertainty propagation. (If the fallback note above is active, acknowledge that $S_i^C$ values are 0 and focus on the input correlation matrix).
            
            3.  **Uncertainty Reduction Strategies**:
                * Based on the $S_i$, $S_i^U$, and $S_i^C$ decomposition, what are the most promising strategies for reducing output uncertainty?
                * Should efforts focus on better characterizing individual parameter uncertainties (reducing variance of inputs with high $S_i^U$) or on understanding/modifying the correlation structure between inputs (if $S_i^C$ effects are large)?
            
            4.  **Model Simplification Potential**:
                * Are there any input variables with consistently low $S_i$ values that might be considered for fixing at their nominal values (parameter fixing) without significantly impacting the output uncertainty? Justify your reasoning.
            
            5.  **Overall Assessment**:
                * Provide a concise summary of the sensitivity profile of the model.
                * Highlight any particularly interesting or counter-intuitive findings revealed by the ANCOVA.

            Use precise scientific language, quantitative statements, and refer to specific numerical values from the results to substantiate your conclusions.
            {RETURN_INSTRUCTION} 
            """
            
            # Determine model name
            model_name_for_api = language_model
            if not language_model or language_model.lower() == 'groq': # Handle if 'groq' is passed as a generic identifier
                model_name_for_api = "llama3-70b-8192" # Example: A capable Groq model, adjust as needed. User provided "meta-llama/llama-4-scout-17b-16e-instruct" before.
            
            max_attempts = 3
            attempts = 0
            while attempts < max_attempts:
                try:
                    llm_insights = call_groq_api(prompt, model_name=model_name_for_api)
                    break
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        llm_insights = f"Error generating insights after {max_attempts} attempts: {str(e)}"
                        st.error(llm_insights) # Show error in UI if LLM fails
                    import time; time.sleep(2) # Wait before retrying
        
        return {
            'indices_df': indices_df,
            'fig_combined': fig_combined,
            'fig_pie': fig_pie,
            'fig_heatmap': fig_heatmap,
            'explanation': ancova_explanation,
            'llm_insights': llm_insights,
            'correlation_matrix': correlation_matrix,
            'variable_names': variable_names,
            'has_copula_defining_correlation': has_copula_defining_correlation, # Renamed for clarity
            'has_significant_correlation_effects': has_significant_correlation_effects, # Renamed for clarity
            'functional_chaos_result': result,
            'ancova_calculation_failed_flag': ancova_calculation_failed_flag
        }
    except Exception as e:
        st.error(f"Error in ANCOVA sensitivity_analysis function: {str(e)}")
        # Return a dictionary with empty/default values to prevent downstream errors
        # or re-raise if the calling function should handle it completely.
        # For robustness in a larger app, providing a structured empty result can be better.
        empty_df = pd.DataFrame(columns=['Variable', 'ANCOVA Index', 'Uncorrelated Index', 'Correlated Index', 'Correlation %'])
        error_explanation = f"ANCOVA analysis could not be completed due to an error: {str(e)}"
        return {
            'indices_df': empty_df,
            'fig_combined': go.Figure().update_layout(title="Error: ANCOVA analysis failed"),
            'fig_pie': go.Figure().update_layout(title="Error: ANCOVA analysis failed"),
            'fig_heatmap': go.Figure().update_layout(title="Error: ANCOVA analysis failed"),
            'explanation': error_explanation,
            'llm_insights': "LLM insights could not be generated due to an analysis error.",
            'correlation_matrix': np.array([]),
            'variable_names': [],
            'has_copula_defining_correlation': False,
            'has_significant_correlation_effects': False,
            'functional_chaos_result': None,
            'ancova_calculation_failed_flag': True # Mark as failed
        }


def display_ancova_results(ancova_results, language_model=None, model_code_str=None):
    """
    Display ANCOVA sensitivity analysis results in the Streamlit interface.
    
    Parameters
    ----------
    ancova_results : dict
        Dictionary containing the results of the ANCOVA analysis.
    language_model : str, optional
        Language model used for analysis (passed for context, not directly used here).
    model_code_str : str, optional
        String representation of the model code (passed for context).
    """
    st.header("📈 ANCOVA Sensitivity Analysis Results")

    # Display the method explanation from the results
    if 'explanation' in ancova_results:
        st.markdown(ancova_results['explanation'])
    else:
        st.markdown("ANCOVA (Analysis of Covariance) method explanation not available.")

    if ancova_results.get('ancova_calculation_failed_flag', False):
        st.warning(
            "**Note:** The primary ANCOVA calculation encountered an issue. "
            "The displayed sensitivity indices are First-Order Sobol' indices from the underlying "
            "functional chaos expansion. 'Correlated Index' values are consequently zero. "
            "Interpret results with this advisory in mind."
        )
    
    # Display the indices table and metrics
    st.subheader("📊 Sensitivity Indices & Metrics")
    
    indices_df = ancova_results.get('indices_df')
    
    if indices_df is not None and not indices_df.empty:
        most_influential_var = indices_df.iloc[0]['Variable']
        most_influential_idx = indices_df.iloc[0]['ANCOVA Index']
        
        sum_ancova = indices_df['ANCOVA Index'].sum()
        sum_correlated = indices_df['Correlated Index'].sum()
        
        # Overall correlation effect: sum of absolute correlated indices / sum of absolute total indices
        # This gives a sense of how much correlation contributes overall.
        # Or, sum_correlated / sum_ancova if we care about net effect. Let's use sum_correlated/sum_ancova for now.
        overall_correlation_effect_ratio = (sum_correlated / sum_ancova) if sum_ancova != 0 else 0.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "👑 Most Influential", 
                most_influential_var,
                f"$S_i$: {most_influential_idx:.3f}"
            )
        with col2:
            st.metric("∑ $S_i$ (Total ANCOVA Indices)", f"{sum_ancova:.3f}")
        with col3:
            st.metric(
                r"Net Correlation Effect (${\sum S_i^C} / {\sum S_i}$)",
                f"{overall_correlation_effect_ratio:.2%}",
                help="Ratio of the sum of correlated indices to the sum of total ANCOVA indices. Indicates net impact of correlations."
            )
            
        st.markdown("#### Detailed Numerical Results:")
        display_df_st = indices_df[['Variable', 'ANCOVA Index', 'Uncorrelated Index', 'Correlated Index', 'Correlation %']].copy()
        display_df_st['ANCOVA Index'] = display_df_st['ANCOVA Index'].map('{:.4f}'.format)
        display_df_st['Uncorrelated Index'] = display_df_st['Uncorrelated Index'].map('{:.4f}'.format)
        display_df_st['Correlated Index'] = display_df_st['Correlated Index'].map('{:.4f}'.format)
        display_df_st['Correlation %'] = display_df_st['Correlation %'].map('{:.2f}%'.format)
        st.dataframe(display_df_st, width='stretch', hide_index=True)
        
        # Visualizations
        st.subheader("🎨 Sensitivity Visualizations")
        
        st.markdown("##### ANCOVA Indices Decomposition ($S_i = S_i^U + S_i^C$)")
        st.markdown("""
        This grouped bar chart shows the total ANCOVA sensitivity index ($S_i$) for each variable, 
        decomposed into its uncorrelated part ($S_i^U$ - direct physical effect) and its 
        correlated part ($S_i^C$ - effect due to correlations with other inputs).
        """)
        st.plotly_chart(ancova_results['fig_combined'], width='stretch')
        
        # Pie chart and heatmap in columns
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("##### Share of Total ANCOVA Indices ($S_i$)")
            st.markdown("""
            This pie chart illustrates the relative contribution of each input variable's total ANCOVA index ($S_i$) 
            to the sum of all total indices. Larger slices indicate a greater share of influence.
            """)
            st.plotly_chart(ancova_results['fig_pie'], width='stretch')
        
        with col_viz2:
            st.markdown("##### Input Variable Correlation Matrix")
            st.markdown("""
            This heatmap visualizes the Pearson correlation coefficients between input variables.
            - <font color='red'>Red</font>: Positive correlation.
            - <font color='blue'>Blue</font>: Negative correlation.
            - Intensity: Darker colors indicate stronger correlations.
            This context is vital for interpreting the correlated indices ($S_i^C$).
            """)
            st.plotly_chart(ancova_results['fig_heatmap'], width='stretch')
            
        # Add interpretation based on results
        has_sig_corr_effects = ancova_results.get('has_significant_correlation_effects', False)
        if has_sig_corr_effects and not ancova_results.get('ancova_calculation_failed_flag', False) :
            st.info(f"""
            💡 **Significant Correlation Effects Detected**: The analysis indicates that correlations between input variables 
            play a non-negligible role in the model's output variance (as shown by non-zero $S_i^C$ values).
            The behavior of the model is influenced by the joint distribution of inputs, not just their individual effects.
            Refer to the $S_i^C$ values and the correlation heatmap for specific details.
            """)
        elif not ancova_results.get('ancova_calculation_failed_flag', False):
             st.success("""
            ✅ **Minimal Net Correlation Effects Detected**: The $S_i^C$ values are generally small, suggesting that the 
            direct, uncorrelated effects ($S_i^U$) are the primary drivers of each variable's total sensitivity ($S_i$). 
            Input correlations, if present, do not strongly mediate the output variance through these variables.
            """)

    else:
        st.warning("ANCOVA results (indices dataframe) are not available or empty. Cannot display detailed metrics or plots.")

    # AI Insights Section
    if ancova_results.get('llm_insights'):
        st.subheader("🤖 AI-Generated Expert Analysis")
        st.markdown(ancova_results['llm_insights'])
    elif model_code_str and language_model: # If it was supposed to run but didn't (e.g. error before LLM call)
        st.markdown("AI-generated insights are not available for this run.")


def get_ancova_context_for_chat(ancova_results):
    """
    Generate a formatted string containing ANCOVA analysis results for the global chat context.
    
    Parameters
    ----------
    ancova_results : dict
        Dictionary containing the results of the ANCOVA analysis.
        
    Returns
    -------
    str
        Formatted string with ANCOVA analysis results for chat context.
    """
    context = ""
    
    indices_df = ancova_results.get("indices_df")
    
    if indices_df is not None and not indices_df.empty:
        context += "\n\n### ANCOVA Sensitivity Analysis Summary\n"
        context += "Key findings from ANCOVA (Analysis of Covariance):\n"
        if ancova_results.get('ancova_calculation_failed_flag', False):
            context += "**Note:** ANCOVA calculation failed; results are Sobol first-order indices.\n"
        
        # Summary of top variables
        top_n = min(3, len(indices_df))
        context += f"Top {top_n} influential variables (by Total ANCOVA Index $S_i$):\n"
        for i in range(top_n):
            row = indices_df.iloc[i]
            context += (
                f"- {row['Variable']}: $S_i={row['ANCOVA Index']:.3f}$ "
                f"($S_i^U={row['Uncorrelated Index']:.3f}$, $S_i^C={row['Correlated Index']:.3f}$)\n"
            )
        
        # Overall correlation impact
        if ancova_results.get('has_significant_correlation_effects') and not ancova_results.get('ancova_calculation_failed_flag'):
            context += "Significant correlation effects were observed.\n"
        elif not ancova_results.get('ancova_calculation_failed_flag'):
            context += "Correlation effects on sensitivities appear to be minimal.\n"

        # Provide table for more detail if needed by chat
        context += "\nFull ANCOVA Indices Table:\n"
        context += indices_df.to_markdown(index=False, floatfmt=".4f")
    else:
        context += "\n\nANCOVA Sensitivity Analysis results are not available or are empty."
    
    return context

def ancova_analysis(model, problem, size=2000, model_code_str=None, language_model=None, display_results=True):
    """
    Perform and optionally display ANCOVA sensitivity analysis.
    
    Main entry point for ANCOVA, handling calculation and Streamlit UI display.
    
    Parameters
    ----------
    model : ot.Function
        OpenTURNS function.
    problem : ot.Distribution
        OpenTURNS distribution (can be correlated).
    size : int, optional
        Sample size (default 2000).
    model_code_str : str, optional
        String of the model's code for LLM.
    language_model : str, optional
        LLM to use.
    display_results : bool, optional
        If True (default), display results in Streamlit. Set to False for batch runs.
    
    Returns
    -------
    dict
        ANCOVA results dictionary.
    """
    results_placeholder = None
    if display_results:
        results_placeholder = st.empty() # Placeholder for spinner message
        results_placeholder.info("🚀 Starting ANCOVA Sensitivity Analysis...")

    try:
        ancova_results_data = ancova_sensitivity_analysis(
            model, problem, size=size, model_code_str=model_code_str,
            language_model=language_model
        )
        
        if display_results and results_placeholder:
            results_placeholder.success("✅ ANCOVA Analysis Completed!")
            
        # Save results to session state for potential cross-page/module access or chat
        st.session_state.ancova_results = ancova_results_data 
        
        if display_results:
            display_ancova_results(ancova_results_data, language_model, model_code_str)
            
        return ancova_results_data
        
    except Exception as e:
        # This top-level exception will catch errors from ancova_sensitivity_analysis if not handled there,
        # or from display_ancova_results.
        error_message = f"Critical Error in ANCOVA analysis workflow: {str(e)}"
        if display_results:
            if results_placeholder:
                results_placeholder.error(error_message)
            else:
                st.error(error_message)
        else:
            print(error_message) # Log to console if not in Streamlit display mode
        
        # Ensure a structured (empty/error) result is returned to prevent crashes in calling code
        # if ancova_sensitivity_analysis itself crashes before returning its own error structure.
        empty_df = pd.DataFrame(columns=['Variable', 'ANCOVA Index', 'Uncorrelated Index', 'Correlated Index', 'Correlation %'])
        return {
            'indices_df': empty_df,
            'fig_combined': go.Figure().update_layout(title=error_message),
            'fig_pie': go.Figure().update_layout(title=error_message),
            'fig_heatmap': go.Figure().update_layout(title=error_message),
            'explanation': f"ANCOVA analysis failed: {error_message}",
            'llm_insights': "LLM insights could not be generated due to a critical error.",
            'correlation_matrix': np.array([]),
            'variable_names': [],
            'has_copula_defining_correlation': False,
            'has_significant_correlation_effects': False,
            'functional_chaos_result': None,
            'ancova_calculation_failed_flag': True
        }
