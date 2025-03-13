import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import traceback
import scipy.stats as stats
from utils.core_utils import call_groq_api
from utils.constants import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model, sample_inputs

def hsic_analysis(model, problem, model_code_str, language_model='groq'):
    """
    Perform HSIC (Hilbert-Schmidt Independence Criterion) analysis with enterprise-grade visualizations.
    
    This function calculates HSIC-based sensitivity indices to measure the dependence between
    input variables and model outputs, including both raw and normalized indices.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    model_code_str : str
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis, by default "groq"
        
    Returns
    -------
    dict
        Dictionary containing HSIC analysis results
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
        
        # Display the header and introduction
        st.markdown("## HSIC Analysis")
        st.markdown("""
        HSIC (Hilbert-Schmidt Independence Criterion) analysis is a powerful method for detecting non-linear dependencies 
        between input variables and model outputs. Unlike correlation-based methods, HSIC can capture complex, non-linear 
        relationships without making assumptions about the underlying model structure.
        """)
        
        # Create expandable section for methodology
        with st.expander("HSIC Methodology Explained", expanded=False):
            st.markdown("""
            ### HSIC Methodology
            
            The Hilbert-Schmidt Independence Criterion (HSIC) is a kernel-based measure of dependence between random variables. 
            It quantifies the distance between the joint distribution of two random variables and the product of their marginal 
            distributions using reproducing kernel Hilbert spaces (RKHS).
            
            #### Mathematical Foundation
            
            For random variables X and Y with joint distribution P_XY and marginal distributions P_X and P_Y:
            
            1. **HSIC Definition**: HSIC(X, Y) = ||C_XY||²_HS
               - Where C_XY is the cross-covariance operator in the RKHS
               - ||·||²_HS is the Hilbert-Schmidt norm
            
            2. **Empirical Estimation**: For samples {(x₁, y₁), ..., (xₙ, yₙ)}:
               - HSIC(X, Y) ≈ (1/n²) · Tr(KHLH)
               - K and L are kernel matrices for X and Y
               - H is the centering matrix (I - 1/n · 11ᵀ)
            
            3. **Kernel Choice**: Typically Gaussian kernels are used:
               - k(x, x') = exp(-||x - x'||²/2σ²)
               - The bandwidth parameter σ is often set to the median distance
            
            #### Advantages of HSIC
            
            - Detects both linear and non-linear dependencies
            - Makes no assumptions about the distribution of variables
            - Can handle high-dimensional data
            - Provides a consistent test of independence
            
            #### Interpretation
            
            - **Raw HSIC Values**: Measure the absolute strength of dependence
            - **Normalized HSIC (R²-HSIC)**: Relative importance of each input
            - **p-values**: Statistical significance of the dependence
            """)
        
        # Create expandable section for interpreting results
        with st.expander("Understanding HSIC Results", expanded=False):
            st.markdown("""
            ### Interpreting HSIC Results
            
            #### Raw HSIC Indices
            
            The raw HSIC index for an input variable measures the absolute strength of its dependence with the output:
            
            - **Higher values** indicate stronger dependence (linear or non-linear)
            - Raw values depend on the kernel and data scaling, making direct comparison difficult
            
            #### Normalized R²-HSIC Indices
            
            Normalized indices (R²-HSIC) represent the relative importance of each input variable:
            
            - **Scale**: 0 to 1 (or 0% to 100%)
            - **Interpretation**: Proportion of the total dependence explained by each input
            - **Sum**: All normalized indices sum to 1 (or 100%)
            
            #### p-values
            
            p-values determine the statistical significance of the dependence:
            
            - **p < 0.05**: Statistically significant dependence (95% confidence)
            - **p < 0.01**: Highly significant dependence (99% confidence)
            - **p > 0.05**: No significant evidence of dependence
            
            Two types of p-values are calculated:
            
            1. **Asymptotic p-value**: Based on asymptotic distribution (faster but less accurate)
            2. **Permutation p-value**: Based on permutation tests (more accurate but computationally intensive)
            """)
        
        # Compute HSIC indices with progress indicator
        with st.spinner("Computing HSIC indices..."):
            hsic_results = compute_hsic_indices(model, problem, N=1000, seed=42)
        
        # Create DataFrame from the results
        hsic_df = create_hsic_dataframe(hsic_results)
        
        # Create visualizations using Plotly
        fig = create_hsic_plots(hsic_results)
        
        # Display the interactive plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the results table
        st.markdown("### HSIC Indices")
        st.dataframe(hsic_df.style.format({
            'HSIC_Index': '{:.6e}',
            'Normalized_Index': '{:.4f}',
            'p_value_asymptotic': '{:.4e}',
            'p_value_permutation': '{:.4e}'
        }), use_container_width=True)
        
        # Display interpretation
        st.markdown("### Interpretation")
        
        # Find most influential variables
        top_var = hsic_df.sort_values('Normalized_Index', ascending=False).iloc[0]
        significant_vars = hsic_df[hsic_df['p_value_asymptotic'] < 0.05]
        
        st.markdown(f"""
        **Key Insights:**
        
        - **Most influential variable**: {top_var['Variable']} (Normalized HSIC: {top_var['Normalized_Index']:.4f})
        - **Statistically significant variables**: {len(significant_vars)} out of {len(hsic_df)} variables
        - **Total captured dependence**: The HSIC analysis captures both linear and non-linear dependencies
        
        **Significance Analysis:**
        
        - Variables with p-values < 0.05 have statistically significant relationships with the output
        - Lower p-values indicate stronger evidence against the null hypothesis of independence
        """)
        
        # Generate AI insights
        with st.expander("Expert Analysis", expanded=True):
            st.markdown("### AI-Generated Expert Analysis")
            
            # Prepare the data for the API call
            hsic_md_table = hsic_df.to_markdown(index=False, floatfmt=".4e")
            
            # Format the model code for inclusion in the prompt
            model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])
            
            # Prepare the inputs description
            input_parameters = []
            for i in range(dimension):
                marginal = problem.getMarginal(i)
                name = input_names[i]
                dist_type = marginal.__class__.__name__
                params = marginal.getParameter()
                input_parameters.append(f"- **{name}**: {dist_type} distribution with parameters {list(params)}")
            
            inputs_description = '\n'.join(input_parameters)
            
            # Prepare the prompt
            prompt = f"""
            {RETURN_INSTRUCTION}
            
            Given the following user-defined model defined in Python code:
            
            ```python
            {model_code_formatted}
            ```
            
            and the following uncertain input distributions:
            
            {inputs_description}
            
            The results of the HSIC analysis are given in the table below:
            
            {hsic_md_table}
            
            Please provide an expert analysis of the HSIC results:
            
            1. **Methodology Overview**
               - Explain the mathematical basis of the HSIC method in sensitivity analysis
               - Discuss the advantages of HSIC over traditional correlation-based methods
               - Explain how HSIC captures both linear and non-linear dependencies
            
            2. **Results Interpretation**
               - Identify which variables have the strongest dependencies with the output based on HSIC indices
               - Discuss the statistical significance of these dependencies based on p-values
               - Explain what these patterns suggest about the model behavior and input-output relationships
            
            3. **Comparison with Other Methods**
               - Discuss how HSIC results might differ from variance-based or correlation-based methods
               - Explain when HSIC is particularly valuable (e.g., for highly non-linear models)
               - Identify potential limitations of the HSIC approach
            
            4. **Recommendations**
               - Suggest which variables should be prioritized for uncertainty reduction based on HSIC indices
               - Recommend additional analyses that might be valuable given these HSIC patterns
               - Provide guidance on how these results can inform decision-making or model refinement
            
            Format your response with clear section headings and bullet points. Focus on actionable insights and quantitative recommendations.
            """
            
            # Check if the results are already in session state
            response_key = 'hsic_response_markdown'
            if response_key not in st.session_state:
                # Call the AI API
                with st.spinner("Generating expert analysis..."):
                    response_markdown = call_groq_api(prompt, model_name=language_model)
                # Store the response in session state
                st.session_state[response_key] = response_markdown
            else:
                response_markdown = st.session_state[response_key]
            
            # Display the response
            st.markdown(response_markdown)
        
        # Return results dictionary
        return {
            "hsic_results": hsic_results,
            "hsic_df": hsic_df
        }
    
    except Exception as e:
        st.error(f"Error in HSIC analysis: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        raise

def compute_hsic_indices(model, problem, N=1000, seed=42):
    """
    Compute HSIC-based sensitivity indices.
    
    Parameters
    ----------
    model : callable
        The model function to analyze
    problem : ot.Distribution
        OpenTURNS distribution representing the input uncertainty
    N : int, optional
        Number of samples, by default 1000
    seed : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    dict
        Dictionary containing HSIC results
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

    # Set random seed
    np.random.seed(seed)
    ot.RandomGenerator.SetSeed(seed)

    # Generate samples
    sample_X = problem.getSample(N)
    X = np.array(sample_X)
    
    # Evaluate model
    Y_list = []
    for i in range(N):
        Y_list.append(model(sample_X[i]))
    Y = np.array(Y_list)
    
    # Make sure Y is 1D
    if len(Y.shape) > 1 and Y.shape[1] > 1:
        Y = Y[:, 0]  # Take first output for multivariate outputs

    # Compute kernel matrices
    K = np.zeros((N, N))
    L = np.zeros((N, N))
    H = np.eye(N) - np.ones((N, N)) / N

    # Compute output kernel matrix with median heuristic for bandwidth
    median_dist_y = np.median(np.abs(Y[:, np.newaxis] - Y[np.newaxis, :]))
    sigma_y = median_dist_y if median_dist_y > 0 else 1.0
    
    for i in range(N):
        for j in range(N):
            L[i, j] = np.exp(-0.5 * ((Y[i] - Y[j]) / sigma_y)**2)

    # Center kernel matrix
    L = H @ L @ H

    # Compute HSIC indices and p-values
    hsic_indices = np.zeros(dimension)
    p_values_asymptotic = np.zeros(dimension)
    p_values_permutation = np.zeros(dimension)
    
    for d in range(dimension):
        # Compute input kernel matrix for variable d with median heuristic
        X_d = X[:, d]
        median_dist_x = np.median(np.abs(X_d[:, np.newaxis] - X_d[np.newaxis, :]))
        sigma_x = median_dist_x if median_dist_x > 0 else 1.0
        
        for i in range(N):
            for j in range(N):
                K[i, j] = np.exp(-0.5 * ((X_d[i] - X_d[j]) / sigma_x)**2)
        
        # Center kernel matrix
        K = H @ K @ H
        
        # Compute HSIC
        hsic = np.trace(K @ L) / (N - 1)**2
        hsic_indices[d] = hsic
        
        # Compute asymptotic p-value
        # Based on the asymptotic distribution of HSIC under the null hypothesis
        mean_H0 = 1.0 / N * (1 + np.trace(K @ K) * np.trace(L @ L) / (N - 1)**2)
        var_H0 = 2.0 / (N * (N - 1)) * np.trace(K @ K @ L @ L)
        z_score = (hsic - mean_H0) / np.sqrt(var_H0)
        p_values_asymptotic[d] = 1 - stats.norm.cdf(z_score)
        
        # Compute permutation p-value
        # Perform a limited number of permutations for computational efficiency
        n_perm = 100
        hsic_perm = np.zeros(n_perm)
        
        for p in range(n_perm):
            perm = np.random.permutation(N)
            K_perm = K[perm, :][:, perm]
            hsic_perm[p] = np.trace(K_perm @ L) / (N - 1)**2
        
        p_values_permutation[d] = np.mean(hsic_perm >= hsic)

    # Normalize indices
    total_hsic = np.sum(hsic_indices)
    normalized_indices = hsic_indices / total_hsic if total_hsic > 0 else np.zeros_like(hsic_indices)

    # Create results dictionary
    results = {
        'hsic_indices': hsic_indices,
        'normalized_indices': normalized_indices,
        'p_values_asymptotic': p_values_asymptotic,
        'p_values_permutation': p_values_permutation,
        'input_names': input_names
    }

    return results

def create_hsic_dataframe(results):
    """
    Create a DataFrame with HSIC analysis results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing HSIC results
        
    Returns
    -------
    pd.DataFrame
        DataFrame with HSIC results
    """
    df = pd.DataFrame({
        'Variable': results['input_names'],
        'HSIC_Index': results['hsic_indices'],
        'Normalized_Index': results['normalized_indices'],
        'p_value_asymptotic': results['p_values_asymptotic'],
        'p_value_permutation': results['p_values_permutation']
    })
    
    # Sort by normalized index
    df = df.sort_values('Normalized_Index', ascending=False).reset_index(drop=True)
    
    return df

def create_hsic_plots(results):
    """
    Create interactive Plotly visualizations for HSIC results.
    
    Parameters
    ----------
    results : dict
        Dictionary containing HSIC results
        
    Returns
    -------
    go.Figure
        Plotly figure with HSIC visualizations
    """
    # Extract data
    normalized_indices = results['normalized_indices']
    hsic_indices = results['hsic_indices']
    p_values_asymptotic = results['p_values_asymptotic']
    p_values_permutation = results['p_values_permutation']
    input_names = results['input_names']
    
    # Sort by normalized index
    idx = np.argsort(normalized_indices)[::-1]
    normalized_indices = normalized_indices[idx]
    hsic_indices = hsic_indices[idx]
    p_values_asymptotic = p_values_asymptotic[idx]
    p_values_permutation = p_values_permutation[idx]
    input_names = [input_names[i] for i in idx]
    
    # Create a subplot with 3 plots
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=[
            "Normalized HSIC Indices", 
            "Raw HSIC Indices", 
            "p-values (Asymptotic vs Permutation)",
            "Significance Threshold"
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    # Add normalized HSIC indices bar chart
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=normalized_indices,
            name="Normalized HSIC",
            marker_color='rgb(55, 83, 109)',
            text=[f"{v:.4f}" for v in normalized_indices],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Add raw HSIC indices bar chart
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=hsic_indices,
            name="Raw HSIC",
            marker_color='rgb(26, 118, 255)',
            text=[f"{v:.2e}" for v in hsic_indices],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Add p-values bar chart
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=p_values_asymptotic,
            name="p-value (Asymptotic)",
            marker_color='rgb(219, 64, 82)',
            text=[f"{v:.2e}" for v in p_values_asymptotic],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=p_values_permutation,
            name="p-value (Permutation)",
            marker_color='rgb(154, 18, 179)',
            text=[f"{v:.2e}" for v in p_values_permutation],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Add significance threshold line
    fig.add_trace(
        go.Scatter(
            x=input_names,
            y=[0.05] * len(input_names),
            mode='lines',
            name='Significance (p=0.05)',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # Add significance visualization
    significant_asymp = p_values_asymptotic < 0.05
    significant_perm = p_values_permutation < 0.05
    
    categories = []
    for a, p in zip(significant_asymp, significant_perm):
        if a and p:
            categories.append("Both")
        elif a:
            categories.append("Asymptotic only")
        elif p:
            categories.append("Permutation only")
        else:
            categories.append("Not significant")
    
    # Create a categorical color map
    color_map = {
        "Both": "green",
        "Asymptotic only": "orange",
        "Permutation only": "blue",
        "Not significant": "red"
    }
    
    colors = [color_map[cat] for cat in categories]
    
    # Add significance scatter plot
    fig.add_trace(
        go.Scatter(
            x=input_names,
            y=normalized_indices,
            mode='markers',
            marker=dict(
                size=15,
                color=colors,
                symbol='circle',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name='Significance',
            text=[f"{input_names[i]}: {categories[i]}" for i in range(len(input_names))],
            hoverinfo='text'
        ),
        row=2, col=2
    )
    
    # Add a horizontal line at y=0 for the significance plot
    fig.add_trace(
        go.Scatter(
            x=[input_names[0], input_names[-1]],
            y=[0, 0],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="HSIC Sensitivity Analysis",
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Input Variables", row=1, col=1, tickangle=45)
    fig.update_yaxes(title_text="Normalized HSIC Index", row=1, col=1)
    
    fig.update_xaxes(title_text="Input Variables", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Raw HSIC Index", row=1, col=2)
    
    fig.update_xaxes(title_text="Input Variables", row=2, col=1, tickangle=45)
    fig.update_yaxes(title_text="p-value", row=2, col=1, type="log")
    
    fig.update_xaxes(title_text="Input Variables", row=2, col=2, tickangle=45)
    fig.update_yaxes(title_text="Normalized HSIC Index", row=2, col=2)
    
    return fig
