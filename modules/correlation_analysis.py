import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.core_utils import call_groq_api
from utils.constants import RETURN_INSTRUCTION

def compute_correlation_analysis(model, problem, size=1000):
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
    size : int, optional
        Number of samples for correlation analysis (default is 1000)
        
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
    sample_X = problem.getSample(size)
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
            current_output = ot.Sample(size, 1)
            for i in range(size):
                current_output[i, 0] = sample_Y[i, output_idx]
            current_output.setDescription([output_name])
        else:
            current_output = sample_Y
        
        # Use OpenTURNS CorrelationAnalysis for robust calculations
        corr_analysis = ot.CorrelationAnalysis(sample_X, current_output)
        
        # Calculate all correlation methods
        methods = {
            "Pearson": list(corr_analysis.computeLinearCorrelation()),
            "Spearman": list(corr_analysis.computeSpearmanCorrelation()),
            "PCC": list(corr_analysis.computePCC()),
            "PRCC": list(corr_analysis.computePRCC()),
            "SRC": list(corr_analysis.computeSRC()),
            "SRRC": list(corr_analysis.computeSRRC())
        }
        
        # Create DataFrame for correlation results
        corr_df = pd.DataFrame(methods, index=input_names)
        
        # Sort by absolute Pearson correlation for better visualization
        corr_df['abs_pearson'] = corr_df['Pearson'].abs()
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)
        corr_df = corr_df.drop('abs_pearson', axis=1)
        
        # Store results for this output
        all_correlation_results[output_name] = corr_df
    
    # Return all results
    return {
        'all_correlation_results': all_correlation_results,
        'input_names': input_names,
        'output_names': output_names,
        'output_dimension': output_dimension,
        'is_multivariate': is_multivariate,
        'sample_X': sample_X,
        'sample_Y': sample_Y,
        'size': size
    }

def create_correlation_visualizations(corr_df, output_name):
    """
    Create visualizations for correlation analysis results.
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        DataFrame containing correlation results
    output_name : str
        Name of the output variable
        
    Returns
    -------
    dict
        Dictionary containing visualization figures
    """
    # Find strongest positive and negative correlations
    strongest_pos = corr_df['Pearson'].max()
    strongest_pos_var = corr_df['Pearson'].idxmax()
    strongest_neg = corr_df['Pearson'].min()
    strongest_neg_var = corr_df['Pearson'].idxmin()
    
    # Find strongest absolute correlation (regardless of sign)
    strongest_abs = corr_df['Pearson'].abs().max()
    strongest_abs_var = corr_df['Pearson'].abs().idxmax()
    strongest_abs_sign = "positive" if corr_df.loc[strongest_abs_var, 'Pearson'] > 0 else "negative"
    
    # Create a comprehensive correlation visualization
    fig_combined = go.Figure()
    
    # Add all correlation methods as grouped bars
    methods = ['Pearson', 'Spearman', 'PCC', 'PRCC', 'SRC', 'SRRC']
    colors = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 'rgba(44, 160, 44, 0.8)', 
              'rgba(214, 39, 40, 0.8)', 'rgba(148, 103, 189, 0.8)', 'rgba(140, 86, 75, 0.8)']
    
    for i, method in enumerate(methods):
        fig_combined.add_trace(go.Bar(
            x=corr_df.index,
            y=corr_df[method],
            name=f"{method}",
            marker_color=colors[i],
            hovertemplate='%{x}: %{y:.4f}<extra></extra>'
        ))
    
    # Add zero line
    fig_combined.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(corr_df.index) - 0.5,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Update layout
    fig_combined.update_layout(
        title=f'Correlation Analysis',
        xaxis_title='Input Variables',
        yaxis_title='Correlation Coefficient',
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
        margin=dict(t=120, r=20, b=80, l=80)
    )
    
    # Create heatmap visualization
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=methods,
        y=corr_df.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_df.values, 4),
        texttemplate="%{text:.4f}",
        hovertemplate='%{y} - %{x}: %{z:.4f}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title='Correlation Coefficients Heatmap',
        xaxis_title='Correlation Method',
        yaxis_title='Input Variables',
        template='plotly_white',
        height=500,
        margin=dict(t=80, r=20, b=50, l=80)
    )
    
    # Create a comparison plot for linear vs. rank methods
    fig_comparison = go.Figure()
    
    # Add scatter points for Pearson vs Spearman
    fig_comparison.add_trace(go.Scatter(
        x=corr_df['Pearson'],
        y=corr_df['Spearman'],
        mode='markers+text',
        marker=dict(
            size=12,
            color='rgba(31, 119, 180, 0.8)',
            line=dict(width=1, color='black')
        ),
        text=corr_df.index,
        textposition="top center",
        name='Pearson vs Spearman'
    ))
    
    # Add diagonal line (y=x)
    max_abs = max(
        abs(corr_df['Pearson'].max()), 
        abs(corr_df['Pearson'].min()),
        abs(corr_df['Spearman'].max()), 
        abs(corr_df['Spearman'].min())
    )
    max_abs = max(max_abs, 0.1) * 1.1  # Add some margin
    
    fig_comparison.add_shape(
        type="line",
        x0=-max_abs,
        y0=-max_abs,
        x1=max_abs,
        y1=max_abs,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Add zero lines
    fig_comparison.add_shape(
        type="line",
        x0=-max_abs,
        y0=0,
        x1=max_abs,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    fig_comparison.add_shape(
        type="line",
        x0=0,
        y0=-max_abs,
        x1=0,
        y1=max_abs,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Update layout
    fig_comparison.update_layout(
        title='Linear vs. Rank Correlation (Pearson vs. Spearman)',
        xaxis_title='Pearson Correlation',
        yaxis_title='Spearman Correlation',
        template='plotly_white',
        height=500,
        xaxis=dict(range=[-max_abs, max_abs]),
        yaxis=dict(range=[-max_abs, max_abs]),
        margin=dict(t=80, r=20, b=50, l=80)
    )
    
    # Add annotations to explain the quadrants
    quadrant_explanations = [
        {"x": max_abs*0.7, "y": max_abs*0.7, "text": "Strong positive\nlinear & monotonic", "align": "right"},
        {"x": -max_abs*0.7, "y": max_abs*0.7, "text": "Strong positive monotonic\nweak/negative linear", "align": "left"},
        {"x": -max_abs*0.7, "y": -max_abs*0.7, "text": "Strong negative\nlinear & monotonic", "align": "left"},
        {"x": max_abs*0.7, "y": -max_abs*0.7, "text": "Strong negative monotonic\nweak/positive linear", "align": "right"}
    ]
    
    for exp in quadrant_explanations:
        fig_comparison.add_annotation(
            x=exp["x"],
            y=exp["y"],
            text=exp["text"],
            showarrow=False,
            font=dict(size=10, color="gray"),
            align=exp["align"]
        )
    
    # Return all figures
    return {
        'fig_combined': fig_combined,
        'fig_heatmap': fig_heatmap,
        'fig_comparison': fig_comparison,
        'strongest_pos': strongest_pos,
        'strongest_pos_var': strongest_pos_var,
        'strongest_neg': strongest_neg,
        'strongest_neg_var': strongest_neg_var,
        'strongest_abs': strongest_abs,
        'strongest_abs_var': strongest_abs_var,
        'strongest_abs_sign': strongest_abs_sign
    }

def display_correlation_results(analysis_results, language_model=None):
    """
    Display correlation analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing correlation analysis results
    language_model : str, optional
        Language model to use for AI insights, by default None
    """
    try:
        # Extract results
        all_correlation_results = analysis_results['all_correlation_results']
        input_names = analysis_results['input_names']
        output_names = analysis_results['output_names']
        output_dimension = analysis_results['output_dimension']
        is_multivariate = analysis_results['is_multivariate']
        
        # Display basic information
        st.markdown(f"""
        ### Correlation Analysis Results
        """)
        
        # For each output, create a tab
        if is_multivariate and output_dimension > 1:
            tabs = st.tabs(output_names)
            for i, tab in enumerate(tabs):
                with tab:
                    output_name = output_names[i]
                    corr_df = all_correlation_results[output_name]
                    
                    # Create visualizations for this output
                    viz_results = create_correlation_visualizations(corr_df, output_name)
                    
                    # Display visualizations and results
                    display_single_output_results(corr_df, output_name, viz_results)
        else:
            # Single output case - no tabs needed
            main_output = output_names[0]
            main_corr_df = all_correlation_results[main_output]
            
            # Create visualizations
            viz_results = create_correlation_visualizations(main_corr_df, main_output)
            
            # Display visualizations and results
            display_single_output_results(main_corr_df, main_output, viz_results)
        
        # Display AI insights if available
        if 'llm_insights' in analysis_results:
            st.markdown("### AI Insights")
            st.markdown(analysis_results['llm_insights'])
        # If not available but language model is provided, this means we're in display mode
        # and insights were not pre-generated (should not happen with our new implementation)
        elif language_model:
            st.warning("AI insights were not pre-generated. This is unusual and may indicate an issue with the analysis.")
            
    except Exception as e:
        st.error(f"Error displaying correlation results: {str(e)}")

def display_single_output_results(corr_df, output_name, viz_results):
    """
    Display correlation results for a single output.
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        DataFrame containing correlation results
    output_name : str
        Name of the output variable
    viz_results : dict
        Dictionary containing visualization figures
    """
    # Extract visualization results
    strongest_pos = viz_results['strongest_pos']
    strongest_pos_var = viz_results['strongest_pos_var']
    strongest_neg = viz_results['strongest_neg']
    strongest_neg_var = viz_results['strongest_neg_var']
    strongest_abs = viz_results['strongest_abs']
    strongest_abs_var = viz_results['strongest_abs_var']
    strongest_abs_sign = viz_results['strongest_abs_sign']
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Most Influential Variable", 
            strongest_abs_var,
            f"Pearson: {strongest_abs:.4f} ({strongest_abs_sign})"
        )
    with col2:
        st.metric(
            "Strongest Positive", 
            strongest_pos_var,
            f"Pearson: {strongest_pos:.4f}"
        )
    with col3:
        st.metric(
            "Strongest Negative", 
            strongest_neg_var,
            f"Pearson: {strongest_neg:.4f}"
        )
    
    # Display the combined figure
    st.markdown("#### Correlation Coefficients")
    st.markdown("""
    This grouped bar chart shows all correlation coefficients for each input variable.
    Positive values indicate that the output increases as the input increases,
    while negative values indicate that the output decreases as the input increases.
    """)
    st.plotly_chart(viz_results['fig_combined'], use_container_width=True)
    
    # Display the heatmap and comparison plot in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Correlation Heatmap")
        st.markdown("""
        This heatmap shows all correlation coefficients with color coding.
        Red indicates positive correlation, blue indicates negative correlation.
        """)
        st.plotly_chart(viz_results['fig_heatmap'], use_container_width=True)
    
    with col2:
        st.markdown("#### Linear vs. Rank Correlation")
        st.markdown("""
        This scatter plot compares Pearson (linear) and Spearman (rank) correlations.
        Points far from the diagonal indicate nonlinear relationships.
        """)
        st.plotly_chart(viz_results['fig_comparison'], use_container_width=True)
    
    # Display correlation table
    st.subheader("Correlation Coefficients Table")
    st.dataframe(corr_df.style.format("{:.4f}"), use_container_width=True)
    
    # Display interpretation
    st.subheader("Interpretation")
    
    # Check for consistency across methods
    consistency_df = corr_df.copy()
    # Get the sign of each correlation
    sign_df = consistency_df.applymap(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    # Check if all methods agree on the sign for each variable
    consistent_signs = sign_df.apply(lambda row: row.nunique() == 1, axis=1)
    
    # Display key insights
    st.markdown(f"""
    **Key Insights:**
    
    - **Strongest overall influence**: {strongest_abs_var} ({strongest_abs:.4f} absolute Pearson, {strongest_abs_sign} correlation)
    - **Strongest positive correlation**: {strongest_pos_var} ({strongest_pos:.4f} Pearson)
      - As {strongest_pos_var} increases, {output_name} tends to increase
    - **Strongest negative correlation**: {strongest_neg_var} ({strongest_neg:.4f} Pearson)
      - As {strongest_neg_var} increases, {output_name} tends to decrease
    - **Consistency across methods**: {consistent_signs.mean()*100:.1f}% of variables show consistent direction
    """)
    
    # Add method comparison
    st.markdown("""
    **Method Comparison:**
    
    - **Linear vs. Rank-based**: Differences between Pearson and Spearman indicate nonlinear relationships
    - **Direct vs. Partial**: Differences between PCC and Pearson indicate indirect effects through other variables
    - **Regression vs. Correlation**: SRC and SRRC provide standardized sensitivity measures
    
    **Understanding Negative Correlations:**
    
    Negative correlations are equally important as positive ones and indicate an inverse relationship between the input and output. 
    They show that as the input variable increases, the output tends to decrease. This can be crucial for understanding system 
    behavior and for controlling the output by adjusting inputs in the appropriate direction.
    """)

def correlation_analysis(model, problem, model_code_str=None, size=1000, language_model="groq", display_results=True):
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
    size : int, optional
        Number of samples for correlation analysis (default is 1000)
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
        with st.spinner("Running Correlation Analysis..."):
            # Compute all correlation analysis results
            results = compute_correlation_analysis(model, problem, size=size)
            
            # Create visualizations for each output
            all_viz_results = {}
            for output_name, corr_df in results['all_correlation_results'].items():
                all_viz_results[output_name] = create_correlation_visualizations(corr_df, output_name)
            
            # Add visualizations to results
            main_output = results['output_names'][0]  # Use first output for main visualizations
            main_corr_df = results['all_correlation_results'][main_output]
            main_viz = all_viz_results[main_output]
            
            # Add main visualizations to results
            results['fig_bar'] = main_viz['fig_combined']
            results['fig_heatmap'] = main_viz['fig_heatmap']
            results['fig_comparison'] = main_viz['fig_comparison']
            
            # Create a clean correlation dataframe for display
            results['correlation_df'] = main_corr_df.copy()
            
            # Generate AI insights if a language model is provided
            if language_model and model_code_str:
                # Create a markdown table of the correlation results
                correlation_md_table = main_corr_df.to_markdown(index=True, floatfmt=".4f")
                
                # Find strongest correlations
                strongest_pearson_var = main_corr_df['Pearson'].abs().idxmax()
                strongest_pearson = main_corr_df.loc[strongest_pearson_var, 'Pearson']
                strongest_spearman_var = main_corr_df['Spearman'].abs().idxmax()
                strongest_spearman = main_corr_df.loc[strongest_spearman_var, 'Spearman']
                
                # Create distribution information for the prompt
                dist_info = []
                for i, name in enumerate(results['input_names']):
                    marginal = problem.getMarginal(i)
                    dist_info.append({
                        'Variable': name,
                        'Distribution': marginal.__class__.__name__,
                        'Parameters': str(list(marginal.getParameter())),
                        'Mean': float(marginal.getMean()[0]),
                        'Std': float(marginal.getStandardDeviation()[0])
                    })
                dist_df = pd.DataFrame(dist_info)
                
                # Format the model code for inclusion in the prompt
                model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])
                
                # Generate prompt
                prompt = f"""
{RETURN_INSTRUCTION}

## Correlation Analysis Results

I need your expert analysis of these correlation results for the following model:

```python
{model_code_formatted}
```

### Input Distributions:
{dist_df.to_markdown(index=False)}

### Correlation Analysis Results:
{correlation_md_table}

### Key observations:
- Strongest Pearson correlation: {strongest_pearson_var} ({strongest_pearson:.4f})
- Strongest Spearman correlation: {strongest_spearman_var} ({strongest_spearman:.4f})
- Input variables: {", ".join(results['input_names'])}
- Output: {main_output}

Please provide a comprehensive analysis of these correlation results, focusing on:

1. **Executive Summary**
   - Key findings and their practical implications
   - Most influential parameters identified by the correlation analysis
   - Assessment of linear vs. nonlinear relationships

2. **Methodology Overview**
   - Explanation of each correlation method (Pearson, Spearman, PCC, PRCC, SRC, SRRC)
   - When each method is most appropriate to use
   - Differences between these methods and what insights each provides

3. **Results Interpretation**
   - Variables with the strongest positive and negative correlations
   - Consistency or inconsistency across different correlation methods
   - What these patterns suggest about model behavior
   - Implications of negative correlations

4. **Nonlinearity Assessment**
   - Comparison of Pearson vs. Spearman and PCC vs. PRCC to assess nonlinearity
   - Variables showing significant differences between linear and rank-based methods
   - What these differences indicate about the underlying model structure

5. **Recommendations**
   - Variables that should be prioritized for uncertainty reduction
   - Additional analyses that might be valuable given these correlation patterns
   - Guidance for decision-making or model refinement

Format your response with clear section headings and bullet points where appropriate.
Focus on actionable insights that would be valuable for decision-makers.
"""
                
                # Call the AI API with retry logic and always use 'llm_insights' as the key for consistency
                max_retries = 3
                retry_count = 0
                ai_insights = None
                while ai_insights is None and retry_count < max_retries:
                    try:
                        ai_insights = call_groq_api(prompt, model_name=language_model)
                        results['llm_insights'] = ai_insights
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            ai_insights = f"""
                            ## Error Generating Insights
                            There was an error connecting to the language model API: {str(e)}
                            Please try again later or check your API key/model configuration.
                            """
                            results['llm_insights'] = ai_insights
            
            # Display results if requested
            if display_results:
                display_correlation_results(results, language_model)
            
            # Clean up results to remove unnecessary data
            # Only remove large raw data, not summary keys needed for display
            keys_to_remove = ['sample_X', 'sample_Y']
            for key in keys_to_remove:
                if key in results:
                    del results[key]
            
            return results
    except Exception as e:
        if display_results:
            st.error(f"Error in correlation analysis: {str(e)}")
        raise
