import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.core_utils import call_groq_api
from utils.constants import RETURN_INSTRUCTION
from utils.model_utils import get_ot_model

def correlation_analysis(model, problem, model_code_str, language_model="groq"):
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
    model_code_str : str
        String representation of the model code for documentation
    language_model : str, optional
        Language model to use for analysis, by default "groq"
        
    Returns
    -------
    dict
        Dictionary containing correlation analysis results
    """
    try:
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
        
        # Display the header and introduction
        st.markdown("## Correlation Analysis")
        st.markdown("""
        Correlation analysis quantifies the statistical relationship between input variables and model outputs.
        This analysis helps identify which inputs have the strongest linear and monotonic relationships with the output.
        
        Multiple correlation metrics are calculated to provide a comprehensive understanding of variable relationships:
        """)
        
        # Create expandable section for methodology
        with st.expander("Correlation Methods Explained", expanded=True):
            st.markdown("""
            ### Correlation Methods
            
            This analysis uses six different correlation measures:
            
            1. **Pearson Correlation Coefficient (PCC)**
               - Measures linear correlation between variables
               - Range: -1 (perfect negative) to +1 (perfect positive)
               - Assumption: Linear relationship between variables
            
            2. **Partial Correlation Coefficient (PCC)**
               - Measures linear correlation between two variables while controlling for other variables
               - Helps identify direct relationships by removing indirect effects
            
            3. **Partial Rank Correlation Coefficient (PRCC)**
               - Rank-based version of PCC
               - More robust to nonlinear relationships
               - Maintains the control for other variables
            
            4. **Spearman Rank Correlation Coefficient**
               - Measures monotonic relationships (not just linear)
               - Based on ranks rather than raw values
               - More robust to outliers and nonlinear relationships
            
            5. **Standardized Regression Coefficient (SRC)**
               - Regression coefficients standardized to have unit variance
               - Represents the change in output (in standard deviations) per unit change in input
               - Useful for comparing the relative importance of variables
            
            6. **Standardized Rank Regression Coefficient (SRRC)**
               - Rank-based version of SRC
               - More robust to nonlinear relationships
               - Maintains the standardization for comparison
            """)
        
        # Create expandable section for interpreting negative correlations
        with st.expander("Understanding Correlation Signs", expanded=True):
            st.markdown("""
            ### Interpreting Positive and Negative Correlations
            
            The sign of a correlation coefficient (positive or negative) indicates the direction of the relationship between an input variable and the output:
            
            #### Positive Correlation (> 0)
            - As the input variable increases, the output tends to increase
            - Indicates a direct relationship
            - Example: Increasing temperature might increase reaction rate
            
            #### Negative Correlation (< 0)
            - As the input variable increases, the output tends to decrease
            - Indicates an inverse relationship
            - Example: Increasing pressure might decrease volume (Boyle's Law)
            
            #### Correlation Strength
            - Values close to +1 or -1 indicate strong relationships
            - Values close to 0 indicate weak or no relationship
            - The absolute value of the correlation is what matters for sensitivity ranking
            
            #### Engineering Implications
            - Negative correlations are just as important as positive ones for sensitivity analysis
            - For risk reduction, focus on variables with high absolute correlation values
            - Understanding the sign helps in determining control strategies (increase or decrease the variable)
            """)
        
        # Generate samples
        n_samples = 1000  # Number of samples for correlation analysis
        st.info(f"Generating {n_samples} samples for correlation analysis...")
        
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
        
        # Create tabs for multiple outputs if needed
        if is_multivariate and output_dimension > 1:
            tabs = st.tabs(output_names)
        
        for output_idx in range(output_dimension):
            # Get the current output name
            output_name = output_names[output_idx]
            
            # Extract current output data
            if is_multivariate:
                current_output = ot.Sample(n_samples, 1)
                for i in range(n_samples):
                    current_output[i, 0] = sample_Y[i, output_idx]
                current_output.setDescription([output_name])
            else:
                current_output = sample_Y
            
            # Perform correlation analysis
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
            
            # Store results
            all_correlation_results[output_name] = corr_df
            
            # Display results in the appropriate context
            if is_multivariate and output_dimension > 1:
                # Use the appropriate tab for this output
                tab_content = tabs[output_idx]
                
                # Create visualization tabs within this output tab
                with tab_content:
                    display_correlation_results(corr_df, input_names, output_name, sample_X, sample_Y, output_idx, n_samples, is_multivariate)
            else:
                # For single output, display directly
                display_correlation_results(corr_df, input_names, output_name, sample_X, sample_Y, output_idx, n_samples, is_multivariate)
        
        # Generate AI insights
        with st.expander("Expert Analysis", expanded=True):
            st.markdown("### AI-Generated Expert Analysis")
            
            # Prepare the data for the API call
            # Use the first output if multivariate
            primary_output = output_names[0]
            correlation_df = all_correlation_results[primary_output]
            correlation_md_table = correlation_df.to_markdown(index=True, floatfmt=".4f")
            
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
            
            # Add information about output dimensionality
            output_info = ""
            if is_multivariate and output_dimension > 1:
                output_info = f"""
                The model has multiple outputs: {', '.join(output_names)}
                The correlation analysis shown below is for the first output: {primary_output}
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
            
            {output_info}
            
            The results of the correlation analysis are given in the table below:
            
            {correlation_md_table}
            
            Please provide an expert analysis of the correlation results:
            
            1. **Methodology Overview**
               - Explain the mathematical basis of each correlation method (PCC, PRCC, Spearman, SRC, SRRC)
               - Discuss when each method is most appropriate to use
               - Explain the differences between these methods and what insights each provides
            
            2. **Results Interpretation**
               - Identify which variables have the strongest positive and negative correlations with the output
               - Discuss consistency or inconsistency in the sensitivity predictions across the different correlation methods
               - Explain what these patterns suggest about the model behavior and input-output relationships
               - Specifically explain what negative correlations indicate about the model behavior
            
            3. **Nonlinearity Assessment**
               - Compare Pearson vs. Spearman and PCC vs. PRCC to assess nonlinearity in relationships
               - Identify variables that show significant differences between linear and rank-based methods
               - Explain what these differences indicate about the underlying model structure
            
            4. **Recommendations**
               - Suggest which variables should be prioritized for uncertainty reduction based on correlation strength
               - Recommend additional analyses that might be valuable given these correlation patterns
               - Provide guidance on how these results can inform decision-making or model refinement
            
            Format your response with clear section headings and bullet points. Focus on actionable insights and quantitative recommendations.
            """
            
            # Check if the results are already in session state
            response_key = f"correlation_response_markdown_{primary_output}"
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
            "correlations": all_correlation_results,
            "input_names": input_names,
            "output_names": output_names,
            "is_multivariate": is_multivariate
        }
    
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        raise

def display_correlation_results(corr_df, input_names, output_name, sample_X, sample_Y, output_idx, n_samples, is_multivariate):
    """
    Display correlation results for a single output.
    
    This helper function handles the visualization and interpretation of correlation results.
    It creates a stacked bar chart and heatmap visualization and provides interpretation.
    
    Parameters
    ----------
    corr_df : pd.DataFrame
        DataFrame containing correlation results
    input_names : list
        List of input variable names
    output_name : str
        Name of the current output
    sample_X : ot.Sample
        Sample of input values
    sample_Y : ot.Sample
        Sample of output values
    output_idx : int
        Index of the current output
    n_samples : int
        Number of samples
    is_multivariate : bool
        Whether the output is multivariate
    """
    # Create a stacked 2x1 subplot with bar chart and heatmap (one above the other)
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=["Correlation Coefficients", "Correlation Heatmap"],
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Add bar chart traces
    for i, method in enumerate(corr_df.columns):
        fig.add_trace(
            go.Bar(
                x=corr_df.index,
                y=corr_df[method],
                name=method
            ),
            row=1, col=1
        )
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_df.T.values,
            x=corr_df.index,
            y=corr_df.columns,
            colorscale='RdBu_r',
            text=np.round(corr_df.T.values, 2),
            texttemplate="%{text:.2f}",
            showscale=True
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"Correlation Analysis for {output_name}",
        height=800,  # Increased height for the stacked layout
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=0.52, xanchor="right", x=1),
        barmode='group'
    )
    
    # Add zero line to bar chart
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(corr_df.index) - 0.5,
        y1=0,
        line=dict(color="black", width=1, dash="dash"),
        row=1, col=1
    )
    
    # Update x and y axis labels
    fig.update_xaxes(title_text="Input Variables", row=1, col=1)
    fig.update_yaxes(title_text="Correlation Coefficient", row=1, col=1)
    fig.update_xaxes(title_text="Input Variables", row=2, col=1)
    fig.update_yaxes(title_text="Correlation Method", row=2, col=1)
    
    # Display the combined figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Display correlation table
    st.markdown("### Correlation Coefficients Table")
    st.dataframe(corr_df.style.format("{:.4f}"), use_container_width=True)
    
    # Display interpretation
    st.markdown("### Interpretation")
    
    # Find strongest positive and negative correlations
    strongest_pos = corr_df['Pearson'].max()
    strongest_pos_var = corr_df['Pearson'].idxmax()
    strongest_neg = corr_df['Pearson'].min()
    strongest_neg_var = corr_df['Pearson'].idxmin()
    
    # Find strongest absolute correlation (regardless of sign)
    strongest_abs = corr_df['Pearson'].abs().max()
    strongest_abs_var = corr_df['Pearson'].abs().idxmax()
    strongest_abs_sign = "positive" if corr_df.loc[strongest_abs_var, 'Pearson'] > 0 else "negative"
    
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
