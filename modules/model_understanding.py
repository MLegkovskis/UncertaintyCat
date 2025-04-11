import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import uuid
import json
import os
from groq import Groq
from utils.core_utils import call_groq_api, create_chat_interface
from utils.constants import RETURN_INSTRUCTION

def compute_model_understanding(model, problem, model_code_str, is_pce_used=False, original_model_code_str=None, metamodel_str=None, language_model='groq'):
    """
    Compute model understanding analysis without UI components.
    
    This function generates a comprehensive description of the model and its uncertain inputs
    without displaying any UI elements.
    
    Parameters
    ----------
    model : callable
        The model function
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str
        String representation of the model code
    is_pce_used : bool, optional
        Whether a polynomial chaos expansion metamodel is used, by default False
    original_model_code_str : str, optional
        Original model code if using a metamodel, by default None
    metamodel_str : str, optional
        String representation of the metamodel, by default None
    language_model : str, optional
        Language model to use for AI insights, by default 'groq'
        
    Returns
    -------
    dict
        Dictionary containing model understanding analysis results
    """
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Get input dimension and names
    dimension = problem.getDimension()
    input_names = problem.getDescription()
    
    # Extract distribution information
    distributions = []
    inputs_df = pd.DataFrame(columns=["Variable", "Distribution", "Parameters"])
    
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        dist_type = marginal.getName()
        dist_params = str(marginal.getParameter())
        
        distributions.append({
            'type': dist_type,
            'params': dist_params
        })
        
        inputs_df = pd.concat([inputs_df, pd.DataFrame({
            "Variable": [input_names[i]],
            "Distribution": [dist_type],
            "Parameters": [dist_params]
        })], ignore_index=True)
    
    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])
    
    # Convert inputs_df to markdown table for the prompt
    inputs_md_table = inputs_df.to_markdown(index=False)

    # Generate prompt for the language model
    def generate_prompt(additional_instructions=""):
        # Enhanced prompt for better model explanation
        return f"""
{RETURN_INSTRUCTION}

Translate the following Python model code into a comprehensive, engaging Markdown explanation for an enterprise audience, focusing on:

1. **Model Overview**:
   - The model's purpose and real-world significance in clear, non-technical language
   - The engineering or scientific domain where this model is applied
   - Why understanding this model is valuable for decision-making

2. **Mathematical Formulation**:
   - The core mathematical relationships as clean LaTeX equations (using $...$ or $$...$$)
   - Physical interpretation of each term in the equations
   - Units for all variables (make reasonable assumptions if not explicit)
   - For complex expressions, include only key terms with \\dots notation

3. **Uncertainty Analysis Context**:
   - How uncertainty in inputs propagates to the output
   - The practical implications of this uncertainty for reliability and risk assessment
   - Which inputs likely have the most significant impact on output uncertainty

4. **Input Parameters**:
   - Clear definitions of each input variable with physical meaning
   - The rationale behind the chosen probability distributions
   - Typical ranges and units for each parameter

Format your response with clear section headings, bullet points where appropriate, and ensure all LaTeX is correctly formatted for Markdown rendering.

Requirements:
- Use \\geq and \\leq instead of >= and <= in equations
- All LaTeX must be syntactically correct and balanced
- For piecewise functions, use proper LaTeX notation with cases
- No code blocks or reprinting of the original code
- Focus on explaining the model's physical meaning, not its implementation
- Do not create tables for input parameters as they are displayed separately

IMPORTANT FOR TABLES:
- When creating tables that include mathematical symbols, ensure each cell's content is properly spaced
- For distribution parameters in tables, use simple notation like "μ = 0.1, σ = 0.01" instead of LaTeX
- Keep table formatting simple with proper pipe separators and header rows
- Avoid complex LaTeX in table cells that might break the markdown rendering
- For units, use simple text like "m/s" or "kg" rather than LaTeX in table cells

Original Python code (for reference, do not reprint):
```python
{model_code_formatted}
```

Input Parameters:
{inputs_md_table}

{additional_instructions}
""".strip()

    # Generate the prompt for the language model
    prompt = generate_prompt()
    
    # Add metamodel comparison instructions if PCE is used
    if is_pce_used and original_model_code_str and metamodel_str:
        additional_instructions = """
        Additionally, compare the original model with the polynomial chaos expansion metamodel.
        Explain how the PCE approximates the original model and what advantages this brings.
        """
        prompt = generate_prompt(additional_instructions)
    
    # Get response from language model
    try:
        response_text = call_groq_api(prompt, model_name=language_model)
    except Exception as e:
        response_text = f"Error generating model explanation: {str(e)}"
    
    # Return results dictionary
    return {
        'model_code_str': model_code_str,
        'inputs_df': inputs_df,
        'inputs_md_table': inputs_md_table,
        'explanation': response_text,
        'is_pce_used': is_pce_used,
        'original_model_code_str': original_model_code_str,
        'metamodel_str': metamodel_str
    }

def display_model_understanding(analysis_results, language_model='groq'):
    """
    Display model understanding analysis results in the Streamlit interface.
    
    Parameters
    ----------
    analysis_results : dict
        Dictionary containing model understanding analysis results
    language_model : str, optional
        Language model to use for AI insights, by default 'groq'
    """
    # Extract results from the analysis_results dictionary
    model_code_str = analysis_results['model_code_str']
    inputs_df = analysis_results['inputs_df']
    inputs_md_table = analysis_results['inputs_md_table']
    explanation = analysis_results['explanation']
    is_pce_used = analysis_results['is_pce_used']
    original_model_code_str = analysis_results['original_model_code_str']
    metamodel_str = analysis_results['metamodel_str']
    
    # Store the explanation in session state for reuse
    if 'model_understanding_response_markdown' not in st.session_state:
        st.session_state['model_understanding_response_markdown'] = explanation
    
    # Display the AI-generated explanation
    with st.expander("Model Explanation", expanded=True):
        st.markdown(explanation)
        
        # Display input distributions
        st.write("### Input Distributions")
        try:
            # Display the dataframe
            st.dataframe(
                inputs_df,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable Name"),
                    "Distribution": st.column_config.TextColumn("Distribution Type"),
                    "Parameters": st.column_config.TextColumn("Parameters"),
                    "Bounds": st.column_config.TextColumn("Bounds"),
                    "Mean": st.column_config.NumberColumn("Mean", format="%.4f"),
                    "Std": st.column_config.NumberColumn("Std. Dev.", format="%.4f")
                },
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error displaying input distributions: {e}")
        
        # Display model code
        st.write("### Model Code")
        st.code(model_code_str, language="python")
        
        # If a metamodel is used, display it
        if is_pce_used and metamodel_str:
            st.write("### Polynomial Chaos Expansion Metamodel")
            st.info("This analysis uses a polynomial chaos expansion (PCE) metamodel instead of the original model.")
            
            # Display the original model
            st.write("#### Original Model")
            st.code(original_model_code_str, language="python")
            
            # Display the metamodel
            st.write("#### PCE Metamodel")
            st.code(metamodel_str, language="python")

def model_understanding(model, problem, model_code_str, is_pce_used=False, original_model_code_str=None, metamodel_str=None, language_model='groq', display_results=True):
    """
    Generate a comprehensive description of the model and its uncertain inputs.
    
    This module introduces the model to users in a clear, engaging way with visual aids
    and explanations that contextualize the model's purpose, inputs, and significance.
    
    Parameters
    ----------
    model : callable
        The model function
    problem : ot.Distribution
        OpenTURNS distribution object defining the problem
    model_code_str : str
        String representation of the model code
    is_pce_used : bool, optional
        Whether a polynomial chaos expansion metamodel is used, by default False
    original_model_code_str : str, optional
        Original model code if using a metamodel, by default None
    metamodel_str : str, optional
        String representation of the metamodel, by default None
    language_model : str, optional
        Language model to use for AI insights, by default 'groq'
    display_results : bool, optional
        Whether to display results in the UI, by default True
    
    Returns
    -------
    dict
        Dictionary containing model understanding analysis results
    """
    try:
        # Compute model understanding analysis
        analysis_results = compute_model_understanding(
            model, 
            problem, 
            model_code_str, 
            is_pce_used, 
            original_model_code_str, 
            metamodel_str, 
            language_model
        )
        
        # Display results if requested
        if display_results:
            display_model_understanding(analysis_results, language_model)
        
        # Return analysis results
        return analysis_results
    
    except Exception as e:
        if display_results:
            st.error(f"Error in model understanding: {str(e)}")
        raise