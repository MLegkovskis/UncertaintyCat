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

def model_understanding(model, problem, model_code_str, is_pce_used=False, original_model_code_str=None, metamodel_str=None, language_model='groq'):
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
    """
    # Create a two-column layout: main content (2/3) and chat interface (1/3)
    main_col, chat_col = st.columns([2, 1])
    
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

    max_attempts = 5

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

    # Initialize session state for chat
    if "model_understanding_chat_messages" not in st.session_state:
        st.session_state.model_understanding_chat_messages = []
    
    with main_col:
        # Generate AI explanation
        with st.spinner("Generating model explanation..."):
            try:
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
                response_key = 'model_understanding_response_markdown'

                if response_key not in st.session_state:
                    best_response_text = ""
                    attempts = 0

                    while attempts < max_attempts:
                        print(f"\n=== Attempt {attempts + 1} ===")
                        print("Using prompt:")
                        print(prompt)

                        response_text = call_groq_api(prompt, model_name=language_model)
                        print("Raw Response from API:")
                        print(response_text)

                        st.session_state[response_key] = response_text
                        break

                    if response_key not in st.session_state:
                        # If we have no valid response after max_attempts, return best effort
                        st.session_state[response_key] = best_response_text

                response_markdown = st.session_state[response_key]
                
                # Display the AI-generated explanation
                with st.expander("Model Explanation", expanded=True):
                    st.markdown(response_markdown)
                    
                    # Display input distributions using OpenTURNS' __repr_markdown__ method
                    st.write("### Input Distributions")
                    try:
                        if hasattr(problem, '__repr_markdown__'):
                            markdown_repr = problem.__repr_markdown__()
                            st.markdown(markdown_repr, unsafe_allow_html=True)
                        else:
                            # Fallback to dataframe display if __repr_markdown__ is not available
                            st.dataframe(
                                inputs_df,
                                column_config={
                                    "Variable": st.column_config.TextColumn("Variable Name"),
                                    "Distribution": st.column_config.TextColumn("Distribution Type"),
                                    "Parameters": st.column_config.TextColumn("Distribution Parameters")
                                },
                                use_container_width=True
                            )
                    except Exception as e:
                        st.warning(f"Could not display input distributions: {str(e)}")
                        # Fallback to dataframe display
                        st.dataframe(
                            inputs_df,
                            column_config={
                                "Variable": st.column_config.TextColumn("Variable Name"),
                                "Distribution": st.column_config.TextColumn("Distribution Type"),
                                "Parameters": st.column_config.TextColumn("Distribution Parameters")
                            },
                            use_container_width=True
                        )
                    
                    # If PCE is used, display metamodel comparison
                    if is_pce_used and original_model_code_str and metamodel_str:
                        st.write("### Original Model vs. Metamodel")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### Original Model")
                            st.code(original_model_code_str, language="python")
                        with col2:
                            st.write("#### PCE Metamodel")
                            st.code(metamodel_str, language="python")
            
            except Exception as e:
                st.error(f"Error generating model explanation: {str(e)}")
                st.write("Please try again or contact support if the issue persists.")
    
    # CHAT INTERFACE in the right column
    with chat_col:
        st.markdown("### Ask Questions About This Model")
        
        # Display a disclaimer about the prompt
        disclaimer_text = """
        **Note:** The AI assistant has been provided with the model code, input distributions, 
        and the explanation above. You can ask questions to clarify any aspects of the model.
        """
        st.info(disclaimer_text)
        
        # Define context generator function
        def generate_context(prompt):
            return f"""
            You are an expert assistant helping users understand a computational model. 
            
            Here is the model code:
            ```python
            {model_code_str}
            ```
            
            Here is information about the input distributions:
            {inputs_md_table}
            
            Here is the explanation that was previously generated:
            {st.session_state.get('model_understanding_response_markdown', 'No explanation available yet.')}
            
            Answer the user's question based on this information. Be concise but thorough.
            If you're not sure about something, acknowledge the limitations of your knowledge.
            Use LaTeX for equations when necessary, formatted as $...$ for inline or $$...$$ for display.
            """
        
        # Initialize session state for chat messages if not already done
        if "model_understanding_chat_messages" not in st.session_state:
            st.session_state.model_understanding_chat_messages = []
        
        # Create a container with fixed height for the chat messages
        chat_container_height = 500  # Height in pixels
        
        # Apply CSS to create a scrollable container
        st.markdown(f"""
        <style>
        .chat-container {{
            height: {chat_container_height}px;
            overflow-y: auto;
            border: 1px solid #e6e6e6;
            border-radius: 5px;
            padding: 10px;
            background-color: #f9f9f9;
            margin-bottom: 15px;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Create a container for the chat messages
        with st.container():
            # Use HTML to create a scrollable container
            chat_messages_html = "<div class='chat-container'>"
            
            # Display existing messages
            for message in st.session_state.model_understanding_chat_messages:
                role_style = "background-color: #e1f5fe; border-radius: 10px; padding: 8px; margin: 5px 0;" if message["role"] == "assistant" else "background-color: #f0f0f0; border-radius: 10px; padding: 8px; margin: 5px 0;"
                role_label = "Assistant:" if message["role"] == "assistant" else "You:"
                chat_messages_html += f"<div style='{role_style}'><strong>{role_label}</strong><br>{message['content']}</div>"
            
            chat_messages_html += "</div>"
            st.markdown(chat_messages_html, unsafe_allow_html=True)
        
        # Chat input below the scrollable container
        prompt = st.chat_input("Ask a question about the model...", key="model_understanding_side_chat_input")
        
        # Process user input
        if prompt:
            # Add user message to chat history
            st.session_state.model_understanding_chat_messages.append({"role": "user", "content": prompt})
            
            # Generate context for the assistant
            context = generate_context(prompt)
            
            # Include previous conversation history
            chat_history = ""
            if len(st.session_state.model_understanding_chat_messages) > 1:
                chat_history = "Previous conversation:\n"
                for i, msg in enumerate(st.session_state.model_understanding_chat_messages[:-1]):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n\n"
            
            # Create the final prompt
            chat_prompt = f"""
            {context}
            
            {chat_history}
            
            Current user question: {prompt}
            
            Please provide a helpful, accurate response to this question.
            """
            
            # Call API with chat history
            with st.spinner("Thinking..."):
                try:
                    response_text = call_groq_api(chat_prompt, model_name=language_model)
                except Exception as e:
                    st.error(f"Error calling API: {str(e)}")
                    response_text = "I'm sorry, I encountered an error while processing your question. Please try again."
            
            # Add assistant response to chat history
            st.session_state.model_understanding_chat_messages.append({"role": "assistant", "content": response_text})
            
            # Rerun to display the new message immediately
            st.rerun()