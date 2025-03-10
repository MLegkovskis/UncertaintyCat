# utils/core_utils.py

"""
Core utilities for the UncertaintyCat application.
This module contains API utilities, session state management, and model options.
"""

import os
import streamlit as st
import re
from groq import Groq

# ================ API Utilities ================

def call_groq_api(prompt, model_name="gemma2-9b-it"):
    """
    Call the Groq API with a provided prompt.
    
    Parameters
    ----------
    prompt : str
        The prompt to send to the API
    model_name : str, optional
        The model to use, by default "gemma2-9b-it"
        
    Returns
    -------
    str
        The response text from the API
    """
    client = Groq(
        api_key=os.getenv('GROQ_API_KEY'),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name
    )
    response_text = chat_completion.choices[0].message.content
    response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)

    return response_text

# ================ Session State Utilities ================

def initialize_session_state():
    """Initialize minimal required session state variables."""
    defaults = {
        "code": "",
        "morris_results": None,
        "fixed_values": {},
        "reduced_model": None,
        "model_file": '(Select or define your own model)',
        "simulation_results": None,
        "run_simulation": False,
        "markdown_output": None,
        "morris_analysis_done": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def reset_analysis():
    """Reset analysis results."""
    st.session_state.morris_results = None
    st.session_state.fixed_values = {}
    st.session_state.reduced_model = None

def get_session_state(key, default=None):
    """Returns the value for the given key from session state, or the default value if the key doesn't exist."""
    return st.session_state.get(key, default)

# ================ Model Options ================

def get_model_options():
    """
    Dynamically get a list of available model files from the examples directory.
    
    Returns
    -------
    list
        List of Python files in the examples directory
    """
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    model_files = []
    
    if os.path.exists(examples_dir) and os.path.isdir(examples_dir):
        for file in os.listdir(examples_dir):
            if file.endswith('.py'):
                model_files.append(file)
    
    # Sort alphabetically for consistent display
    return sorted(model_files)

# ================ Instructions ================

def show_instructions():
    """
    Displays instructions in an expander.
    
    This function creates an expandable section in the Streamlit interface
    that contains detailed instructions on how to use the UncertaintyCat application.
    """
    with st.expander("Instructions"):
        st.markdown("""
        # Uncertainty Quantification and Sensitivity Analysis Application

        ## Introduction

        Welcome to the UncertaintyCat Application! This app allows you to perform comprehensive uncertainty quantification and sensitivity analysis on mathematical models from various fields. It provides AI-driven insights and supports advanced features like dimensionality reduction and surrogate modeling using Polynomial Chaos Expansion (PCE).

        ## How to Use This App

        ### **1. Define Your Model**

        - **Model Function:**
            - Define your model as a Python function named `model(X)`.
            - The function should take a NumPy array `X` as input, representing the input variables.
            - The function should return a list or array containing the output quantity of interest.

        - **Problem Definition:**
            - Create a dictionary named `problem` specifying the input variables and their distributions.
            - The dictionary should include:
                - `'num_vars'`: Number of input variables.
                - `'names'`: List of variable names.
                - `'distributions'`: List of distribution dictionaries for each variable.

        - **Example:**

        ```python
        import numpy as np

        def function_of_interest(X):
            x1, x2 = X
            Y = x1 ** 2 + np.sin(x2)
            return [Y]

        problem = {
            'num_vars': 2,
            'names': ['x1', 'x2'],
            'distributions': [
                {'type': 'Uniform', 'params': [0, 1]},
                {'type': 'Normal', 'params': [0, 1]}
            ]
        }
        model = function_of_interest
        ```

        ### **2. Enter or Select Your Model**

        - **Code Editor:**
            - You can either select a predefined model from the dropdown menu or input your own model code in the code editor.
            - The markdown interpretation of your model will update automatically, providing equations and definitions.

        ### **3. Run Monte Carlo Simulation and Analyses**

        - **Run Simulation:**
            - Click the **"Run Simulation"** button to perform Monte Carlo simulations.
            - The app will execute various analyses, including sensitivity analyses and uncertainty quantification.
            - AI-generated interpretations will provide insights into the results.

        ### **4. Dimensionality Reduction (Optional)**

        - **Purpose:**
            - If your model has a large number of input variables, you can perform dimensionality reduction to identify the most significant variables.
        - **Usage:**
            - Navigate to the **"Dimensionality Reduction"** page.
            - Run the AI-driven analysis to obtain a reduced set of input variables.
            - Use this information to simplify your model or focus on key inputs.

        ### **5. Surrogate Modeling with Polynomial Chaos Expansion (Optional)**

        - **Purpose:**
            - For computationally demanding models, you can create a surrogate model using Polynomial Chaos Expansion (PCE).
            - The surrogate model approximates your original model, allowing for faster analyses.
        - **Usage:**
            - Navigate to the **"Polynomial Chaos Expansion"** page.
            - Generate the PCE surrogate of your model.
            - Copy the generated surrogate model code back into the main page's code editor.
            - Run simulations and analyses using the surrogate to reduce computational costs.

        ## Workflow Summary

        1. **Define or Select Your Model** in the main page.
        2. **Run Monte Carlo Simulations** and perform initial analyses.
        3. **Perform Dimensionality Reduction** if you have many input variables.
        4. **Create a PCE Surrogate** if your model is computationally intensive.
        5. **Use the Surrogate Model** in place of the original model for further analyses.

        ## Additional Notes

        - **Supported Distributions:**
            - Uniform, Normal, LogNormalMuSigma, LogNormal, Beta, Gumbel, Triangular, etc.
            - Specify distributions in the `problem` dictionary with appropriate parameters.

        - **Imports:**
            - You can import external packages (e.g., `numpy`, `scipy`) within your model code.
            - Ensure that any packages you import are available in the app's environment.
        """)
