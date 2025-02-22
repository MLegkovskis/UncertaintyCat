import os
import ast
import importlib
import pandas as pd
import streamlit as st
import openturns as ot

from modules.morris_sensitivity_analysis import run_morris_analysis_for_dimensionality_reduction
from modules.api_utils import call_groq_api
from modules.code_safety import check_code_safety
from modules.model_options_list import model_options
from modules.session_state_utils import initialize_session_state, reset_analysis

# Initialize session state
initialize_session_state()

# Page setup
st.title("Dimensionality Reduction")
st.markdown("""
This page helps you identify and fix non-influential variables in your model:
1. First, it runs a Morris sensitivity analysis to identify variables with low influence
2. Then, you can choose which variables to fix at constant values
3. Finally, it generates a reduced version of your model with those variables fixed
""")

# Model selection
def load_model_code(filename):
    """Load code from examples folder."""
    try:
        with open(os.path.join('examples', filename), 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

# Model dropdown
model_options_list = ["(Select or define your own model)"] + model_options
selected_model = st.selectbox(
    "Select a Model File:",
    model_options_list,
    key="model_file",
    on_change=reset_analysis
)

# Get model code
if selected_model != "(Select or define your own model)":
    st.session_state.code = load_model_code(selected_model)
    st.code(st.session_state.code, language='python')
else:
    st.session_state.code = st.text_area(
        "Or paste your model code here:",
        value=st.session_state.code,
        height=300,
        on_change=reset_analysis
    )

# Execute code if present
if st.session_state.code:
    if not check_code_safety(st.session_state.code):
        st.error("Code contains unsafe operations. Please review and modify.")
    else:
        try:
            # Execute code to get model and problem
            local_dict = {}
            exec(st.session_state.code, globals(), local_dict)
            
            if 'model' not in local_dict or 'problem' not in local_dict:
                st.error("Code must define 'model' and 'problem' variables.")
            else:
                # Run Morris analysis
                if st.button("Run Morris Analysis"):
                    with st.spinner("Running Morris Analysis..."):
                        st.session_state.morris_results = run_morris_analysis_for_dimensionality_reduction(
                            N=500,
                            model=local_dict['model'],
                            problem=local_dict['problem']
                        )

        except Exception as e:
            st.error(f"Error executing code: {e}")

# Display Morris results if available
if st.session_state.morris_results:
    st.markdown("### Morris Results")
    non_influential_indices, mu_star_values, sigma_values, morris_plot = st.session_state.morris_results
    problem = local_dict['problem']

    # Display Morris plot
    st.plotly_chart(morris_plot)

    # Get variable names from OpenTURNS distribution
    dimension = problem.getDimension()
    names = [problem.getMarginal(i).getDescription()[0] for i in range(dimension)]

    # Show non-influential variables
    if non_influential_indices:
        st.markdown("#### Non-influential Variables Detected")
        non_influential_vars = [names[i] for i in non_influential_indices]
        st.write("The following variables have low influence on the output:")
        st.write(", ".join(non_influential_vars))

        # Let user select which variables to fix
        variables_to_fix = st.multiselect(
            "Select variables to fix at constant values:",
            non_influential_vars
        )

        if variables_to_fix:
            # Get fixed values
            fixed_values = {}
            for var in variables_to_fix:
                idx = names.index(var)
                marginal = problem.getMarginal(idx)
                default_value = marginal.getMean()[0]
                fixed_values[var] = st.number_input(
                    f"Value for {var}",
                    value=float(default_value),
                    format="%.6g"
                )

            st.session_state.fixed_values = fixed_values

            # Generate reduced model
            if st.button("Generate Reduced Model"):
                try:
                    with st.spinner("Generating reduced model..."):
                        prompt = f"""
Here is the original model:

{st.session_state.code}

The following variables should be fixed to these values:
{st.session_state.fixed_values}

Generate a reduced version of the model that:
1. Only includes the remaining variables in the function signature
2. Sets the fixed variables to their constant values inside the function
3. Updates the OpenTURNS problem definition to only include the remaining variables
4. Maintains any existing correlations between the remaining variables
"""
                        st.session_state.reduced_model = call_groq_api(prompt)
                except Exception as e:
                    st.error(f"Error generating reduced model: {e}")
    else:
        st.info("No non-influential variables detected.")

# Display reduced model if available
if st.session_state.reduced_model:
    st.markdown("### Reduced Model")
    st.info("Copy this code to use the reduced model:")
    st.code(st.session_state.reduced_model, language='python')
