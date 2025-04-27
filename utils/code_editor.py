import streamlit as st
import numpy as np
from modules.monte_carlo import monte_carlo_simulation
from utils.core_utils import check_code_safety

def render_code_editor(current_code):
    """
    Renders the model code editor and syntax-highlighted preview in the Streamlit app.
    Returns the (possibly updated) current_code string.
    """
    st.header("Model Definition")
    with st.expander("Model Code Editor & Preview", expanded=True):
        col_code, col_preview = st.columns(2)

        with col_code:
            st.subheader("Model Code Editor")
            st.caption("Define your model using Python 3.12. You have access to numpy, scipy, and openturns libraries. Your code must define 'model' (an OpenTURNS Function) and 'problem' (an OpenTURNS Distribution).")
            code_area_value = st.text_area(
                label="",
                value=current_code,
                height=300
            )
            # Update current code if changed in the editor
            current_code = code_area_value
            
            # Add a model validation button
            if st.button("Validate Model Input"):
                if not current_code:
                    st.error("Please provide a model first or select one of the example models to start with.")
                else:
                    try:
                        # First check if the code is safe to execute
                        is_safe, safety_message = check_code_safety(current_code)
                        
                        if not is_safe:
                            st.error(f"Security Error: {safety_message}")
                        else:
                            # Execute the code
                            eval_globals = {}
                            exec(current_code, eval_globals)
                            
                            # Get model and problem
                            model = eval_globals.get('model')
                            problem = eval_globals.get('problem')
                            
                            if not model or not problem:
                                st.error("Model code must define 'model' and 'problem' variables.")
                            else:
                                # Run a small Monte Carlo simulation
                                with st.spinner("Running 10 Monte Carlo samples..."):
                                    try:
                                        results = monte_carlo_simulation(model, problem, N=10, seed=42)
                                        # Check if mean is a numpy array or scalar
                                        mean_value = results['mean']
                                        std_value = results['std']
                                        if isinstance(mean_value, np.ndarray):
                                            mean_str = f"{mean_value[0]:.4f}"
                                            std_str = f"{std_value[0]:.4f}"
                                        else:
                                            mean_str = f"{mean_value:.4f}"
                                            std_str = f"{std_value:.4f}"
                                        st.success(f"Model validated successfully! Sample mean: {mean_str}, std: {std_str}")
                                    except Exception as e:
                                        st.error(f"Error running model: {e}")
                    except Exception as e:
                        st.error(f"Error evaluating model code: {e}")

        with col_preview:
            st.subheader("Syntax-Highlighted Preview")
            if current_code.strip():
                st.code(current_code, language="python")
            else:
                st.info("No code to display. Please select or upload a model.")
    return current_code
