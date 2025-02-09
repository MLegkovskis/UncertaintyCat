import os
import streamlit as st

from modules.code_safety import check_code_safety
from modules.pce_least_squares_utils import pce_sobol, reset_pce_least_squares_results
from modules.model_options_list import model_options
from modules.session_state_utils import (
    initialize_session_state,
    reset_analysis_results,
    get_session_state
)

###############################################################################
# 1) LOAD CODE FROM EXAMPLES
###############################################################################
def load_model_code(selected_model_name: str) -> str:
    """Loads code from examples/ folder."""
    try:
        with open(os.path.join("examples", selected_model_name), "r") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

###############################################################################
# 2) PAGE SETUP
###############################################################################
st.set_page_config(layout="wide")
st.title("PCE Surrogate Model Generation using OpenTURNS")

with st.expander("Instructions"):
    st.markdown(
        """
    **Least Squares PCE Surrogate Page**

    Generate a Polynomial Chaos Expansion (PCE) surrogate model using OpenTURNS.
    Coefficients are estimated with least squares.

    **Instructions:**
    1. Select or define your model code below.
    2. Adjust the PCE parameters, then run the analysis.
    3. Review validation plots to assess surrogate accuracy.
    4. Optionally copy or download the final surrogate model code.
    """
    )

initialize_session_state()

###############################################################################
# 3) MODEL SELECTION with on_change
###############################################################################

# Insert placeholder at index 0 for "no model"
model_dropdown_items = ["(Select or define your own model)"] + model_options
previous_model = get_session_state("model_file", "(Select or define your own model)")

def on_model_change():
    """Callback invoked when the dropdown selection changes."""
    new_model = st.session_state["pce_lsq_selectbox"]
    if new_model == "(Select or define your own model)":
        st.session_state.code = ""
    else:
        st.session_state.code = load_model_code(new_model)

    st.session_state.model_file = new_model
    # Reset results
    reset_pce_least_squares_results()
    reset_analysis_results()

st.markdown("#### Select or Define Model")
st.selectbox(
    "Select a Model File or Enter your own Model:",
    model_dropdown_items,
    index=model_dropdown_items.index(previous_model) if previous_model in model_dropdown_items else 0,
    key="pce_lsq_selectbox",
    on_change=on_model_change
)

###############################################################################
# 4) SIDE-BY-SIDE CODE EDITOR + PREVIEW
###############################################################################
st.markdown("### Model Code Editor & Preview")

col_editor, col_preview = st.columns(2)

with col_editor:
    st.markdown("**Model Code Editor**")
    code_area_value = st.text_area(
        label="",
        value=st.session_state.get("code", ""),
        height=300
    )
    if code_area_value != st.session_state.get("code", ""):
        st.session_state.code = code_area_value
        reset_pce_least_squares_results()
        reset_analysis_results()

with col_preview:
    st.markdown("**Syntax-Highlighted Preview (Read Only)**")
    if st.session_state.get("code", "").strip():
        st.code(st.session_state["code"], language="python")
    else:
        st.info("No code to display.")

###############################################################################
# 5) PCE PARAMETERS
###############################################################################
st.markdown("### PCE Parameters")

training_sample_size = st.number_input(
    "Training Sample Size:",
    min_value=100,
    max_value=10000,
    value=500,
    step=100
)
validation_sample_size = st.number_input(
    "Validation Sample Size:",
    min_value=100,
    max_value=10000,
    value=500,
    step=100
)
use_model_selection = st.checkbox(
    "Use model selection",
    value=False
)
basis_size_factor = st.slider(
    "Basis Size Factor",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

groq_model_options = ["gemma2-9b-it", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
selected_language_model = st.selectbox(
    "Select Language Model:",
    options=groq_model_options,
    index=0
)

###############################################################################
# 6) RUN PCE Surrogate
###############################################################################
if st.button("Generate PCE Surrogate Model"):
    reset_pce_least_squares_results()
    reset_analysis_results()

    code_str = st.session_state.get("code", "")
    if not code_str.strip():
        st.warning("No model code provided.")
        st.stop()

    # Code safety check
    try:
        check_code_safety(code_str)
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
        st.stop()

    # Execute user code
    globals_dict = {}
    try:
        exec(code_str, globals_dict)
        model = globals_dict.get("model")
        problem = globals_dict.get("problem")
        if model is None or problem is None:
            st.error("No 'model' or 'problem' found in code.")
            st.stop()
    except Exception as e:
        st.error(f"Error executing code: {e}")
        st.stop()

    with st.spinner("Generating PCE Surrogate Model..."):
        pce_sobol(
            train_sample_size=training_sample_size,
            validation_sample_size=validation_sample_size,
            model=model,
            problem=problem,
            model_code_str=code_str,
            language_model=selected_language_model,
            basis_size_factor=basis_size_factor,
            use_model_selection=use_model_selection
        )
