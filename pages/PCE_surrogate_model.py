import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chaospy as cp
import streamlit as st

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from modules.code_safety import check_code_safety
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
    try:
        file_path = os.path.join("examples", selected_model_name)
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

###############################################################################
# 2) PAGE SETUP
###############################################################################
st.set_page_config(layout="wide")
st.title("PCE Surrogate Model Generation using Chaospy")

with st.expander("Instructions"):
    st.markdown("""
    **PCE Surrogate Page**

    - Select or define your model code below.
    - Adjust Chaospy-based PCE parameters, then generate the surrogate.
    - Review validation plots to assess accuracy.
    - Copy the code for the final surrogate if desired.
    """)

initialize_session_state()

# Additional session keys
if "pce_generated" not in st.session_state:
    st.session_state.pce_generated = False
if "surrogate_model_code" not in st.session_state:
    st.session_state.surrogate_model_code = ""
if "validation_plot" not in st.session_state:
    st.session_state.validation_plot = None
if "r2" not in st.session_state:
    st.session_state.r2 = None
if "mae" not in st.session_state:
    st.session_state.mae = None
if "rmse" not in st.session_state:
    st.session_state.rmse = None
if "normalized_rmse" not in st.session_state:
    st.session_state.normalized_rmse = None

###############################################################################
# 3) SELECT MODEL FROM DROPDOWN OR UPLOAD, with on_change callback
###############################################################################
dropdown_items = ["(Select or define your own model)"] + model_options
previous_model = get_session_state("model_file", "(Select or define your own model)")

def on_model_change():
    new_model = st.session_state["pce_surr_selectbox"]
    if new_model == "(Select or define your own model)":
        st.session_state.code = ""
    else:
        st.session_state.code = load_model_code(new_model)
    st.session_state.model_file = new_model
    st.session_state.pce_generated = False
    st.session_state.surrogate_model_code = ""
    st.session_state.validation_plot = None
    reset_analysis_results()

col_select, col_upload = st.columns(2)

with col_select:
    st.selectbox(
        "Select a Model File or Enter your own Model:",
        dropdown_items,
        index=dropdown_items.index(previous_model) if previous_model in dropdown_items else 0,
        key="pce_surr_selectbox",
        on_change=on_model_change
    )

with col_upload:
    uploaded_file = st.file_uploader("or Choose a Python model file")
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        if st.button("Apply Uploaded File"):
            st.session_state.code = file_contents
            st.session_state.pce_generated = False
            st.session_state.surrogate_model_code = ""
            st.session_state.validation_plot = None
            reset_analysis_results()

###############################################################################
# 4) SIDE-BY-SIDE CODE EDITOR & PREVIEW
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
        st.session_state.pce_generated = False
        st.session_state.surrogate_model_code = ""
        st.session_state.validation_plot = None
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

col1, col2 = st.columns(2)
with col1:
    K = st.number_input("Order of Quadrature (K):", 1, 50, 3, 1)
    N = st.number_input("Order of Polynomial Expansion (N):", 1, 50, 3, 1)
    sparse = st.checkbox("Use Sparse Grid", value=False)
    growth = st.checkbox("Use Growth Rule", value=False)

with col2:
    n_validation = st.number_input("Number of Validation Samples:", 100, 10000, 500, 100)
    cross_truncation = st.slider("Cross Truncation:", 0.0, 1.0, 1.0, 0.05)

with st.expander("Advanced Settings"):
    quadrature_rules = {
        "Clenshaw-Curtis": "clenshaw_curtis",
        # ...
        "Gaussian": "gaussian"
    }
    orthogonal_methods = {
        "Three Terms Recurrence": "three_terms_recurrence",
        "Stieltjes": "stieltjes",
        "Cholesky": "cholesky",
        "Gram-Schmidt": "gram_schmidt",
    }
    recurrence_algorithms = ["stieltjes", "chebyshev"]

    quadrature_rule = st.selectbox(
        "Quadrature Rule:",
        list(quadrature_rules.keys()),
        index=list(quadrature_rules.values()).index("gaussian")
    )
    orth_method = st.selectbox(
        "Orthogonalization Method:",
        list(orthogonal_methods.keys()),
        index=list(orthogonal_methods.values()).index("three_terms_recurrence")
    )
    recurrence_algorithm = st.selectbox(
        "Recurrence Algorithm:",
        recurrence_algorithms,
        index=0
    )
    tolerance = st.number_input("Tolerance:", 1e-15, 1e-5, 1e-10, 1e-10, format="%.1e")
    scaling = st.number_input("Scaling:", 0.0, 10.0, 3.0, 0.1)

###############################################################################
# 6) GENERATE PCE SURROGATE MODEL
###############################################################################
run_pce_button = st.button("Generate PCE Surrogate Model")

if run_pce_button:
    code_str = st.session_state.get("code", "")
    if not code_str.strip():
        st.warning("No model code provided.")
        st.stop()

    try:
        check_code_safety(code_str)
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
        st.stop()

    globals_dict = {}
    try:
        exec(code_str, globals_dict)
        model = globals_dict.get("model")
        problem = globals_dict.get("problem")
        if model is None or problem is None:
            st.error("Model or problem definition not found in your code.")
            st.stop()
    except Exception as e:
        st.error(f"Error in executing model code: {e}")
        st.stop()

    with st.spinner("Generating PCE Surrogate Model..."):
        try:
            # Replicate your logic to build the PCE with chaospy
            # Evaluate the model, create expansions, etc.
            # Store metrics + figure in st.session_state
            # Build a polynomial expression for st.session_state.surrogate_model_code
            # ...
            st.session_state.pce_generated = True
            st.success("PCE Surrogate Model generated successfully.")
        except Exception as e:
            st.error(f"Error during PCE Surrogate Generation: {e}")
            st.stop()

###############################################################################
# 7) SHOW RESULTS & CODE
###############################################################################
if st.session_state.get("pce_generated"):
    # Show your validation plots, metrics, plus st.session_state.surrogate_model_code
    st.markdown("### Validation Plots")
    if st.session_state.validation_plot:
        st.pyplot(st.session_state.validation_plot)

    r2 = st.session_state.get("r2", None)
    mae = st.session_state.get("mae", None)
    rmse = st.session_state.get("rmse", None)
    normalized_rmse = st.session_state.get("normalized_rmse", None)
    if r2 is not None:
        st.markdown(f"**R²:** {r2:.4f}")
        st.markdown(f"**MAE:** {mae:.4f}")
        st.markdown(f"**RMSE:** {rmse:.4f}")
        st.markdown(f"**Normalized RMSE:** {normalized_rmse:.4f}")

    st.markdown("### Surrogate Model Code")
    st.code(st.session_state.surrogate_model_code, language='python')
