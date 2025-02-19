import os
import streamlit as st

# Import your modules
from modules.monte_carlo import monte_carlo_simulation
from modules.model_understanding import model_understanding
from modules.exploratory_data_analysis import exploratory_data_analysis
from modules.expectation_convergence_analysis import expectation_convergence_analysis
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis
from modules.taylor_analysis import taylor_analysis
from modules.correlation_analysis import correlation_analysis
from modules.hsic_analysis import hsic_analysis
from modules.ml_analysis import ml_analysis
from modules.markdown_utils import get_markdown_from_code
from modules.instructions import show_instructions
from modules.code_safety import check_code_safety
from modules.model_options_list import model_options
from modules.model_validation import (
    validate_problem_structure,
    get_human_friendly_error_explanation,
    test_model,
)
from modules.session_state_utils import (
    initialize_session_state,
    reset_analysis_results,
    get_session_state,
)

###############################################################################
# 1) SURROGATE DETECTION & SNIPPET EXTRACTION
###############################################################################
def is_surrogate_model(code_str: str) -> bool:
    """Returns True if the code snippet includes 'Y = metaModel(X)'."""
    return "Y = metaModel(X)" in code_str

def extract_surrogate_snippet(full_code: str) -> str:
    """
    If code is a surrogate, keep only lines from:
      def function_of_interest(X): ... up to ... model = function_of_interest
    Otherwise, return entire code.
    """
    if not is_surrogate_model(full_code):
        return full_code

    lines = full_code.splitlines(keepends=False)
    start_idx, end_idx = None, None

    for i, line in enumerate(lines):
        if "def function_of_interest(" in line:
            start_idx = i
            break

    for j in range(len(lines) - 1, -1, -1):
        if "model = function_of_interest" in lines[j]:
            end_idx = j
            break

    if (start_idx is None) or (end_idx is None) or (start_idx > end_idx):
        return full_code

    return "\n".join(lines[start_idx : end_idx + 1])

###############################################################################
# 2) LOAD MODEL CODE FROM EXAMPLES
###############################################################################
def load_model_code(selected_model_name: str) -> str:
    """
    Loads code from 'examples/' folder if a valid model is selected.
    Otherwise returns an empty string.
    """
    try:
        file_path = os.path.join('examples', selected_model_name)
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

###############################################################################
# 3) STREAMLIT APP START
###############################################################################
st.set_page_config(layout="wide")

col1, col2 = st.columns([1,4])
with col1:
    st.image("logo.jpg", width=100)
with col2:
    st.title("UncertaintyCat | v4.0")

show_instructions()
initialize_session_state()

###############################################################################
# 4) MODEL SELECT / UPLOAD
###############################################################################
# Insert placeholder item at index 0 for "no model selected"
dropdown_items = ["(Select or define your own model)"] + model_options

# We'll store the dropdown’s current selection in st.session_state["model_selectbox"]
# Use on_change callback to apply that selection immediately
def on_model_change():
    new_model = st.session_state["model_selectbox"]
    if new_model == "(Select or define your own model)":
        st.session_state.code = ""
    else:
        st.session_state.code = load_model_code(new_model)

    st.session_state.model_file = new_model
    st.session_state.run_simulation = False
    st.session_state.simulation_results = None
    st.session_state.markdown_output = None
    reset_analysis_results()

previous_model = get_session_state("model_file", "(Select or define your own model)")

col_select, col_upload = st.columns(2)

with col_select:
    st.selectbox(
        "Select a Model File or Enter your own Model:",
        dropdown_items,
        index=dropdown_items.index(previous_model) if previous_model in dropdown_items else 0,
        key="model_selectbox",
        on_change=on_model_change
    )

with col_upload:
    uploaded_file = st.file_uploader("or Choose a Python model file")
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        if st.button("Apply Uploaded File"):
            st.session_state.code = file_contents
            st.session_state.run_simulation = False
            st.session_state.simulation_results = None
            st.session_state.markdown_output = None
            reset_analysis_results()

###############################################################################
# 5) CODE EDITOR & SYNTAX-PREVIEW SIDE-BY-SIDE
###############################################################################
st.markdown("### Model Code Editor & Preview")

col_code, col_preview = st.columns(2)

with col_code:
    st.markdown("**Model Code Editor**")
    code_area_value = st.text_area(
        label="",
        value=st.session_state.get("code", ""),
        height=300
    )
    if code_area_value != st.session_state.get("code", ""):
        st.session_state.code = code_area_value
        st.session_state.run_simulation = False
        st.session_state.simulation_results = None
        st.session_state.markdown_output = None
        reset_analysis_results()

with col_preview:
    st.markdown("**Syntax-Highlighted Preview (Read Only)**")
    if st.session_state.get("code", "").strip():
        st.code(st.session_state["code"], language="python")
    else:
        st.info("No code to display.")

st.markdown("### Model Interpretation")
if st.session_state.get("code", "").strip():
    # If code changed or no existing markdown, re-run interpretation
    if not st.session_state.get("markdown_output"):
        snippet_for_ai = extract_surrogate_snippet(st.session_state["code"])
        with st.spinner("Generating markdown interpretation..."):
            markdown_output = get_markdown_from_code(snippet_for_ai, "qwen-2.5-32b")
            st.session_state.markdown_output = markdown_output

    if st.session_state.markdown_output:
        st.markdown(st.session_state.markdown_output)
    else:
        st.info("The markdown interpretation will appear here.")
else:
    st.info("No model code is currently provided.")

###############################################################################
# 6) LANGUAGE MODEL & ANALYSES
###############################################################################
groq_model_options = [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "qwen-2.5-32b",
    "deepseek-r1-distill-llama-70b-specdec"
]
selected_language_model = st.selectbox(
    "Select Language Model:",
    options=groq_model_options,
    index=0
)

st.markdown("### Select Analyses to Run")
analysis_options = {
    "Sobol Sensitivity Analysis": True,
    "Taylor Analysis": True,
    "Correlation Analysis": True,
    "HSIC Analysis": True,
    "SHAP Analysis": True
}
for key in analysis_options:
    analysis_options[key] = st.checkbox(key, value=True)

run_button = st.button("Run Simulation")

###############################################################################
# 7) MAIN SIMULATION LOGIC
###############################################################################
def run_simulation():
    code = st.session_state.get("code", "")
    if not code.strip():
        st.warning("No model code provided.")
        return

    surrogate_detected = is_surrogate_model(code)

    # 1) Check code safety
    try:
        check_code_safety(code)
    except Exception as e:
        if surrogate_detected:
            st.error("Surrogate model load error.")
            st.error(e)
        else:
            explanation = get_human_friendly_error_explanation(code, str(e), selected_language_model)
            st.error("Model Load Error:")
            st.error(explanation)
        return

    # 2) Execute user code
    globals_dict = {}
    try:
        exec(code, globals_dict)
        model = globals_dict.get("model")
        problem = globals_dict.get("problem")
        if model is None or problem is None:
            st.error("Model or problem definition not found in your code.")
            return
    except Exception as e:
        if surrogate_detected:
            st.error("Surrogate model load error.")
            st.error(e)
        else:
            explanation = get_human_friendly_error_explanation(code, str(e), selected_language_model)
            st.error("Model Load Error:")
            st.error(explanation)
        return

    # 3) If not surrogate, do validations & test
    if not surrogate_detected:
        try:
            validate_problem_structure(problem)
        except ValueError as ve:
            st.error(f"Validation Error: {ve}")
            return

        try:
            if not test_model(model, problem, code, selected_language_model):
                return
        except Exception as e:
            st.error("Model Test Error (non-surrogate).")
            st.error(str(e))
            return

    # 4) Hard-coded sample sizes
    is_pce_used = False
    original_model_code = code
    metamodel_str = None
    N = 2000
    N_samples = 8000
    N_sobol = 1024

    # 5) Monte Carlo
    try:
        with st.spinner("Running Monte Carlo Simulation..."):
            data = monte_carlo_simulation(N, model, problem)
    except Exception as e:
        if surrogate_detected:
            st.error("Surrogate Simulation Error:")
            st.error(e)
        else:
            explanation = get_human_friendly_error_explanation(code, str(e), selected_language_model)
            st.error("Simulation Error:")
            st.error(explanation)
        return

    # 6) Save results in session
    st.session_state.simulation_results = {
        "data": data,
        "model": model,
        "problem": problem,
        "code": code,
        "is_pce_used": is_pce_used,
        "original_model_code": original_model_code,
        "metamodel_str": metamodel_str,
        "selected_language_model": selected_language_model,
        "N": N,
        "N_samples": N_samples,
        "N_sobol": N_sobol,
        "analysis_options": analysis_options
    }

if run_button:
    st.session_state.run_simulation = True
    reset_analysis_results()

if st.session_state.get("run_simulation"):
    run_simulation()
    st.session_state.run_simulation = False

###############################################################################
# 8) PRESENT RESULTS
###############################################################################
if st.session_state.get("simulation_results"):
    results = st.session_state.simulation_results
    data = results["data"]
    model = results["model"]
    problem = results["problem"]
    code = results["code"]
    is_pce_used = results["is_pce_used"]
    original_model_code = results["original_model_code"]
    metamodel_str = results["metamodel_str"]
    selected_language_model = results["selected_language_model"]
    N = results["N"]
    N_samples = results["N_samples"]
    N_sobol = results["N_sobol"]
    analysis_options = results["analysis_options"]

    snippet_for_modules = extract_surrogate_snippet(code)

    st.markdown("---")
    st.header("Model Understanding")
    with st.spinner("Running Model Understanding..."):
        model_understanding(
            model,
            problem,
            snippet_for_modules,
            is_pce_used=is_pce_used,
            original_model_code_str=original_model_code,
            metamodel_str=metamodel_str,
            language_model=selected_language_model
        )

    st.markdown("---")
    st.header("Exploratory Data Analysis")
    with st.spinner("Running Exploratory Data Analysis..."):
        exploratory_data_analysis(
            data, N, model, problem, snippet_for_modules,
            language_model=selected_language_model
        )

    st.markdown("---")
    st.header("Expectation Convergence Analysis")
    with st.spinner("Running Expectation Convergence Analysis..."):
        expectation_convergence_analysis(
            model, problem, snippet_for_modules,
            N_samples=N_samples,
            language_model=selected_language_model
        )

    if analysis_options["Sobol Sensitivity Analysis"]:
        st.markdown("---")
        st.header("Sobol Sensitivity Analysis")
        with st.spinner("Running Sobol Sensitivity Analysis..."):
            sobol_sensitivity_analysis(
                N_sobol, model, problem, snippet_for_modules,
                language_model=selected_language_model
            )

    if analysis_options["Taylor Analysis"]:
        st.markdown("---")
        st.header("Taylor Analysis")
        with st.spinner("Running Taylor Analysis..."):
            taylor_analysis(
                model, problem, snippet_for_modules,
                language_model=selected_language_model
            )

    if analysis_options["Correlation Analysis"]:
        st.markdown("---")
        st.header("Correlation Analysis")
        with st.spinner("Running Correlation Analysis..."):
            correlation_analysis(
                model, problem, snippet_for_modules,
                language_model=selected_language_model
            )

    if analysis_options["HSIC Analysis"]:
        st.markdown("---")
        st.header("HSIC Analysis")
        with st.spinner("Running HSIC Analysis..."):
            hsic_analysis(
                model, problem, snippet_for_modules,
                language_model=selected_language_model
            )

    if analysis_options["SHAP Analysis"]:
        st.markdown("---")
        st.header("SHAP Analysis")
        with st.spinner("Running SHAP Analysis..."):
            ml_analysis(
                data, problem, snippet_for_modules,
                language_model=selected_language_model
            )

else:
    st.info("Please run the simulation to proceed.")
