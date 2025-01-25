import os
import ast
import importlib
import pandas as pd
import streamlit as st

from modules.morris_sensitivity_analysis import run_morris_analysis_for_dimensionality_reduction
from modules.api_utils import call_groq_api
from modules.statistical_utils import get_constant_value
from modules.code_safety import check_code_safety
from modules.model_options_list import model_options
from modules.session_state_utils import (
    initialize_session_state,
    reset_analysis_results,
    get_session_state
)

###############################################################################
# 1) HELPER: Extract & Import from Code
###############################################################################
def extract_imports_from_code(code_str):
    imports = {}
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    asname = alias.asname or alias.name
                    imports[asname] = importlib.import_module(module_name)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module
                for alias in node.names:
                    asname = alias.asname or alias.name
                    module = importlib.import_module(module_name)
                    imports[asname] = getattr(module, alias.name)
    except Exception as e:
        st.error(f"Error processing imports: {e}")
    return imports

def load_model_code(selected_model_name: str) -> str:
    """Load code from examples/ folder."""
    try:
        with open(os.path.join('examples', selected_model_name), 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

###############################################################################
# 2) PAGE LAYOUT
###############################################################################
st.set_page_config(layout="wide")
st.title("Dimensionality Reduction using AI and Morris Analysis")

with st.expander("Instructions"):
    st.markdown("""
    **Instructions:**
    - Select a model from the dropdown or input your own in the code editor.
    - Include any necessary imports (e.g., `import numpy as np`) in your model.
    - Run the Morris Sensitivity Analysis.
    - Choose variables to fix and adjust their values as needed.
    - Let the AI generate the reduced model code, ensuring user-defined imports are preserved.
    """)

initialize_session_state()

if "morris_analysis_done" not in st.session_state:
    st.session_state.morris_analysis_done = False
if "morris_results" not in st.session_state:
    st.session_state.morris_results = None
if "variables_to_fix" not in st.session_state:
    st.session_state.variables_to_fix = []
if "constant_values" not in st.session_state:
    st.session_state.constant_values = {}
if "reduced_model_code" not in st.session_state:
    st.session_state.reduced_model_code = ""

###############################################################################
# 3) MODEL FILE DROPDOWN / UPLOAD with on_change
###############################################################################
dropdown_items = ["(Select or define your own model)"] + model_options
prev_file = get_session_state("model_file", "(Select or define your own model)")

def on_model_change():
    new_model = st.session_state["dimr_model_selectbox"]
    if new_model == "(Select or define your own model)":
        st.session_state.code = ""
    else:
        st.session_state.code = load_model_code(new_model)

    st.session_state.model_file = new_model
    st.session_state.morris_analysis_done = False
    st.session_state.morris_results = None
    st.session_state.variables_to_fix = []
    st.session_state.constant_values = {}
    st.session_state.reduced_model_code = ""
    reset_analysis_results()

col_select, col_upload = st.columns(2)

with col_select:
    st.selectbox(
        "Select a Model File:",
        dropdown_items,
        index=dropdown_items.index(prev_file) if prev_file in dropdown_items else 0,
        key="dimr_model_selectbox",
        on_change=on_model_change
    )

with col_upload:
    uploaded_file = st.file_uploader("or Choose a Python model file")
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        if st.button("Apply Uploaded File"):
            st.session_state.code = file_contents
            st.session_state.morris_analysis_done = False
            st.session_state.morris_results = None
            st.session_state.variables_to_fix = []
            st.session_state.constant_values = {}
            st.session_state.reduced_model_code = ""
            reset_analysis_results()

###############################################################################
# 4) MODEL CODE EDITOR (TEXT AREA)
###############################################################################
st.markdown("### Model Code Editor")
new_code = st.text_area(
    label="",
    value=st.session_state.get("code", ""),
    height=400
)
if new_code != st.session_state.get("code", ""):
    st.session_state.code = new_code
    st.session_state.morris_analysis_done = False
    st.session_state.morris_results = None
    st.session_state.variables_to_fix = []
    st.session_state.constant_values = {}
    st.session_state.reduced_model_code = ""
    reset_analysis_results()

###############################################################################
# 5) RUN MORRIS ANALYSIS
###############################################################################
if st.button("Run Morris Sensitivity Analysis"):
    code_str = st.session_state.get("code", "")
    if not code_str.strip():
        st.warning("No model code found. Please paste or upload a model.")
        st.stop()

    # 1) Check code safety
    try:
        check_code_safety(code_str)
    except Exception as e:
        st.error(f"Code safety check failed: {e}")
        st.stop()

    # 2) Extract & import modules
    user_imports = extract_imports_from_code(code_str)
    globals_dict = {}
    globals_dict.update(user_imports)

    # 3) Execute user code
    try:
        exec(code_str, globals_dict)
    except Exception as e:
        st.error(f"Error executing model code: {e}")
        st.stop()

    # 4) Retrieve 'model' and 'problem'
    model = globals_dict.get('model')
    problem = globals_dict.get('problem')
    if not model or not problem:
        st.error("Your code must define a 'model' and a 'problem'.")
        st.stop()

    # 5) Morris analysis
    try:
        with st.spinner("Running Morris Analysis..."):
            results = run_morris_analysis_for_dimensionality_reduction(100, model, problem)
        st.session_state.morris_analysis_done = True
        st.session_state.morris_results = results
        st.session_state.problem = problem
        st.session_state.model = model
    except Exception as e:
        st.error(f"Morris Analysis Error: {e}")
        st.stop()

###############################################################################
# 6) SHOW RESULTS & ALLOW FIXING VARIABLES
###############################################################################
if st.session_state.morris_analysis_done and st.session_state.morris_results:
    non_influential_indices, mu_star_values, sigma_values, morris_plot = st.session_state.morris_results
    problem = st.session_state.problem
    model = st.session_state.model
    names = problem.get('names', [])

    st.write("### Morris Sensitivity Analysis Results")
    recommendations = ['Recommended' if i in non_influential_indices else '' for i in range(len(names))]
    morris_df = pd.DataFrame({
        'Parameter': names,
        'μ*': mu_star_values,
        'σ': sigma_values,
        'Recommendation': recommendations
    })

    st.dataframe(morris_df.style.apply(
        lambda row: [
            'background-color: lightgreen' if row['Recommendation'] == 'Recommended' else ''
            for _ in row
        ],
        axis=1
    ))

    st.markdown("""
    **Threshold Lines Explanation:**

    - The **vertical dashed line** represents the threshold for μ* (Mean of Absolute Elementary Effects), calculated as the 25th percentile of μ* values.
    - The **horizontal dashed line** represents the threshold for σ (Standard Deviation of EE), calculated as the 25th percentile of σ values.
    - Variables plotted **below** both thresholds are considered **non-influential** and are recommended for fixing.

    **How Recommendations are Made:**

    - Variables with both μ* and σ less than their respective thresholds (lower quartiles) are deemed non-influential.
    - Fixing non-influential variables can simplify the model without significantly affecting output variability.
    """)

    st.plotly_chart(morris_plot, use_container_width=True)

    variables_to_fix = st.multiselect(
        "Variables to Fix:",
        options=names,
        default=st.session_state.variables_to_fix,
        help="Variables recommended for fixing are marked in green above."
    )
    st.session_state.variables_to_fix = variables_to_fix

    if variables_to_fix:
        indices_to_fix = [names.index(var) for var in variables_to_fix]
        constant_values = {names[i]: get_constant_value(problem['distributions'][i]) for i in indices_to_fix}
        for k, v in constant_values.items():
            if k not in st.session_state.constant_values:
                st.session_state.constant_values[k] = v

        st.write("### Fixed Variables & Assigned Constant Values")
        fixed_vars_df = pd.DataFrame({
            'Variable': list(st.session_state.constant_values.keys()),
            'Assigned Value': list(st.session_state.constant_values.values())
        })

        edited_df = st.data_editor(
            fixed_vars_df,
            column_config={'Variable': st.column_config.Column(disabled=True)},
            key="fixed_vars_editor"
        )
        st.session_state.constant_values = dict(zip(edited_df['Variable'], edited_df['Assigned Value']))

        if st.button("Generate Reduced Model Definition"):
            code_for_ai = st.session_state.code
            prompt = (
                f"Here is the original model:\n\n{code_for_ai}\n\n"
                f"The user has selected the following variables to fix:\n{st.session_state.constant_values}\n\n"
                "Please provide a reduced version of the model in Python, following these requirements:\n"
                "1. Fixed variables must be initialized inside the function body with their fixed values.\n"
                "2. The 'function_of_interest' should take only the remaining variables as input.\n"
                "3. The updated 'problem' definition should:\n"
                "   a. Reflect the reduced number of variables by updating 'problem['num_vars']'.\n"
                "   b. Update 'problem['names']' to list only the remaining variable names.\n"
                "   c. Update 'problem['distributions']' to include distributions for only the remaining variables.\n"
                "4. The generated code should be valid Python, without any additional explanations or comments.\n\n"
                "5. You must preserve any imports or external libraries used in the original model.\n"
                "6. The reduced model must not change the underlying logic or calculations of the original model.\n\n"
                "For example, if the original model was:\n"
                "```\n"
                "import numpy as np\n"
                "def function_of_interest(X):\n"
                "    A, B, C = X\n"
                "    Z = A * B + C\n"
                "    return Z\n\n"
                "problem = {\n"
                "    'num_vars': 3,\n"
                "    'names': ['A', 'B', 'C'],\n"
                "    'distributions': [\n"
                "        {'type': 'Uniform', 'params': [0, 1]},\n"
                "        {'type': 'Normal', 'params': [1, 2]},\n"
                "        {'type': 'Beta', 'params': [2, 3]}\n"
                "    ]\n"
                "}\n\n"
                "model = function_of_interest\n"
                "```\n"
                "And the user fixed `C` to 0.5, the reduced model should be:\n"
                "```\n"
                "import numpy as np\n"
                "def function_of_interest(X):\n"
                "    A, B = X\n"
                "    C = 0.5\n"
                "    Z = A * B + C\n"
                "    return Z\n\n"
                "problem = {\n"
                "    'num_vars': 2,\n"
                "    'names': ['A', 'B'],\n"
                "    'distributions': [\n"
                "        {'type': 'Uniform', 'params': [0, 1]},\n"
                "        {'type': 'Normal', 'params': [1, 2]}\n"
                "    ]\n"
                "}\n\n"
                "model = function_of_interest\n"
                "```\n"
                "Another example is:\n"
                "```\n"
                "import numpy as np\n"
                "from scipy.optimize import fsolve\n\n"
                "def bike_speed_eq(speed, *params):\n"
                "    rider_power, air_density, drag_coefficient, frontal_area, rolling_resistance, bike_mass, gravity_earth = params\n"
                "    return rider_power - (\n"
                "        0.5 * air_density * drag_coefficient * frontal_area * speed**3 +\n"
                "        rolling_resistance * bike_mass * gravity_earth * speed\n"
                "    )\n\n"
                "def function_of_interest(inputs):\n"
                "    tire_radius, bike_mass, rider_power, air_density, rolling_resistance, drag_coefficient, gravity_earth, frontal_area = inputs\n"
                "    initial_guess = 10\n"
                "    params = (rider_power, air_density, drag_coefficient, frontal_area, rolling_resistance, bike_mass, gravity_earth)\n"
                "    speed_solution = fsolve(bike_speed_eq, initial_guess, args=params)\n"
                "    return [speed_solution[0]]\n\n"
                "problem = {\n"
                "    'num_vars': 8,\n"
                "    'names': ['tire_radius', 'bike_mass', 'rider_power', 'air_density', 'rolling_resistance', 'drag_coefficient', 'gravity_earth', 'frontal_area'],\n"
                "    'distributions': [\n"
                "        {'type': 'Uniform', 'params': [0.3, 0.7]},\n"
                "        {'type': 'Uniform', 'params': [5.0, 15.0]},\n"
                "        {'type': 'Uniform', 'params': [150.0, 400.0]},\n"
                "        {'type': 'Uniform', 'params': [1.18, 1.3]},\n"
                "        {'type': 'Uniform', 'params': [0.002, 0.005]},\n"
                "        {'type': 'Uniform', 'params': [0.7, 1.0]},\n"
                "        {'type': 'Uniform', 'params': [9.78, 9.82]},\n"
                "        {'type': 'Uniform', 'params': [0.3, 0.6]}\n"
                "    ]\n"
                "}\n\n"
                "model = function_of_interest\n"
                "```\n"
                "If the user fixed 'tire_radius' to 0.5 and 'gravity_earth' to 9.8, the reduced model should be:\n"
                "```\n"
                "import numpy as np\n"
                "from scipy.optimize import fsolve\n\n"
                "def bike_speed_eq(speed, *params):\n"
                "    rider_power, air_density, drag_coefficient, frontal_area, rolling_resistance, bike_mass, _ = params\n"
                "    return rider_power - (\n"
                "        0.5 * air_density * drag_coefficient * frontal_area * speed**3 +\n"
                "        rolling_resistance * bike_mass * 9.8 * speed\n"
                "    )\n\n"
                "def function_of_interest(inputs):\n"
                "    bike_mass, rider_power, air_density, rolling_resistance, drag_coefficient, frontal_area = inputs\n"
                "    tire_radius = 0.5\n"
                "    gravity_earth = 9.8\n"
                "    initial_guess = 10\n"
                "    params = (rider_power, air_density, drag_coefficient, frontal_area, rolling_resistance, bike_mass, gravity_earth)\n"
                "    speed_solution = fsolve(bike_speed_eq, initial_guess, args=params)\n"
                "    return [speed_solution[0]]\n\n"
                "problem = {\n"
                "    'num_vars': 6,\n"
                "    'names': ['bike_mass', 'rider_power', 'air_density', 'rolling_resistance', 'drag_coefficient', 'frontal_area'],\n"
                "    'distributions': [\n"
                "        {'type': 'Uniform', 'params': [5.0, 15.0]},\n"
                "        {'type': 'Uniform', 'params': [150.0, 400.0]},\n"
                "        {'type': 'Uniform', 'params': [1.18, 1.3]},\n"
                "        {'type': 'Uniform', 'params': [0.002, 0.005]},\n"
                "        {'type': 'Uniform', 'params': [0.7, 1.0]},\n"
                "        {'type': 'Uniform', 'params': [0.3, 0.6]}\n"
                "    ]\n"
                "}\n\n"
                "model = function_of_interest\n"
                "```\n"
                "Ensure that the updated 'num_vars', 'names', and 'distributions' in the 'problem' definition are consistent with the reduced inputs.\n"
                "Now, generate the reduced model for the given user-defined input."
            )
            try:
                with st.spinner("Generating reduced model..."):
                    from modules.api_utils import call_groq_api
                    st.session_state.reduced_model_code = call_groq_api(
                        prompt, model_name='llama-3.3-70b-versatile'
                    ).replace("```python", "").replace("```", "")
            except Exception as e:
                st.error(f"Error during AI generation: {e}")

if st.session_state.reduced_model_code:
    st.markdown("### Reduced Model Definition (Copy-Paste into the main app!):")
    st.code(st.session_state.reduced_model_code, language='python')
