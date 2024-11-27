import streamlit as st
from streamlit_ace import st_ace
import ast
import pandas as pd
from modules.morris_sensitivity_analysis import run_morris_analysis_for_dimensionality_reduction
from modules.api_utils import call_groq_api
from modules.statistical_utils import get_constant_value
import numpy as np  # Commonly used library
import importlib

st.set_page_config(layout="wide")

st.title('Dimensionality Reduction using AI and Morris Analysis')

with st.expander("Instructions"):
    st.markdown("""
    **Instructions:**
    - Select a model from the dropdown or input your own in the code editor.
    - Include any necessary imports (e.g., `import numpy as np`) in your model.
    - Run the Morris Sensitivity Analysis.
    - Choose variables to fix and adjust their values as needed.
    - Let the AI generate the reduced model code, ensuring user-defined imports are preserved.
    """)

# Initialize session state variables
if 'model_file' not in st.session_state:
    st.session_state.model_file = 'Beam.py'
if 'code_editor' not in st.session_state:
    try:
        st.session_state.code_editor = open(f'examples/{st.session_state.model_file}').read()
    except FileNotFoundError:
        st.session_state.code_editor = "def function_of_interest(X):\n    pass\n\nproblem = {}\nmodel = function_of_interest"
if 'morris_analysis_done' not in st.session_state:
    st.session_state.morris_analysis_done = False
if 'variables_to_fix' not in st.session_state:
    st.session_state.variables_to_fix = []
if 'constant_values' not in st.session_state:
    st.session_state.constant_values = {}
if 'reduced_model_code' not in st.session_state:
    st.session_state.reduced_model_code = ''

# Function to dynamically extract and import modules from code
def extract_imports_from_code(code_str):
    """
    Extract and dynamically import modules specified in the code.
    Returns a dictionary of dynamically imported modules or objects.
    """
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

# Function to check code safety
class UnsafeNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.disallowed_modules = {'os', 'sys', 'subprocess', 'shutil'}
        super().__init__()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in self.disallowed_modules:
                raise ValueError(
                    f"Importing module '{alias.name}' is not allowed for security reasons."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in self.disallowed_modules:
            raise ValueError(
                f"Importing from module '{node.module}' is not allowed for security reasons."
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', '__import__']:
                raise ValueError(
                    f"Use of function '{node.func.id}' is not allowed for security reasons."
                )
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in self.disallowed_modules:
                    raise ValueError(
                        f"Use of module '{node.func.value.id}' is not allowed for security reasons."
                    )
            if node.func.attr in ['system', 'popen', 'remove', 'rmdir']:
                raise ValueError(
                    f"Use of method '{node.func.attr}' is not allowed for security reasons."
                )
        self.generic_visit(node)

def check_code_safety(code_str):
    try:
        tree = ast.parse(code_str)
        UnsafeNodeVisitor().visit(tree)
        return True
    except Exception as e:
        raise ValueError(f"Unsafe code detected: {e}")

# Dropdown to select model file
model_options = [
    'Beam.py', 'Bike_Speed.py', 'Borehole_Model.py', 'Chemical_Reactor.py',
    'Cylinder_heating.py', 'Damped_Oscillator.py', 'Epidemic_Model.py',
    'FloodModel.py', 'Ishigami.py', 'Material_Stress.py', 'Morris_Function.py',
    'Portfolio_Risk.py', 'Rocket_Trajectory.py', 'Solar_Panel_Output.py',
    'Truss_Model.py', 'Undamped_Oscillator.py', 'Wind_Turbine_Power.py'
]

model_file = st.selectbox(
    'Select a Model File:',
    model_options,
    index=model_options.index(st.session_state.model_file)
)

if model_file != st.session_state.model_file:
    st.session_state.model_file = model_file
    try:
        st.session_state.code_editor = open(f'examples/{model_file}').read()
    except FileNotFoundError:
        st.session_state.code_editor = "def function_of_interest(X):\n    pass\n\nproblem = {}\nmodel = function_of_interest"
    st.session_state.morris_analysis_done = False
    st.session_state.variables_to_fix = []
    st.session_state.constant_values = {}
    st.session_state.reduced_model_code = ''

# Display code editor
st.markdown("### Model Code Editor")
code = st_ace(
    value=st.session_state.code_editor,
    language='python',
    theme='xcode',
    height=400,
    key=f"code_editor_{st.session_state.model_file}"
)

if code != st.session_state.code_editor:
    st.session_state.code_editor = code

# Run Morris Analysis Button
if st.button("Run Morris Sensitivity Analysis"):
    try:
        # Step 1: Check for unsafe code
        check_code_safety(code)

        # Step 2: Dynamically extract and import modules
        user_imports = extract_imports_from_code(code)

        # Step 3: Prepare the globals dictionary with extracted imports
        globals_dict = {}
        globals_dict.update(user_imports)

        # Step 4: Execute the user's code in this namespace
        exec(code, globals_dict)

        # Step 5: Retrieve 'model' and 'problem' from the executed code
        model = globals_dict.get('model')
        problem = globals_dict.get('problem')

        if not model or not problem:
            st.error("Your code must define a 'model' and a 'problem'.")
            st.stop()

        # Step 6: Run Morris Analysis
        with st.spinner("Running Morris Analysis..."):
            results = run_morris_analysis_for_dimensionality_reduction(100, model, problem)
            st.session_state.morris_analysis_done = True
            st.session_state.morris_results = results
            st.session_state.problem = problem
            st.session_state.model = model
    except Exception as e:
        st.error(f"Error during execution: {e}")

# After running Morris analysis
if st.session_state.morris_analysis_done:
    non_influential_indices, mu_star_values, sigma_values, morris_plot = st.session_state.morris_results
    problem = st.session_state.problem
    names = problem['names']

    # Display Morris analysis results
    recommendations = ['Recommended' if i in non_influential_indices else '' for i in range(len(names))]
    morris_df = pd.DataFrame({
        'Parameter': names,
        'μ*': mu_star_values,
        'σ': sigma_values,
        'Recommendation': recommendations
    })

    st.write("Morris Sensitivity Analysis Results:")
    st.dataframe(morris_df.style.apply(
        lambda row: ['background-color: lightgreen' if row['Recommendation'] == 'Recommended' else '' for _ in row],
        axis=1
    ))

    st.plotly_chart(morris_plot, use_container_width=True)

        # Explain the threshold lines
    st.markdown("""
    **Threshold Lines Explanation:**

    - The **vertical dashed line** represents the threshold for μ* (Mean of Absolute Elementary Effects), calculated as the 25th percentile of μ* values.
    - The **horizontal dashed line** represents the threshold for σ (Standard Deviation of EE), calculated as the 25th percentile of σ values.
    - Variables plotted **below** both thresholds are considered **non-influential** and are recommended for fixing.

    **How Recommendations are Made:**

    - Variables with both μ* and σ less than their respective thresholds (lower quartiles) are deemed non-influential.
    - Fixing non-influential variables can simplify the model without significantly affecting output variability.
    """)

    # Variable selection
    variables_to_fix = st.multiselect(
        "Variables to fix:",
        options=names,
        default=st.session_state.variables_to_fix,
        help="Variables recommended for fixing are marked in the table."
    )
    st.session_state.variables_to_fix = variables_to_fix

    # Assign constant values
    if variables_to_fix:
        indices_to_fix = [names.index(var) for var in variables_to_fix]
        constant_values = {names[i]: get_constant_value(problem['distributions'][i]) for i in indices_to_fix}
        st.session_state.constant_values = constant_values

        st.write("Fixed Variables and Assigned Constant Values:")
        fixed_vars_df = pd.DataFrame({
            'Variable': list(constant_values.keys()),
            'Assigned Value': list(constant_values.values())
        })

        edited_df = st.data_editor(
            fixed_vars_df,
            column_config={'Variable': st.column_config.Column(disabled=True)},
            key="fixed_vars_editor"
        )
        st.session_state.constant_values = dict(zip(edited_df['Variable'], edited_df['Assigned Value']))

        # Generate reduced model
        if st.button("Generate Reduced Model Definition"):
            try:

                prompt = (
                    f"Here is the original model:\n\n{code}\n\n"
                    f"The user has selected the following variables to fix:\n{constant_values}\n\n"
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

                with st.spinner("Generating reduced model..."):
                    st.session_state.reduced_model_code = call_groq_api(prompt, model_name='llama-3.1-70b-versatile').replace("```python", "").replace("```", "")
            except Exception as e:
                st.error(f"Error during AI generation: {e}")

if st.session_state.reduced_model_code:
    st.markdown("### Reduced Model Definition (Copy-Paste into the app!):")
    st.code(st.session_state.reduced_model_code, language='python')
