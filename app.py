import streamlit as st
from streamlit_ace import st_ace
import os
import ast
import plotly.express as px
from modules.monte_carlo import monte_carlo_simulation
from modules.model_understanding import model_understanding
from modules.exploratory_data_analysis import exploratory_data_analysis
from modules.expectation_convergence_analysis import expectation_convergence_analysis
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis
from modules.taylor_analysis import taylor_analysis
from modules.correlation_analysis import correlation_analysis
from modules.hsic_analysis import hsic_analysis
from modules.ml_analysis import ml_analysis
from groq import Groq
import numpy as np

st.set_page_config(layout="wide")  # Use wide layout

# Create two columns
col1, col2 = st.columns([1, 4])  # Adjust the ratio to control column widths

with col1:
    st.image("logo.jpg", width=100)

with col2:
    st.title('UncertaintyCat | v3.3')

# Initialize session state
if 'model_file' not in st.session_state:
    st.session_state.model_file = 'Beam.py'
if 'code_editor' not in st.session_state:
    # Load default model code when the app first loads
    st.session_state.code_editor = open('examples/' + st.session_state.model_file).read()
if 'code_editor_counter' not in st.session_state:
    st.session_state.code_editor_counter = 0
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'run_simulation' not in st.session_state:
    st.session_state.run_simulation = False  # Initialize the flag
if 'code_updated' not in st.session_state:
    st.session_state.code_updated = False
if 'markdown_output' not in st.session_state:
    st.session_state.markdown_output = None

# Function to reset analysis results
def reset_analysis_results():
    keys_to_reset = [
        'correlation_response_markdown', 'correlation_fig',
        'expectation_response_markdown', 'expectation_fig',
        'eda_response_markdown', 'eda_fig',
        'hsic_response_markdown', 'hsic_fig',
        'ml_response_markdown', 'ml_shap_summary_fig', 'ml_dependence_fig',
        'model_understanding_response_markdown',
        'sobol_response_markdown', 'sobol_fig',
        'taylor_response_markdown', 'taylor_fig',
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def generate_prompt(code_snippet):
    return f"""
Translate the following NumPy-based Python function into a markdown document that explains the mathematical model using LaTeX for equations. 

**Important:** For large expressions, such as lengthy polynomials that will always contain 'Y = (', include only the first three terms and the last two terms in the LaTeX equations. Represent the omitted middle terms with an ellipsis (\\dots) to indicate that the expression continues. Ensure the equations remain clear and readable.

Python code:
python
{code_snippet}


Expected markdown output:
- Present the mathematical equations using LaTeX, following the guidelines for large expressions.
- Define each variable used in the equation.
- Tabulate the associated input uncertainties and their characteristics as detailed in the 'problem' dictionary.

Provide the output in pure markdown without additional explanations.
"""

# Function to get markdown from code using Groq API
def get_markdown_from_code(code_snippet, model_name):
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        prompt = generate_prompt(code_snippet)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name
        )
        response_text = chat_completion.choices[0].message.content
        return response_text
    except Exception as e:
        st.error(f"Error generating markdown interpretation: {e}")
        return None

# Function to check code safety
class UnsafeNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        # List of disallowed module names
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
        # Check for dangerous built-in functions
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', '__import__']:
                raise ValueError(
                    f"Use of function '{node.func.id}' is not allowed for security reasons."
                )
        # Check for dangerous methods
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
    tree = ast.parse(code_str)
    UnsafeNodeVisitor().visit(tree)
    return True

# Function to load code
def load_model_code(selected_model):
    try:
        with open('examples/' + selected_model, 'r') as f:
            code = f.read()
        return code
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ''

# Dropdown to select model file
model_options = [
    'Beam.py', 'Bike_Speed.py', 'Borehole_Model.py', 'Chemical_Reactor.py',
    'Cylinder_heating.py', 'Damped_Oscillator.py', 'Epidemic_Model.py',
    'FloodModel.py', 'Ishigami.py', 'Material_Stress.py', 'Portfolio_Risk.py',
    'Rocket_Trajectory.py', 'Solar_Panel_Output.py',
    'Undamped_Oscillator.py', 'Wind_Turbine_Power.py'
]

model_file = st.selectbox(
    'Select a Model File or Enter your own Model:',
    model_options,
    index=model_options.index(st.session_state.model_file) if st.session_state.model_file in model_options else 0
)

# If the model selection has changed, update the code_editor
if model_file != st.session_state.model_file:
    st.session_state.model_file = model_file
    st.session_state.code_editor = load_model_code(model_file)
    st.session_state.code_editor_counter += 1
    st.session_state.run_simulation = False  # Reset the flag when model changes
    st.session_state.simulation_results = None  # Clear previous results
    st.session_state.markdown_output = None  # Reset markdown output
    reset_analysis_results()  # Reset analyses results

# Dropdown for selecting Groq model
groq_model_options = [
    'gemma2-9b-it',
    'llama-3.1-70b-versatile',
    'mixtral-8x7b-32768'
]

selected_language_model = st.selectbox(
    'Select Language Model:',
    options=groq_model_options,
    index=0
)

# --- Analysis Options ---
st.markdown("### Select Analyses to Run")
analysis_options = {
    "Sobol Sensitivity Analysis": True,
    "Taylor Analysis": True,
    "Correlation Analysis": True,
    "HSIC Analysis": True,
    "SHAP Analysis": True
}

for analysis in analysis_options.keys():
    analysis_options[analysis] = st.checkbox(
        analysis, value=True
    )

# Run Simulation Button
run_button = st.button('Run Simulation')

# --- Code Editor and Markdown Rendering ---

# Set code editor options
language = 'python'
theme = 'xcode'
height = 650

# Use a dynamic key to force re-instantiation when model_file changes or code is updated
code_editor_key = f'code_editor_{st.session_state.code_editor_counter}'

# Split the page into two columns at the top
col_code, col_md = st.columns(2)

with col_code:
    st.markdown("### Model Code Editor")

    # Store previous code in session state
    previous_code = st.session_state.get('previous_code', '')

    # Display the code editor with syntax highlighting
    code = st_ace(
        value=st.session_state.code_editor,
        language=language,
        theme=theme,
        key=code_editor_key,
        height=height
    )
    st.session_state.code_editor = code

    # Check if the code has changed
    if code != previous_code:
        st.session_state.code_updated = True
        st.session_state.previous_code = code  # Update the previous code
        # Reset flags and results when code changes
        st.session_state.run_simulation = False
        st.session_state.simulation_results = None
        st.session_state.markdown_output = None
        reset_analysis_results()  # Reset analyses results
    else:
        st.session_state.code_updated = False

with col_md:
    st.markdown("### Model Interpretation")
    if st.session_state.code_editor and (
        st.session_state.code_updated or st.session_state.markdown_output is None
    ):
        with st.spinner("Generating markdown interpretation..."):
            markdown_output = get_markdown_from_code(
                st.session_state.code_editor,
                selected_language_model
            )
            st.session_state.markdown_output = markdown_output
            st.session_state.code_updated = False  # Reset the flag
    if st.session_state.markdown_output:
        st.markdown(st.session_state.markdown_output)
    else:
        st.info("The markdown interpretation will appear here.")

# Validation functions
expected_params = {
    'Uniform': 2,
    'Normal': 2,
    'LogNormalMuSigma': 3,
    'LogNormal': 3,
    'Beta': 4,
    'Gumbel': 2,
    'Triangular': 3
}

def validate_problem_structure(problem):
    required_keys = ['num_vars', 'names', 'distributions']
    for key in required_keys:
        if key not in problem:
            raise ValueError(f"Missing required key '{key}' in problem definition.")

    if not isinstance(problem['num_vars'], int):
        raise ValueError("problem['num_vars'] must be an integer.")

    if not isinstance(problem['names'], list):
        raise ValueError("problem['names'] must be a list.")

    if len(problem['names']) != problem['num_vars']:
        raise ValueError("Length of problem['names'] must match problem['num_vars'].")

    if not isinstance(problem['distributions'], list):
        raise ValueError("problem['distributions'] must be a list.")

    if len(problem['distributions']) != problem['num_vars']:
        raise ValueError("Length of problem['distributions'] must match problem['num_vars'].")

    distribution_expected_structures = {
        'Uniform':  "{'type': 'Uniform', 'params': [lower_bound, upper_bound]}",
        'Normal': "{'type': 'Normal', 'params': [mean, standard_deviation]}",
        'LogNormalMuSigma': "{'type': 'LogNormalMuSigma', 'params': [mu, sigma, gamma]}",
        'LogNormal': "{'type': 'LogNormal', 'params': [mu_log, sigma_log, gamma]}",
        'Beta': "{'type': 'Beta', 'params': [alpha, beta, lower_bound, upper_bound]}",
        'Gumbel': "{'type': 'Gumbel', 'params': [beta_param, gamma_param]}",
        'Triangular': "{'type': 'Triangular', 'params': [a, mode, b]}"
    }

    for dist_info in problem['distributions']:
        if not isinstance(dist_info, dict):
            raise ValueError("Each distribution entry must be a dictionary.")
        if 'type' not in dist_info or 'params' not in dist_info:
            raise ValueError("Each distribution must have 'type' and 'params' keys.")
        dist_type = dist_info['type']
        params = dist_info['params']
        if dist_type not in expected_params:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        if not isinstance(params, list):
            raise ValueError(f"params for distribution '{dist_type}' must be a list.")
        expected_count = expected_params[dist_type]
        if len(params) != expected_count:
            expected_structure = distribution_expected_structures[dist_type]
            raise ValueError(
                f"Incorrect number of parameters for '{dist_type}' distribution. "
                f"Expected {expected_count}, got {len(params)}. The expected structure is:\n\n"
                f"{expected_structure}"
        )

# Function to generate human-friendly error explanation using Groq
def get_human_friendly_error_explanation(code_snippet, error_message, model_name):
    examples = """
    Examples of helpful explanations and corrections:

    - "It seems you've specified an unsupported or improperly configured distribution. For example, if you're using a `Beta` distribution, you must provide four parameters `[alpha, beta, lower_bound, upper_bound]`. Using something like `{'type': 'Beta', 'params': [2.5, 4]}` is invalid because it's missing the lower and upper bounds. A correct Beta distribution example would be:
    ```python
    {'type': 'Beta', 'params': [2.5, 4, 310., 450.]}
    ```

    - If you tried to use a `Triangular` distribution but provided only two parameters (e.g., `{'type': 'Triangular', 'params': [55, 55.5]}`), you need three parameters `[a, mode, b]`. For example:
    ```python
    {'type': 'Triangular', 'params': [55, 55.5, 56]}
    ```
    
    - "It seems you've specified an unsupported distribution (e.g., 'BetaX'). Supported distributions are: Uniform, Normal, LogNormalMuSigma, LogNormal, Beta, Gumbel, and Triangular. Please use one of these. For instance, if you intended to use 'Beta', change 'BetaX' to 'Beta'.

    UncertaintyCat supports the following distributions and requires them to be correctly specified:

    1. **Uniform**
    ```python
    {'type': 'Uniform', 'params': [lower_bound, upper_bound]}
    ```

    2. **Normal**
    ```python
    {'type': 'Normal', 'params': [mean, standard_deviation]}
    ```

    3. **LogNormalMuSigma**
    ```python
    {'type': 'LogNormalMuSigma', 'params': [mu, sigma, gamma]}
    ```
    - `mu`: Mean of the LogNormal variable.
    - `sigma`: Standard deviation.
    - `gamma`: Location parameter (default is `0` if not specified).

    4. **LogNormal**
    ```python
    {'type': 'LogNormal', 'params': [mu_log, sigma_log, gamma]}
    ```

    5. **Beta**
    ```python
    {'type': 'Beta', 'params': [alpha, beta, lower_bound, upper_bound]}
    ```

    6. **Gumbel**
    ```python
    {'type': 'Gumbel', 'params': [beta_param, gamma_param]}
    ```

    7. **Triangular**
    ```python
    {'type': 'Triangular', 'params': [a, mode, b]}
    ```

    - Make sure that the number of input variables you unpack in your model function matches `problem['num_vars']` and `problem['names']`. For example, if:
    ```python
    problem = {
        'num_vars': 4,
        'names': ['E', 'F', 'L', 'I'],
        ...
    }
    ```
    Your model function must do:
    ```python
    E, F, L, I = X
    ```
    not fewer or more.

    - Ensure your model function returns a list or NumPy array. For example, if your function calculates `Y` and currently just does `return Y`, change it to `return [Y]`.

    Below are three valid model definitions from the `examples` folder that you can reference as starting points:

    **Valid Model Example 1 (from Beam.py)**:
    ```python
    def function_of_interest(X):
        E, F, L, I = X
        Y = F * L**3 / (3 * E * I)
        return [Y]

    problem = {
        'num_vars': 4,
        'names': ['E', 'F', 'L', 'I'],
        'distributions': [
            {'type': 'Beta', 'params': [0.9, 3.1, 2.8e7, 4.8e7]},  # E
            {'type': 'LogNormalMuSigma', 'params': [3.0e4, 9.0e3, 15.0e3]}, # F
            {'type': 'Uniform', 'params': [250., 260.]},             # L
            {'type': 'Beta', 'params': [2.5, 4, 310., 450.]}         # I
        ]
    }

    model = function_of_interest
    ```

    **Valid Model Example 2 (from Ishigami.py)**:
    ```python
    import numpy as np

    def function_of_interest(X):
        x1, x2, x3 = X
        Y = np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)
        return [Y]

    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'distributions': [
            {'type': 'Uniform', 'params': [-np.pi, np.pi]}, # x1
            {'type': 'Uniform', 'params': [-np.pi, np.pi]}, # x2
            {'type': 'Uniform', 'params': [-np.pi, np.pi]}  # x3
        ]
    }

    model = function_of_interest
    ```

    **Valid Model Example 3 (from Truss_Model.py)**:
    ```python
    import numpy as np

    def function_of_interest(X):
        E1, E2, A1, A2, P1, P2, P3, P4, P5, P6 = X
        # Convert units and compute Y as shown in the Truss_Model.py
        # Ensure all parameters and distributions match the expected [type, params] format.
        # ...
        return [y]

    problem = {
        'num_vars': 10,
        'names': ['E1', 'E2', 'A1', 'A2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'distributions': [
            {'type': 'LogNormalMuSigma', 'params': [210, 21, 0]},   # E1
            {'type': 'LogNormalMuSigma', 'params': [210, 21, 0]},   # E2
            {'type': 'LogNormalMuSigma', 'params': [20, 2, 0]},     # A1
            {'type': 'LogNormalMuSigma', 'params': [10, 1, 0]},     # A2
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P1
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P2
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P3
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P4
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P5
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]}         # P6
        ]
    }

    model = function_of_interest
    ```

    These examples show the correct structure, supported distributions, proper parameter counts, and ensure that the number of variables and their names match `num_vars` and `names`.

    Also consult other models like `Borehole_Model.py`, `Material_Stress.py`, and `Portfolio_Risk.py` in the `examples` folder for different distributions and inputs. Ensuring all parameters are valid and consistent will fix the error and allow your simulation to run successfully.
    """

    prompt = f"""
    The user has provided a Python code snippet (a model) that should produce a numerical output from a NumPy array `X`, but it failed execution from a Python standpoint. Based on the instructions and examples below, please:

    1. Clearly explain what went wrong (e.g., unsupported distribution type, incorrect return type, missing parameters for a given distribution).
    2. Suggest how to fix it (e.g., use a supported distribution name, supply the correct number of parameters for that distribution, ensure the model returns a list, and ensure the number of variables matches `problem['num_vars']` and `problem['names']`).
    3. Provide at least two valid model definitions as direct examples (three are provided here).
    4. Refer the user to the `examples` folder (Beam.py, Borehole_Model.py, Ishigami.py, Material_Stress.py, Portfolio_Risk.py, Truss_Model.py) for further guidance.
    5. Never include phrases like "Let me know if you have any more questions" as this is a one-time interaction.

    Error Message:
    ```
    {error_message}
    ```

    User Code:
    ```
    {code_snippet}
    ```

    {examples}
    """
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name
        )
        response_text = chat_completion.choices[0].message.content
        return response_text
    except Exception as e:
        return f"Failed to generate human-friendly explanation: {e}"

def test_model(model, problem, code, language_model):
    from modules.statistical_utils import sample_inputs
    # Try a small sample
    try:
        test_samples = sample_inputs(5, problem)  # 5 samples
    except Exception as e:
        # If sample_inputs fails (e.g. unsupported distribution), show friendly error
        explanation = get_human_friendly_error_explanation(code, str(e), language_model)
        st.error("Model Input Sampling Error:")
        st.error(explanation)
        return False

    for X in test_samples:
        try:
            result = model(X)
            if not isinstance(result, (list, np.ndarray)) or len(result) == 0:
                raise ValueError("Model did not return a non-empty list/array.")
        except Exception as e:
            # Model failed during execution
            explanation = get_human_friendly_error_explanation(code, str(e), language_model)
            st.error("Model Execution Error:")
            st.error(explanation)
            return False
    return True

def run_simulation():
    code = st.session_state.code_editor

    # Check code safety before execution
    try:
        check_code_safety(code)
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
        st.session_state.run_simulation = False
        return

    # Execute the code
    try:
        globals_dict = {}
        exec(code, globals_dict)
        model = globals_dict.get('model')
        problem = globals_dict.get('problem')
        if model is None or problem is None:
            st.error("Model or problem definition not found in the code. Please ensure they are defined.")
            st.session_state.run_simulation = False
            return
    except Exception as e:
        explanation = get_human_friendly_error_explanation(code, str(e), selected_language_model)
        st.error("Model Load Error:")
        st.error(explanation)
        st.session_state.run_simulation = False
        return

    # Validate the problem structure before running the model test
    try:
        validate_problem_structure(problem)
    except ValueError as ve:
        st.error(f"Validation Error: {ve}")
        st.session_state.run_simulation = False
        return

    # Test the model with a small sample before doing full simulation
    if not test_model(model, problem, code, selected_language_model):
        st.session_state.run_simulation = False
        return

    # Set up parameters
    is_pce_used = False
    original_model_code = code
    metamodel_str = None

    # Hardcoded sample sizes
    N = 2000
    N_samples = 8000
    N_sobol = 1024

    # Run Monte Carlo Simulation with error handling
    try:
        with st.spinner('Running Monte Carlo Simulation...'):
            data = monte_carlo_simulation(N, model, problem)
    except Exception as e:
        explanation = get_human_friendly_error_explanation(code, str(e), selected_language_model)
        st.error("Simulation Error:")
        st.error(explanation)
        st.session_state.run_simulation = False
        return

    # If no errors, store results and proceed
    st.session_state.simulation_results = {
        'data': data,
        'model': model,
        'problem': problem,
        'code': code,
        'is_pce_used': is_pce_used,
        'original_model_code': original_model_code,
        'metamodel_str': metamodel_str,
        'selected_language_model': selected_language_model,
        'N': N,
        'N_samples': N_samples,
        'N_sobol': N_sobol,
        'analysis_options': analysis_options
    }

    # Present each module one after another
    st.markdown("---")
    st.header("Model Understanding")
    with st.spinner('Running Model Understanding...'):
        model_understanding(
            model,
            problem,
            code,
            is_pce_used=is_pce_used,
            original_model_code_str=original_model_code,
            metamodel_str=metamodel_str,
            language_model=selected_language_model
        )

    st.markdown("---")
    st.header("Exploratory Data Analysis")
    with st.spinner('Running Exploratory Data Analysis...'):
        exploratory_data_analysis(
            data, N, model, problem, code, language_model=selected_language_model
        )

    st.markdown("---")
    st.header("Expectation Convergence Analysis")
    with st.spinner('Running Expectation Convergence Analysis...'):
        expectation_convergence_analysis(
            model, problem, code, N_samples=N_samples,
            language_model=selected_language_model
        )

    if analysis_options["Sobol Sensitivity Analysis"]:
        st.markdown("---")
        st.header("Sobol Sensitivity Analysis")
        with st.spinner('Running Sobol Sensitivity Analysis...'):
            sobol_sensitivity_analysis(
                N_sobol, model, problem, code, language_model=selected_language_model
            )

    if analysis_options["Taylor Analysis"]:
        st.markdown("---")
        st.header("Taylor Analysis")
        with st.spinner('Running Taylor Analysis...'):
            taylor_analysis(
                model, problem, code, language_model=selected_language_model
            )

    if analysis_options["Correlation Analysis"]:
        st.markdown("---")
        st.header("Correlation Analysis")
        with st.spinner('Running Correlation Analysis...'):
            correlation_analysis(
                model, problem, code, language_model=selected_language_model
            )

    if analysis_options["HSIC Analysis"]:
        st.markdown("---")
        st.header("HSIC Analysis")
        with st.spinner('Running HSIC Analysis...'):
            hsic_analysis(
                model, problem, code, language_model=selected_language_model
            )

    if analysis_options["SHAP Analysis"]:
        st.markdown("---")
        st.header("SHAP Analysis")
        with st.spinner('Running SHAP Analysis...'):
            ml_analysis(
                data, problem, code, language_model=selected_language_model
            )

# If Run Simulation button is clicked
if run_button:
    st.session_state.run_simulation = True
    reset_analysis_results()

# If run_simulation is True, run the simulation
if st.session_state.run_simulation:
    run_simulation()
    st.session_state.run_simulation = False  # Reset the flag after running simulation

if st.session_state.simulation_results is None:
    st.info("Please run the simulation to proceed.")
