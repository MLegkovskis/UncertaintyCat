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

st.set_page_config(layout="wide")  # Use wide layout

# Create two columns
col1, col2 = st.columns([1, 4])  # Adjust the ratio to control column widths

with col1:
    st.image("logo.jpg", width=100)

with col2:
    st.title('UncertaintyCat | v3.5')

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
        - Run the analysis to obtain a reduced set of input variables.
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
        - Uniform, Normal, LogNormal, Beta, Gumbel, Triangular, etc.
        - Specify distributions in the `problem` dictionary with appropriate parameters.

    - **Imports:**
        - You can import external packages (e.g., `numpy`, `scipy`) within your model code.
        - Ensure that any packages you import are available in the app's environment.
    """)

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
        # Add any other keys you've used
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def generate_prompt(code_snippet):
    return f"""
Translate the following NumPy-based Python function into a markdown document that explains the mathematical model using LaTeX for equations. 

**Important:** For large expressions, such as lengthy polynomials that will always contain 'Y = (', include only the first three terms and the last two terms in the LaTeX equations. Represent the omitted middle terms with an ellipsis (\\dots) to indicate that the expression continues. Ensure the equations remain clear and readable.

Python code:
```python
{code_snippet}
```

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
    try:
        tree = ast.parse(code_str)
        UnsafeNodeVisitor().visit(tree)
        return True
    except Exception as e:
        raise ValueError(f"Unsafe code detected: {e}")

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
    'FloodModel.py', 'Ishigami.py', 'Material_Stress.py', 'Morris_Function.py',
    'Portfolio_Risk.py', 'Rocket_Trajectory.py', 'Solar_Panel_Output.py',
    'Truss_Model.py', 'Undamped_Oscillator.py', 'Wind_Turbine_Power.py'
]

model_file = st.selectbox(
    'Select a Model File or Enter your own Model:',
    model_options,
    index=model_options.index(st.session_state.model_file)
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
theme = 'xcode'  # You can set this to 'github', 'monokai', 'xcode', etc.
height = 650  # Set the height in pixels

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
            # Get the markdown interpretation
            markdown_output = get_markdown_from_code(
                st.session_state.code_editor,
                selected_language_model  # Ensure this matches the model used
            )
            st.session_state.markdown_output = markdown_output
            st.session_state.code_updated = False  # Reset the flag
    if st.session_state.markdown_output:
        st.markdown(st.session_state.markdown_output)
    else:
        st.info("The markdown interpretation will appear here.")

# Function to run simulation
def run_simulation():
    code = st.session_state.code_editor

    # Check code safety before execution
    try:
        check_code_safety(code)
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
        st.session_state.run_simulation = False  # Reset the flag
        st.stop()

    # Execute the code
    try:
        # Create globals with access to all modules
        globals_dict = {}

        # Execute the user's code
        exec(code, globals_dict)
        model = globals_dict.get('model')
        problem = globals_dict.get('problem')
        if model is None or problem is None:
            st.error("Model or problem definition not found in the code. Please ensure they are defined.")
            st.session_state.run_simulation = False  # Reset the flag
            st.stop()
    except Exception as e:
        st.error(f"Error in executing model code: {e}")
        st.session_state.run_simulation = False  # Reset the flag
        st.stop()

    # PCE is not used
    is_pce_used = False
    original_model_code = code
    metamodel_str = None

    # Hardcoded sample sizes
    N = 2000          # Sample Size for Monte Carlo Simulation
    N_samples = 8000  # Expectation Convergence Samples
    N_sobol = 1024    # Sobol Sensitivity Samples (must be power of 2)

    # Run Monte Carlo Simulation
    with st.spinner('Running Monte Carlo Simulation...'):
        data = monte_carlo_simulation(N, model, problem)

    # Store the results in st.session_state
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
        'analysis_options': analysis_options  # Store the selected analyses
    }

# Run the simulation when the button is clicked
if run_button:
    st.session_state.run_simulation = True
    reset_analysis_results()  # Reset analyses results

# If run_simulation is True, run the simulation
if st.session_state.run_simulation:
    run_simulation()
    st.session_state.run_simulation = False  # Reset the flag after running simulation

# If simulation results exist, display analyses
if st.session_state.simulation_results is not None:
    results = st.session_state.simulation_results
    data = results['data']
    model = results['model']
    problem = results['problem']
    code = results['code']
    is_pce_used = results['is_pce_used']
    original_model_code = results['original_model_code']
    metamodel_str = results['metamodel_str']
    selected_language_model = results['selected_language_model']
    N = results['N']
    N_samples = results['N_samples']
    N_sobol = results['N_sobol']
    analysis_options = results['analysis_options']

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
else:
    st.info("Please run the simulation to proceed.")
