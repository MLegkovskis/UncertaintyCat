# pages/dimensionality_reduction.py

import streamlit as st
from streamlit_ace import st_ace
import ast
import pandas as pd
from modules.morris_sensitivity_analysis import run_morris_analysis_for_dimensionality_reduction
from modules.model_utils import generate_reduced_model_function
from modules.statistical_utils import get_constant_value

st.set_page_config(layout="wide")

st.title('Dimensionality Reduction using Morris Analysis')

with st.expander("Instructions"):
    st.markdown("""
    **Dimensionality Reduction Page**

    This page allows you to perform dimensionality reduction on your model using Morris Sensitivity Analysis.

    **Instructions:**
    - Select a model from the dropdown or input your own in the code editor.
    - Run the Morris Sensitivity Analysis.
    - Choose variables to fix and generate the reduced model.
    - Copy the reduced model code and paste it into the main app's code editor.
    """)

# Initialize session state variables
if 'model_file' not in st.session_state:
    st.session_state.model_file = 'Beam.py'
if 'code_editor' not in st.session_state:
    # Load default model code when the app first loads
    st.session_state.code_editor = open('examples/' + st.session_state.model_file).read()
if 'code_editor_counter' not in st.session_state:
    st.session_state.code_editor_counter = 0
if 'morris_analysis_done' not in st.session_state:
    st.session_state.morris_analysis_done = False
if 'variables_to_fix' not in st.session_state:
    st.session_state.variables_to_fix = []
if 'constant_values' not in st.session_state:
    st.session_state.constant_values = {}
if 'morris_plot' not in st.session_state:
    st.session_state.morris_plot = None
if 'reduced_model_code' not in st.session_state:
    st.session_state.reduced_model_code = ''
if 'morris_results' not in st.session_state:
    st.session_state.morris_results = {}

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
    st.session_state.morris_analysis_done = False  # Reset
    st.session_state.variables_to_fix = []
    st.session_state.constant_values = {}
    st.session_state.morris_plot = None
    st.session_state.reduced_model_code = ''
    st.session_state.morris_results = {}

# Set code editor options
language = 'python'
theme = 'xcode'  # You can set this to 'github', 'monokai', 'xcode', etc.
height = 650  # Set the height in pixels

# Use a dynamic key to force re-instantiation when model_file changes or code is updated
code_editor_key = f'code_editor_{st.session_state.code_editor_counter}'

# Display the code editor
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
    st.session_state.previous_code = code  # Update the previous code
    st.session_state.morris_analysis_done = False  # Reset Morris analysis
    st.session_state.variables_to_fix = []
    st.session_state.constant_values = {}
    st.session_state.morris_plot = None
    st.session_state.reduced_model_code = ''
    st.session_state.morris_results = {}

# Run Morris Analysis Button
run_morris_button = st.button('Run Morris Sensitivity Analysis')

if run_morris_button:
    # Check code safety before execution
    try:
        check_code_safety(code)
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
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
            st.stop()
    except Exception as e:
        st.error(f"Error in executing model code: {e}")
        st.stop()

    # Perform Morris sensitivity analysis
    with st.spinner('Running Morris Sensitivity Analysis...'):
        N_morris = 100  # Sample size for Morris analysis
        (
            non_influential_indices, mu_star_values,
            sigma_values, fig
        ) = run_morris_analysis_for_dimensionality_reduction(
            N_morris, model, problem
        )

        # Store results in session state
        st.session_state.morris_analysis_done = True
        st.session_state.non_influential_indices = non_influential_indices
        st.session_state.mu_star_values = mu_star_values
        st.session_state.sigma_values = sigma_values
        st.session_state.problem = problem
        st.session_state.model = model
        st.session_state.original_model_code = code
        st.session_state.N_morris = N_morris
        st.session_state.morris_plot = fig  # Store the plot

        # Store the results DataFrame
        st.session_state.morris_results = {
            'non_influential_indices': non_influential_indices,
            'mu_star_values': mu_star_values,
            'sigma_values': sigma_values
        }

        # Explain the threshold lines
        st.markdown("""
        **Threshold Lines Explanation:**

        - The **vertical dashed line** represents the threshold for μ* (Mean of Absolute Elementary Effects).
        - The **horizontal dashed line** represents the threshold for σ (Standard Deviation of EE).
        - Variables plotted **below** both thresholds are considered **non-influential** and are recommended for fixing.

        **How Recommendations are Made:**

        - Variables with both μ* and σ less than their respective thresholds are deemed non-influential.
        - Fixing non-influential variables can simplify the model without significantly affecting output variability.
        """)

# After running Morris analysis
if st.session_state.morris_analysis_done:
    problem = st.session_state.problem
    model = st.session_state.model
    names = problem['names']
    mu_star_values = st.session_state.mu_star_values
    sigma_values = st.session_state.sigma_values
    non_influential_indices = st.session_state.non_influential_indices

    # Create a DataFrame for display
    recommendations = [
        'Recommended' if i in non_influential_indices else ''
        for i in range(len(names))
    ]
    morris_df = pd.DataFrame({
        'Parameter': names,
        'μ*': mu_star_values,
        'σ': sigma_values,
        'Recommendation': recommendations
    })

    st.write("Morris Sensitivity Analysis Results:")

    # Define the styling function
    def highlight_recommended(row):
        return [
            'background-color: lightgreen' if row['Recommendation'] == 'Recommended' else ''
            for _ in row
        ]

    # Apply the styling to the DataFrame
    st.dataframe(morris_df.style.apply(highlight_recommended, axis=1))

    # Display the Morris plot
    st.plotly_chart(
        st.session_state.morris_plot,
        use_container_width=True,
        key='morris_plot_display'
    )

    st.write("Select variables to fix (non-influential variables are recommended):")
    variables_to_fix = st.multiselect(
        "Variables to fix:",
        options=names,
        default=st.session_state.variables_to_fix,  # Use session state to persist selections
        help="Variables recommended for fixing are marked in the table."
    )

    st.session_state.variables_to_fix = variables_to_fix

    # Display the fixed variables and their assigned constant values
    if variables_to_fix:
        indices_to_fix = [names.index(var) for var in variables_to_fix]

        constant_values = {}
        distributions = {}
        for i in indices_to_fix:
            var_name = problem['names'][i]
            dist = problem['distributions'][i]
            default_value = get_constant_value(dist)
            constant_values[var_name] = default_value
            dist_type = dist['type']
            dist_params = dist['params']
            dist_str = f"{dist_type}({', '.join(map(str, dist_params))})"
            distributions[var_name] = dist_str

        st.write("Fixed Variables and Assigned Constant Values:")
        fixed_vars_df = pd.DataFrame({
            'Variable': list(constant_values.keys()),
            'Original Distribution': [distributions[var] for var in constant_values.keys()],
            'Assigned Value': list(constant_values.values())
        })

        # Make the 'Assigned Value' column editable
        edited_df = st.data_editor(
            fixed_vars_df,
            column_config={
                'Variable': st.column_config.Column(disabled=True),
                'Original Distribution': st.column_config.Column(disabled=True),
                'Assigned Value': st.column_config.NumberColumn()
            },
            hide_index=True,
            num_rows="dynamic",
            key="fixed_vars_editor"
        )

        # Update constant_values with the edited values
        constant_values = dict(zip(edited_df['Variable'], edited_df['Assigned Value']))
        st.session_state.constant_values = constant_values  # Store in session state

    # Generate Reduced Model Button
    generate_model_button = st.button("Generate Reduced Model")

    if generate_model_button:
        if not variables_to_fix:
            st.warning("Please select at least one variable to fix.")
        else:
            indices_to_fix = [names.index(var) for var in variables_to_fix]
            indices_to_keep = [
                i for i in range(len(names)) if names[i] not in variables_to_fix
            ]

            # Use the updated constant_values from session state
            constant_values = st.session_state.constant_values

            try:
                reduced_model_code = generate_reduced_model_function(
                    model,
                    constant_values,
                    indices_to_keep,
                    problem,
                    st.session_state.original_model_code
                )
                st.session_state.reduced_model_code = reduced_model_code  # Store in session state
            except Exception as e:
                st.error(f"Error generating reduced model: {e}")
                st.stop()

            st.success(
                "Dimensionality reduction applied. Please copy the reduced model code below and paste it into the main app's code editor."
            )

# Display the Reduced Model Code if it exists in session state
if st.session_state.get('reduced_model_code', ''):
    st.write("### Reduced Model Definition (Copy-Paste into the app!):")
    st.code(st.session_state.reduced_model_code, language='python')
