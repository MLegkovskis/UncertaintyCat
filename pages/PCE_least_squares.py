import streamlit as st
from streamlit_ace import st_ace
from modules.code_safety import check_code_safety
import streamlit as st  # Added import statement
from modules.pce_least_squares_utils import pce_sobol, reset_pce_least_squares_results

st.set_page_config(layout="wide")

st.title("PCE Surrogate Model Generation using OpenTURNS")

with st.expander("Instructions"):
    st.markdown(
        """
    **Least Squares PCE Surrogate Page**

    This page allows you to generate a Polynomial Chaos Expansion (PCE) surrogate model for your original model using OpenTURNS.
    The coefficients are estimated using least squares.

    **Instructions:**
    - Select a model from the dropdown or input your own in the code editor.
    - Adjust the PCE parameters and advanced settings as needed to improve surrogate accuracy.
    - Run the PCE Surrogate Generation.
    - Review the validation plots to assess surrogate accuracy.
    """
    )

# Initialize session state
if "model_file" not in st.session_state:
    st.session_state.model_file = "Beam.py"
if "code_editor" not in st.session_state:
    # Load default model code when the app first loads
    with open("examples/" + st.session_state.model_file) as f:
        st.session_state.code_editor = f.read()
if "code_editor_counter" not in st.session_state:
    st.session_state.code_editor_counter = 0
if "pce_generated" not in st.session_state:
    st.session_state.pce_generated = False


# Function to load code
def load_model_code(selected_model):
    try:
        with open("examples/" + selected_model, "r") as f:
            code = f.read()
        return code
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""


# Dropdown to select model file
model_options = [
    "Beam.py",
    "Bike_Speed.py",
    "Borehole_Model.py",
    "Chaboche_Model.py",
    "Chemical_Reactor.py",
    "Cylinder_heating.py",
    "Damped_Oscillator.py",
    "Epidemic_Model.py",
    "FloodModel.py",
    "Ishigami.py",
    "Logistic_Model.py",
    "Material_Stress.py",
    "Morris_Function.py",
    "Portfolio_Risk.py",
    "Rocket_Trajectory.py",
    "Stiffened_Panel.py",
    "Solar_Panel_Output.py",
    "Truss_Model.py",
    "Tube_Deflection.py",
    "Undamped_Oscillator.py",
    "Viscous_Freefall.py",
    "Wind_Turbine_Power.py",
]

model_file = st.selectbox(
    "Select a Model File or Enter your own Model:",
    model_options,
    index=model_options.index(st.session_state.model_file),
)

# If the model selection has changed, update the code_editor
if model_file != st.session_state.model_file:
    st.session_state.model_file = model_file
    st.session_state.code_editor = load_model_code(model_file)
    st.session_state.code_editor_counter += 1
    reset_pce_least_squares_results()

# Set code editor options
code_editor_key = f"code_editor_{st.session_state.code_editor_counter}"

# Display the code editor
st.markdown("### Model Code Editor")

# Store previous code in session state
previous_code = st.session_state.get("previous_code", "")

code = st_ace(
    value=st.session_state.code_editor,
    language="python",
    theme="xcode",
    key=code_editor_key,
    height=400,
)
st.session_state.code_editor = code

# Check if the code has changed
if code != previous_code:
    st.session_state.previous_code = code
    reset_pce_least_squares_results()


# PCE Parameters
st.markdown("### PCE Parameters")
training_sample_size = st.number_input(
    "Training Sample Size:",
    min_value=100,
    max_value=10000,
    value=500,
    step=100,
    help="Sample size used to train the surrogate model.",
)

validation_sample_size = st.number_input(
    "Validation Sample Size:",
    min_value=100,
    max_value=10000,
    value=500,
    step=100,
    help="Sample size used to validate the surrogate model.",
)

use_model_selection = st.checkbox(
    "Use model selection:",
    value=False,
    help="Whether to use Least Angle Regression Stepwise (LARS).",
)

basis_size_factor = st.slider(
    "Basis Size Factor:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="The basis size is set equal to this fraction of the training sample size.",
)

# Dropdown for selecting Groq model
groq_model_options = ["gemma2-9b-it", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

selected_language_model = st.selectbox(
    "Select Language Model:", options=groq_model_options, index=0
)

# Run the analysis

# Run PCE Surrogate Generation Button
run_pce_button = st.button("Generate PCE Surrogate Model")

if run_pce_button:
    reset_pce_least_squares_results()
    # Check code safety before execution
    try:
        check_code_safety(code)
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
        st.stop()

    # Execute the code
    try:
        globals_dict = {}
        exec(code, globals_dict)
        model = globals_dict.get("model")
        problem = globals_dict.get("problem")
        if model is None or problem is None:
            st.error(
                "Model or problem definition not found in the code. Please ensure they are defined."
            )
            st.stop()
    except Exception as e:
        st.error(f"Error in executing model code: {e}")
        st.stop()

    with st.spinner("Generating PCE Surrogate Model..."):
        input_names = problem["names"]
        num_vars = problem["num_vars"]
        distributions = problem["distributions"]

        # Print results
        st.markdown("---")
        st.header("Polynomial Chaos Expansion using least squares")
        with st.spinner("Running PCE..."):
            pce_sobol(
                training_sample_size,
                validation_sample_size,
                model,
                problem,
                code,
                use_model_selection=use_model_selection,
                language_model=selected_language_model,
                basis_size_factor=basis_size_factor,
            )
            st.success("PCE Surrogate Model generated successfully.")
