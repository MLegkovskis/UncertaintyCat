import streamlit as st
from streamlit_ace import st_ace

from modules.monte_carlo import monte_carlo_simulation
from modules.model_understanding import model_understanding
from modules.exploratory_data_analysis import exploratory_data_analysis
from modules.expectation_convergence_analysis import expectation_convergence_analysis
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis
from modules.pce_sobol import pce_sobol
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

st.set_page_config(layout="wide")  # Use wide layout

# Create two columns
col1, col2 = st.columns([1, 4])  # Adjust the ratio to control column widths

with col1:
    st.image("logo.jpg", width=100)

with col2:
    st.title('UncertaintyCat | v3.89')

# Display the instructions
show_instructions()

# Initialize session state using the function from the module
initialize_session_state()

# Function to load code
def load_model_code(selected_model):
    try:
        with open('examples/' + selected_model, 'r') as f:
            code = f.read()
        return code
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ''

model_file = st.selectbox(
    'Select a Model File or Enter your own Model:',
    model_options,
    index=model_options.index(get_session_state('model_file'))
)

# --- Code Editor and Markdown Rendering ---

# Set code editor options
language = 'python'
theme = 'xcode'  # You can set this to 'github', 'monokai', 'xcode', etc.
height = 400  # Set the height in pixels

# Use a dynamic key to force re-instantiation when model_file changes or code is updated
code_editor_key = f'code_editor_{get_session_state("code_editor_counter")}'

# Split the page into two columns at the top
col_code, col_md = st.columns(2)

with col_code:
    st.markdown("### Model Code Editor")

    # Store previous code in session state
    previous_code = get_session_state('previous_code', '')

    # Display the code editor with syntax highlighting
    code = st_ace(
        value=get_session_state('code_editor'),
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
        get_session_state('code_updated') or get_session_state('markdown_output') is None
    ):
        with st.spinner("Generating markdown interpretation..."):
            # Get the markdown interpretation
            markdown_output = get_markdown_from_code(
                st.session_state.code_editor,
                'gemma2-9b-it'  # Ensure this matches the model used
            )
            st.session_state.markdown_output = markdown_output
            st.session_state.code_updated = False  # Reset the flag
    if get_session_state('markdown_output'):
        st.markdown(get_session_state('markdown_output'))
    else:
        st.info("The markdown interpretation will appear here.")

# If the model selection has changed, update the code_editor
if model_file != get_session_state('model_file'):
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
    'llama-3.3-70b-versatile',
    'mixtral-8x7b-32768'
]

selected_language_model = st.selectbox(
    'Select Language Model:',
    options=groq_model_options,
    index=0
)

# --- Analysis Options ---
st.markdown("### Select Analyses to Run")
analysis_default_options = {
    "Exploratory Analysis": True,
    "Expectation Analysis": True,
    "Sobol Sensitivity Analysis": True,
    "Taylor Analysis": True,
    "Correlation Analysis": True,
    "HSIC Analysis": True,
    "SHAP Analysis": True,
    "Sobol from PCE": True
}

analysis_options = dict()
for analysis in analysis_default_options.keys():
    analysis_options[analysis] = st.checkbox(
        analysis, value=analysis_default_options[analysis] 
    )

run_button = st.button('Run Simulation')

# Function to run simulation
def run_simulation():
    code = get_session_state('code_editor')

    # Check code safety before execution
    try:
        check_code_safety(code)  # Use imported function
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
        st.session_state.run_simulation = False  # Reset the flag
        return

    # Execute the code
    try:
        globals_dict = {}
        exec(code, globals_dict)
        model = globals_dict.get('model')
        problem = globals_dict.get('problem')
        if model is None or problem is None:
            st.error("Model or problem definition not found in the code. Please ensure they are defined.")
            st.session_state.run_simulation = False  # Reset the flag
            return
    except Exception as e:
        explanation = get_human_friendly_error_explanation(code, str(e), selected_language_model)  # Use imported function
        st.error("Model Load Error:")
        st.error(explanation)
        st.session_state.run_simulation = False
        return

    # Validate the problem structure before running the model test
    try:
        validate_problem_structure(problem)  # Use imported function
    except ValueError as ve:
        st.error(f"Validation Error: {ve}")
        st.session_state.run_simulation = False
        return

    # Test the model with a small sample before full simulation
    if not test_model(model, problem, code, selected_language_model):  # Use imported function
        st.session_state.run_simulation = False
        return

    # PCE is not used
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
        explanation = get_human_friendly_error_explanation(code, str(e), selected_language_model)  # Use imported function
        st.error("Simulation Error:")
        st.error(explanation)
        st.session_state.run_simulation = False
        return

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
if get_session_state('run_simulation'):
    run_simulation()
    st.session_state.run_simulation = False  # Reset the flag after running simulation

# If simulation results exist, display analyses
if get_session_state('simulation_results') is not None:
    results = get_session_state('simulation_results')
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

    if analysis_options["Exploratory Analysis"]:
        st.markdown("---")
        st.header("Exploratory Data Analysis")
        with st.spinner('Running Exploratory Data Analysis...'):
            exploratory_data_analysis(
                data, N, model, problem, code, language_model=selected_language_model
            )
            # Access the figures from st.session_state:
            st.session_state['eda_fig'] = st.session_state['eda_fig']
            st.session_state['eda_clustermap_fig'] = st.session_state['eda_clustermap_fig']

    if analysis_options["Expectation Analysis"]:
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

    if analysis_options["Sobol from PCE"]:
        st.markdown("---")
        st.header("Sobol from PCE")
        with st.spinner('Running Sobol from PCE...'):
            pce_sobol(
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