# pages/pce_surrogate.py

import streamlit as st
from streamlit_ace import st_ace
import ast
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import time  # Import time module for timing model evaluations

st.set_page_config(layout="wide")

st.title('PCE Surrogate Model Generation using Chaospy')

with st.expander("Instructions"):
    st.markdown("""
    **PCE Surrogate Page**

    This page allows you to generate a Polynomial Chaos Expansion (PCE) surrogate model for your original model using Chaospy.

    **Instructions:**
    - Select a model from the dropdown or input your own in the code editor.
    - Adjust the PCE parameters and advanced settings as needed to improve surrogate accuracy.
    - Run the PCE Surrogate Generation.
    - Review the validation plots to assess surrogate accuracy.
    - Copy the surrogate model code and paste it into the main app's code editor.

    **Note:** If the estimated number of quadrature nodes exceeds 5,000, the app will advise you to reduce the dimensionality or adjust the quadrature settings.
    """)

# Initialize session state
if 'model_file' not in st.session_state:
    st.session_state.model_file = 'Beam.py'
if 'code_editor' not in st.session_state:
    # Load default model code when the app first loads
    with open('examples/' + st.session_state.model_file) as f:
        st.session_state.code_editor = f.read()
if 'code_editor_counter' not in st.session_state:
    st.session_state.code_editor_counter = 0
if 'pce_generated' not in st.session_state:
    st.session_state.pce_generated = False
if 'surrogate_model_code' not in st.session_state:
    st.session_state.surrogate_model_code = ''
if 'validation_plot' not in st.session_state:
    st.session_state.validation_plot = None
if 'r2' not in st.session_state:
    st.session_state.r2 = None
if 'mae' not in st.session_state:
    st.session_state.mae = None
if 'rmse' not in st.session_state:
    st.session_state.rmse = None
if 'normalized_rmse' not in st.session_state:
    st.session_state.normalized_rmse = None

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
    st.session_state.pce_generated = False
    st.session_state.surrogate_model_code = ''
    st.session_state.validation_plot = None

# Set code editor options
language = 'python'
theme = 'xcode'
height = 650

code_editor_key = f'code_editor_{st.session_state.code_editor_counter}'

# Display the code editor
st.markdown("### Model Code Editor")

# Store previous code in session state
previous_code = st.session_state.get('previous_code', '')

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
    st.session_state.previous_code = code
    st.session_state.pce_generated = False
    st.session_state.surrogate_model_code = ''
    st.session_state.validation_plot = None

# PCE Parameters
st.markdown("### PCE Parameters")
col1, col2 = st.columns(2)

with col1:
    K = st.number_input(
        "Order of Quadrature (K):",
        min_value=1, max_value=50, value=3, step=1,
        help="Order of the quadrature rule."
    )
    N = st.number_input(
        "Order of Polynomial Expansion (N):",
        min_value=1, max_value=50, value=3, step=1,
        help="Order of the polynomial expansion."
    )
    sparse = st.checkbox(
        "Use Sparse Grid:",
        value=False,
        help="Whether to use sparse grid quadrature."
    )
    growth = st.checkbox(
        "Use Growth Rule:",
        value=False,
        help="Use growth rule in quadrature generation."
    )

with col2:
    n_validation = st.number_input(
        "Number of Validation Samples:",
        min_value=100, max_value=10000, value=500, step=100,
        help="Number of samples used to validate the surrogate model."
    )
    cross_truncation = st.slider(
        "Cross Truncation:",
        min_value=0.0, max_value=1.0, value=1.0, step=0.05,
        help="Hyperbolic cross truncation parameter."
    )

# Advanced Settings
with st.expander("Advanced Settings"):
    quadrature_rules = {
        "Clenshaw-Curtis": "clenshaw_curtis",
        "Fejer's First Rule": "fejer_1",
        "Fejer's Second Rule": "fejer_2",
        "Gaussian": "gaussian",
        "Legendre": "legendre",
        "Lobatto": "lobatto",
        "Kronrod": "kronrod",
        "Patterson": "patterson",
        "Radau": "radau",
        "Leja": "leja",
        "Newton-Cotes": "newton_cotes",
        "Genz-Keister 16": "genz_keister_16",
        "Genz-Keister 18": "genz_keister_18",
        "Genz-Keister 22": "genz_keister_22",
        "Genz-Keister 24": "genz_keister_24",
    }

    quadrature_rule = st.selectbox(
        "Quadrature Rule:",
        list(quadrature_rules.keys()),
        index=list(quadrature_rules.values()).index("gaussian"),
        help="Quadrature rule used in generating nodes and weights."
    )

    orthogonal_methods = {
        "Three Terms Recurrence": "three_terms_recurrence",
        "Stieltjes": "stieltjes",
        "Cholesky": "cholesky",
        "Gram-Schmidt": "gram_schmidt",
    }

    orth_method = st.selectbox(
        "Orthogonalization Method:",
        list(orthogonal_methods.keys()),
        index=list(orthogonal_methods.values()).index("three_terms_recurrence"),
        help="Method used for generating orthogonal polynomials."
    )

    recurrence_algorithms = ["stieltjes", "chebyshev"]
    recurrence_algorithm = st.selectbox(
        "Recurrence Algorithm:",
        recurrence_algorithms,
        index=0,
        help="Algorithm used in quadrature generation for recurrence coefficients."
    )

    tolerance = st.number_input(
        "Tolerance:",
        min_value=1e-15, max_value=1e-5, value=1e-10, step=1e-10,
        format="%.1e",
        help="Tolerance for convergence in quadrature generation."
    )

    scaling = st.number_input(
        "Scaling:",
        min_value=0.0, max_value=10.0, value=3.0, step=0.1,
        help="Scaling factor for adaptive quadrature order increments."
    )

# Run PCE Surrogate Generation Button
run_pce_button = st.button('Generate PCE Surrogate Model')

if run_pce_button:
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
        model = globals_dict.get('model')
        problem = globals_dict.get('problem')
        if model is None or problem is None:
            st.error("Model or problem definition not found in the code. Please ensure they are defined.")
            st.stop()
    except Exception as e:
        st.error(f"Error in executing model code: {e}")
        st.stop()

    with st.spinner('Generating PCE Surrogate Model...'):
        try:
            input_names = problem['names']
            num_vars = problem['num_vars']
            distributions = problem['distributions']

            # Create distributions using Chaospy
            def create_distribution(dist_info):
                dist_type = dist_info['type']
                params = dist_info['params']
                if dist_type == 'Uniform':
                    a, b = params
                    return cp.Uniform(a, b)
                elif dist_type == 'Normal':
                    mu, sigma = params
                    return cp.Normal(mu, sigma)
                elif dist_type == 'Gumbel':
                    gamma_param, beta_param = params
                    return cp.Gumbel(loc=gamma_param, scale=beta_param)
                elif dist_type == 'Triangular':
                    a, m, b = params
                    if not (a <= m <= b):
                        raise ValueError(f"Triangular distribution requires a <= m <= b, but got a={a}, m={m}, b={b}")
                    return cp.Triangle(lower=a, midpoint=m, upper=b)
                elif dist_type == 'Beta':
                    alpha, beta_value, a, b = params
                    return cp.Beta(alpha, beta_value, lower=a, upper=b)
                elif dist_type == 'LogNormal':
                    mu_log, sigma_log, gamma = params
                    return cp.LogNormal(mu_log, sigma_log, shift=gamma)
                elif dist_type == 'LogNormalMuSigma':
                    mu, sigma, gamma = params
                    variance = sigma ** 2
                    sigma_log = np.sqrt(np.log(1 + (variance / mu ** 2)))
                    mu_log = np.log(mu) - 0.5 * sigma_log ** 2
                    return cp.LogNormal(mu_log, sigma_log, shift=gamma)
                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")

            marginals = [create_distribution(dist_info) for dist_info in distributions]

            # Create joint distribution
            distr_overall = cp.J(*marginals)

            if not sparse:
                num_nodes_estimated = (K + 1) ** num_vars
            else:
                # Estimate for sparse grid quadrature
                from math import comb
                level = K
                growth_rule = 'linear' if not growth else 'exponential'
                # Estimate number of nodes for sparse grid
                num_nodes_estimated = comb(num_vars + level, level)
                if growth:
                    num_nodes_estimated = comb(num_vars + (2 ** level - 1), 2 ** level - 1)

            if num_nodes_estimated > 5000:
                # Time the model execution over 10 runs
                sample_points = distr_overall.sample(10, rule="R")
                start_time = time.time()
                for i in range(10):
                    _ = model(sample_points[:, i])
                end_time = time.time()
                elapsed_time = end_time - start_time
                avg_time_per_eval = elapsed_time / 10

                # Estimate time for 100,000 evaluations
                total_time_seconds = avg_time_per_eval * 100000
                total_time_minutes = total_time_seconds / 60

                # Format the estimated time
                if total_time_minutes < 60:
                    estimated_time_str = f"{total_time_minutes:.2f} minutes"
                else:
                    total_time_hours = total_time_minutes / 60
                    estimated_time_str = f"{total_time_hours:.2f} hours"

                st.error(
                    f"The estimated number of quadrature nodes ({num_nodes_estimated}) exceeds 5,000. "
                    "Generating a PCE surrogate model with more than 5,000 nodes is not recommended. "
                    "Please consider reducing the number of variables, lowering the quadrature order, using sparse grid quadrature, "
                    "or performing dimensionality reduction.\n\n"
                    f"Also, based on the model execution time, it would take approximately {estimated_time_str} "
                    "to propagate 100,000 samples through your model. If this is reasonable, you may not need to create a PCE surrogate "
                    "and can proceed with direct Monte Carlo simulation in the main app."
                )
                st.stop()

            # Generate quadrature nodes and weights
            nodes, weights = cp.generate_quadrature(
                order=K,
                dist=distr_overall,
                rule=quadrature_rule,
                sparse=sparse,
                growth=growth,
                recurrence_algorithm=recurrence_algorithm,
                tolerance=tolerance,
                scaling=scaling
            )

            # Evaluate the model at the quadrature nodes
            num_nodes = nodes.shape[1]
            evals = np.array([model(nodes[:, i])[0] for i in range(num_nodes)])

            # Map orthogonalization method
            orth_method_selected = orthogonal_methods[orth_method]

            # Create the orthogonal polynomials
            poly_expansion = cp.generate_expansion(
                N,
                distr_overall,
                rule=orth_method_selected,
                cross_truncation=cross_truncation
            )

            # Fit the PCE surrogate
            pce_model = cp.fit_quadrature(poly_expansion, nodes, weights, evals)

            # Validation
            n = n_validation
            sampleX = distr_overall.sample(n, rule="R")

            # Evaluate the original model
            direct_results = np.array([model(sampleX[:, i])[0] for i in range(sampleX.shape[1])])

            # Evaluate the PCE surrogate model
            PCE_results = np.array([pce_model(*sampleX[:, i]) for i in range(sampleX.shape[1])])

            # Check shapes
            if direct_results.shape != PCE_results.shape:
                st.error(f'Shape mismatch: direct_results.shape = {direct_results.shape}, PCE_results.shape = {PCE_results.shape}')
                st.stop()

            # Calculate Metrics
            r2 = r2_score(direct_results, PCE_results)
            mae = mean_absolute_error(direct_results, PCE_results)
            rmse = np.sqrt(mean_squared_error(direct_results, PCE_results))
            normalized_rmse = rmse / (np.max(direct_results) - np.min(direct_results))

            # Store metrics in session state
            st.session_state.r2 = r2
            st.session_state.mae = mae
            st.session_state.rmse = rmse
            st.session_state.normalized_rmse = normalized_rmse

            # --- Combined Dashboard Plot ---

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Scatter Plot with 1:1 Line
            axes[0, 0].scatter(direct_results, PCE_results, color='blue', alpha=0.6, label='PCE vs Direct')
            min_val = min(min(direct_results), min(PCE_results))
            max_val = max(max(direct_results), max(PCE_results))
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
            axes[0, 0].set_title('PCE Surrogate Validation - Scatter Plot', fontsize=12)
            axes[0, 0].set_xlabel('Direct Model Results', fontsize=10)
            axes[0, 0].set_ylabel('PCE Surrogate Results', fontsize=10)

            # Add R^2 score to the plot
            axes[0, 0].text(0.7, 0.05, f'$R^2$ = {r2:.4f}', transform=axes[0, 0].transAxes, fontsize=12)

            axes[0, 0].legend(fontsize=10)
            axes[0, 0].grid(True, linestyle='--', alpha=0.7)

            # Residual Plot
            residuals = direct_results - PCE_results

            # Calculate LOESS smoothing manually
            def loess_ag(x, y, frac=0.2):
                """
                Simple LOESS implementation using tricubic kernel weighting.
                """
                n = len(x)
                span = int(np.ceil(frac * n))
                xout = np.sort(x)
                yout = np.zeros_like(xout)
                for i in range(n):
                    dist = np.abs(xout[i] - x)
                    w = np.clip(dist / dist[np.argsort(dist)[span]], 0.0, 1.0)
                    w = (1 - w ** 3) ** 3
                    yout[i] = np.sum(w * y) / np.sum(w)
                return xout, yout

            x_smooth, y_smooth = loess_ag(direct_results, residuals)

            axes[0, 1].scatter(direct_results, residuals, color='blue', alpha=0.6, label='Residuals')
            axes[0, 1].plot(x_smooth, y_smooth, color='red', label='LOESS Smoothing')
            axes[0, 1].hlines(y=0, xmin=np.min(direct_results), xmax=np.max(direct_results), color='gray', linestyles='dashed')
            axes[0, 1].set_title('Residual Plot with LOESS Smoothing', fontsize=12)
            axes[0, 1].set_xlabel('Direct Model Results', fontsize=10)
            axes[0, 1].set_ylabel('Residuals', fontsize=10)
            axes[0, 1].legend(fontsize=10)
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)

            # Histogram of Residuals
            axes[1, 0].hist(residuals, bins=20, color='blue', alpha=0.7)
            axes[1, 0].set_title('Histogram of Residuals', fontsize=12)
            axes[1, 0].set_xlabel('Residuals', fontsize=10)
            axes[1, 0].set_ylabel('Frequency', fontsize=10)
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)

            # PDF Comparison Plot
            sns.kdeplot(direct_results, color='blue', fill=True, alpha=0.4, label='Direct Model', ax=axes[1, 1])
            sns.kdeplot(PCE_results, color='orange', fill=True, alpha=0.4, label='PCE Surrogate', ax=axes[1, 1])
            axes[1, 1].set_title('PDF Comparison: Direct vs. PCE', fontsize=12)
            axes[1, 1].set_xlabel('Model Output', fontsize=10)
            axes[1, 1].set_ylabel('Density', fontsize=10)
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)

            # Add overall title to the figure
            fig.suptitle('PCE Surrogate Model Validation Dashboard', fontsize=16)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate the overall title

            # Store the validation plot
            st.session_state.validation_plot = fig

            # --- Generate the surrogate model code ---
            # Extract the polynomial terms
            coeff_dict = pce_model.todict()
            var_names = input_names  # List of variable names

            terms_list = []
            for exponent_tuple, coeff in coeff_dict.items():
                # Build the term
                term_str = ''
                # Handle the coefficient
                if coeff >= 0 and terms_list:
                    term_str += '    + '
                else:
                    term_str += '    '
                term_str += f'{coeff}'
                # For each variable, if exponent is non-zero, include it
                for var_name, exp in zip(var_names, exponent_tuple):
                    if exp != 0:
                        term_str += f' * ({var_name} ** {int(exp)})'
                terms_list.append(term_str)

            # Combine all terms into a single expression
            polynomial_expression = '\n'.join(terms_list)

            # Format the problem definition
            distributions_formatted = ',\n        '.join(
                [f"{dist}" for dist in problem['distributions']]
            )

            problem_code = f'''
problem = {{
    'num_vars': {problem['num_vars']},
    'names': {problem['names']},
    'distributions': [
        {distributions_formatted}
    ]
}}
'''

            # Generate the function code
            surrogate_model_code = f'''
def function_of_interest(X):
    {', '.join(var_names)} = X
    Y = (
{polynomial_expression}
    )
    return [Y]

# Problem definition
{problem_code}
model = function_of_interest
'''

            st.session_state.surrogate_model_code = surrogate_model_code
            st.session_state.pce_generated = True
            st.success("PCE Surrogate Model generated successfully.")

        except Exception as e:
            st.error(f"Error during PCE Surrogate Generation: {e}")
            st.stop()

# After generating the PCE surrogate
if st.session_state.pce_generated:
    st.markdown("### Validation Plots")
    st.pyplot(st.session_state.validation_plot)

    r2 = st.session_state.get('r2', None)
    mae = st.session_state.get('mae', None)
    rmse = st.session_state.get('rmse', None)
    normalized_rmse = st.session_state.get('normalized_rmse', None)

    if r2 is not None:
        st.markdown(f"**Coefficient of Determination (R²):** {r2:.4f}")
        st.markdown(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.markdown(f"**Normalized RMSE:** {normalized_rmse:.4f}")
        st.markdown("""
        **Interpretation:**
        - An R² value close to 1 indicates an excellent fit between the surrogate model and the original model.
        - Lower MAE and RMSE values indicate better predictive accuracy.
        - Normalized RMSE provides a scale-independent measure of fit.
        - Adjust the PCE parameters to improve the surrogate accuracy if needed.
        """)
    else:
        st.warning("Validation metrics not available.")

    st.markdown("### Surrogate Model Code")
    st.info("Please copy the surrogate model code below and paste it into the main app's code editor.")
    st.code(st.session_state.surrogate_model_code, language='python')

    st.markdown("""
    **Note:**
    - The surrogate model code includes the `problem` definition, so you can paste it directly into the main app.
    - The polynomial expression is formatted across multiple lines for better readability.
    - The surrogate model is an explicit polynomial function and does not require Chaospy or NumPy imports.
    """)
