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
    """
    Loads code from 'examples/' if valid, otherwise returns an empty string.
    """
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
    - Copy or download the final surrogate code if desired.

    **Detailed Steps**:
    1. Ensure your model defines a Python function named `function_of_interest(X)` 
       (returning a list/array), and a `problem` dict with `'num_vars'`, `'names'`, 
       `'distributions'`.
    2. Select a model from the dropdown or paste your own code in the editor.
    3. Configure the PCE parameters (quadrature order, polynomial order, etc.).
    4. Click **"Generate PCE Surrogate Model"**. The script will:
       - Estimate quadrature nodes,
       - Potentially warn if > 5,000 nodes,
       - Produce validation plots and metrics if feasible.
    5. The resulting polynomial snippet is shown for you to integrate into the main app.
    """)

initialize_session_state()

# Additional session-state keys
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
# 3) SELECT MODEL FROM DROPDOWN
###############################################################################
model_dropdown_items = ["(Select or define your own model)"] + model_options
previous_model = get_session_state("model_file", "(Select or define your own model)")

def on_model_change():
    """Callback that updates code upon a single dropdown selection."""
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

st.markdown("#### Select or Define Model")
st.selectbox(
    "Select a Model File or Enter your own Model:",
    model_dropdown_items,
    index=model_dropdown_items.index(previous_model) if previous_model in model_dropdown_items else 0,
    key="pce_surr_selectbox",
    on_change=on_model_change
)

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

st.markdown("#### Advanced Settings")
with st.expander("Quadrature & Orthogonalization Settings"):
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
        index=list(quadrature_rules.values()).index("gaussian"),
        help="Rule used to generate quadrature nodes & weights."
    )
    orth_method = st.selectbox(
        "Orthogonalization Method:",
        list(orthogonal_methods.keys()),
        index=list(orthogonal_methods.values()).index("three_terms_recurrence"),
        help="Method used for generating the orthonormal polynomials."
    )
    recurrence_algorithm = st.selectbox(
        "Recurrence Algorithm:",
        recurrence_algorithms,
        index=0,
        help="Algorithm for computing recurrence coefficients in quadrature."
    )
    tolerance = st.number_input(
        "Tolerance:", 1e-15, 1e-5, 1e-10, 1e-10,
        format="%.1e",
        help="Tolerance for quadrature convergence."
    )
    scaling = st.number_input(
        "Scaling:", 0.0, 10.0, 3.0, 0.1,
        help="Scaling factor for adaptive quadrature increments."
    )

###############################################################################
# 6) GENERATE PCE SURROGATE MODEL
###############################################################################
run_pce_button = st.button("Generate PCE Surrogate Model")

if run_pce_button:
    code_str = st.session_state.get("code", "")
    if not code_str.strip():
        st.warning("No model code provided.")
        st.stop()

    # 1) Safety check
    try:
        check_code_safety(code_str)
    except Exception as e:
        st.error(f"Error in code safety check: {e}")
        st.stop()

    # 2) Execute user code
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

    # 3) Attempt to build the PCE
    with st.spinner("Generating PCE Surrogate Model..."):
        try:
            input_names = problem["names"]
            num_vars = problem["num_vars"]
            distributions = problem["distributions"]

            # Helper to build Chaospy distributions
            def create_distribution(dist_info):
                dist_type = dist_info["type"]
                params = dist_info["params"]
                if dist_type == "Uniform":
                    a, b = params
                    return cp.Uniform(a, b)
                elif dist_type == "Normal":
                    mu, sigma = params
                    return cp.Normal(mu, sigma)
                elif dist_type == "Gumbel":
                    gamma_param, beta_param = params
                    return cp.Gumbel(loc=gamma_param, scale=beta_param)
                elif dist_type == "Triangular":
                    a, m, b = params
                    if not (a <= m <= b):
                        raise ValueError(
                            f"Triangular requires a <= m <= b, but got a={a}, m={m}, b={b}"
                        )
                    return cp.Triangle(a, m, b)
                elif dist_type == "Beta":
                    alpha, beta_value, lo, hi = params
                    return cp.Beta(alpha, beta_value, lower=lo, upper=hi)
                elif dist_type == "LogNormal":
                    mu_log, sigma_log, gamma = params
                    return cp.LogNormal(mu_log, sigma_log, shift=gamma)
                elif dist_type == "LogNormalMuSigma":
                    mu, sigma, gamma = params
                    var = sigma**2
                    sigma_log = np.sqrt(np.log(1 + var / mu**2))
                    mu_log = np.log(mu) - 0.5 * sigma_log**2
                    return cp.LogNormal(mu_log, sigma_log, shift=gamma)
                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")

            # Build marginals
            marginals = [create_distribution(d) for d in distributions]
            distr_overall = cp.J(*marginals)

            # Estimate # of quadrature nodes
            if not sparse:
                num_nodes_estimated = (K + 1) ** num_vars
            else:
                from math import comb
                if growth:
                    # naive approach for demonstration
                    num_nodes_estimated = 0
                    for dim in range(num_vars + 1):
                        num_nodes_estimated += comb(num_vars, dim) * (2**K)**dim
                else:
                    num_nodes_estimated = comb(num_vars + K, K)

            if num_nodes_estimated > 5000:
                sample_points = distr_overall.sample(10, rule="R")
                t0 = time.time()
                for i in range(10):
                    _ = model(sample_points[:, i])
                t1 = time.time()
                avg_eval_time = (t1 - t0) / 10
                total_time_seconds = avg_eval_time * 100000
                total_time_minutes = total_time_seconds / 60
                if total_time_minutes < 60:
                    est_str = f"{total_time_minutes:.2f} minutes"
                else:
                    est_str = f"{(total_time_minutes/60):.2f} hours"

                st.error(
                    f"Estimated nodes: {num_nodes_estimated} > 5000. "
                    f"It may be infeasible to build a PCE with that many nodes. "
                    f"One might need ~{est_str} for 100k evaluations. "
                    "Try dimensionality reduction or smaller K."
                )
                st.stop()

            # Generate quadrature nodes
            nodes, weights = cp.generate_quadrature(
                order=K,
                dist=distr_overall,
                rule=quadrature_rules[quadrature_rule],
                sparse=sparse,
                growth=growth,
                recurrence_algorithm=recurrence_algorithm,
                tolerance=tolerance,
                scaling=scaling
            )

            # Evaluate model on quadrature nodes
            num_nodes = nodes.shape[1]
            evals = np.array([model(nodes[:, i])[0] for i in range(num_nodes)])

            # Map orth method
            orthogonal_mapping = {
                "Three Terms Recurrence": "three_terms_recurrence",
                "Stieltjes": "stieltjes",
                "Cholesky": "cholesky",
                "Gram-Schmidt": "gram_schmidt"
            }
            orth_method_selected = orthogonal_mapping[orth_method]

            # Build expansions
            poly_expansion = cp.generate_expansion(
                N,
                distr_overall,
                rule=orth_method_selected,
                cross_truncation=cross_truncation
            )

            # Fit PCE
            pce_model = cp.fit_quadrature(poly_expansion, nodes, weights, evals)

            # Validation
            sampleX = distr_overall.sample(n_validation, rule="R")
            direct_results = np.array([model(sampleX[:, i])[0] for i in range(sampleX.shape[1])])
            PCE_results = np.array([pce_model(*sampleX[:, i]) for i in range(sampleX.shape[1])])

            if direct_results.shape != PCE_results.shape:
                st.error("Mismatch in shapes of direct_results vs. PCE_results.")
                st.stop()

            # Metrics
            r2 = r2_score(direct_results, PCE_results)
            mae = mean_absolute_error(direct_results, PCE_results)
            rmse = np.sqrt(mean_squared_error(direct_results, PCE_results))
            normalized_rmse = rmse / (np.max(direct_results) - np.min(direct_results))

            st.session_state.r2 = r2
            st.session_state.mae = mae
            st.session_state.rmse = rmse
            st.session_state.normalized_rmse = normalized_rmse

            # Plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Scatter
            axes[0, 0].scatter(direct_results, PCE_results, alpha=0.6, label='PCE vs Direct')
            min_val = min(direct_results.min(), PCE_results.min())
            max_val = max(direct_results.max(), PCE_results.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[0, 0].set_title('PCE Validation Scatter')
            axes[0, 0].set_xlabel('Direct Model')
            axes[0, 0].set_ylabel('PCE Surrogate')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Residual histogram
            residuals = direct_results - PCE_results
            axes[0, 1].hist(residuals, bins=20, color='blue', alpha=0.7)
            axes[0, 1].set_title('Residuals Histogram')
            axes[0, 1].set_xlabel('Residual')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)

            # Residual plot
            axes[1, 0].scatter(direct_results, residuals, alpha=0.6)
            axes[1, 0].hlines(y=0, xmin=direct_results.min(), xmax=direct_results.max(), linestyles='--', color='red')
            axes[1, 0].set_title('Residuals vs. Direct Model')
            axes[1, 0].set_xlabel('Direct Model')
            axes[1, 0].set_ylabel('Residual')
            axes[1, 0].grid(True)

            # PDF comparison
            sns.kdeplot(direct_results, ax=axes[1,1], fill=True, alpha=0.4, label='Direct')
            sns.kdeplot(PCE_results, ax=axes[1,1], fill=True, alpha=0.4, label='PCE')
            axes[1,1].set_title('PDF Comparison')
            axes[1,1].grid(True)
            axes[1,1].legend()

            plt.tight_layout()
            fig.suptitle('Chaospy PCE Validation', fontsize=16)
            plt.subplots_adjust(top=0.92)  # adjust for suptitle

            st.session_state.validation_plot = fig

            # Extract polynomial terms
            coeff_dict = pce_model.todict()
            terms_list = []
            for exponent_tuple, coeff in coeff_dict.items():
                line = ''
                if coeff >= 0 and len(terms_list) > 0:
                    line += '    + '
                else:
                    line += '    '
                line += f'{coeff}'
                for var_name, exp_val in zip(input_names, exponent_tuple):
                    if exp_val != 0:
                        line += f' * ({var_name} ** {exp_val})'
                terms_list.append(line)

            polynomial_expression = '\n'.join(terms_list)

            # Problem code
            distr_str = ",\n        ".join(str(d) for d in distributions)
            problem_code = f'''
problem = {{
    'num_vars': {num_vars},
    'names': {input_names},
    'distributions': [
        {distr_str}
    ]
}}
'''

            # Surrogate code
            surrogate_model_code = f'''
def function_of_interest(X):
    {', '.join(input_names)} = X
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

###############################################################################
# 7) SHOW RESULTS & CODE
###############################################################################
if st.session_state.get("pce_generated"):
    st.markdown("### Validation Plots")
    if st.session_state.validation_plot:
        st.pyplot(st.session_state.validation_plot)

    r2 = st.session_state.get("r2", None)
    mae = st.session_state.get("mae", None)
    rmse = st.session_state.get("rmse", None)
    normalized_rmse = st.session_state.get("normalized_rmse", None)

    if r2 is not None:
        st.markdown(f"**Coefficient of Determination (R²):** {r2:.4f}")
        st.markdown(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.markdown(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.markdown(f"**Normalized RMSE:** {normalized_rmse:.4f}")
        st.markdown("""
        **Interpretation**:
        - R² close to 1 => excellent fit.
        - Lower MAE & RMSE => better accuracy.
        - Normalized RMSE => scale-independent measure.
        """)
    else:
        st.warning("Validation metrics not available. Please check your model output type.")

    st.markdown("### Surrogate Model Code")
    st.info("You can copy the snippet below or download it as a .py file.")

    st.code(st.session_state.surrogate_model_code, language='python')

    # Provide a Download button for the final surrogate code
    st.download_button(
        label="Download Surrogate Code",
        data=st.session_state.surrogate_model_code,
        file_name="pce_surrogate.py",
        mime="text/x-python"
    )

    st.markdown("""
    **Note**:
    - The snippet includes the updated `problem` dict.
    - The final polynomial expression is explicit and doesn't require Chaospy to evaluate.
    - If you want the surrogate in the main app, just paste this code in place of your original model.
    """)
