import streamlit as st

from app.components import render_app_shell, render_code_editor, render_sidebar_chat
from app.core import get_compiled_model
from app.displays import display_pce_results
from app.state import (
    get_pce_build_results,
    get_pce_chaos_result,
    get_pce_diag_results,
    init_session_state,
    set_pce_build_results,
    set_pce_chaos_result,
)
from modules import pce_surrogate
from utils.core_utils import check_code_safety


def main() -> None:
    init_session_state()
    current_code, selected_language_model = render_app_shell()
    current_code = render_code_editor(current_code)

    st.title("🛠️ Surrogate Modeling (Polynomial Chaos)")
    st.caption("Build and analyze polynomial chaos surrogates for your current model.")

    if not current_code:
        st.info("Select or write a model to begin building a surrogate.")
        render_sidebar_chat(current_code, selected_language_model)
        return

    is_safe, safety_message = check_code_safety(current_code)
    if not is_safe:
        st.error(f"Security Error: {safety_message}")
        render_sidebar_chat(current_code, selected_language_model)
        return

    try:
        model, problem = get_compiled_model(current_code)
    except Exception as exc:
        st.error(f"Error compiling model: {exc}")
        render_sidebar_chat(current_code, selected_language_model)
        return

    st.divider()
    input_dimension = problem.getDimension()

    with st.expander("1. Configure PCE Model", expanded=True):
        estimation_method = st.selectbox(
            "Estimation Method",
            ["Least Squares (Regression)", "Integration (Quadrature)"],
        )
        max_degree = st.slider("Max Polynomial Degree", 1, 15, 8)

        if estimation_method == "Least Squares (Regression)":
            train_sample_size = st.number_input(
                "Training Sample Size", min_value=50, max_value=20000, value=500
            )
            sparsity_strategy = st.radio(
                "Sparsity Strategy",
                ["Sparse (LARS)", "Full (Least Squares)"],
                horizontal=True,
            )
            quadrature_order = 0
            alpha_ratio = st.slider(
                "Max coefficient/sample ratio (α)",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Controls the recommended degree so that the number of coefficients stays within α × sample size.",
            )
            suggested_degree = pce_surrogate.suggest_degree_from_ratio(
                int(train_sample_size), input_dimension, alpha_ratio
            )
            if suggested_degree is not None:
                st.caption(
                    f"Suggested maximum degree (α={alpha_ratio:.2f}): {suggested_degree}"
                )
        else:
            quadrature_order = st.number_input(
                "Gauss Product Order (per dimension)", min_value=2, max_value=10, value=4
            )
            train_sample_size = 0
            sparsity_strategy = "Sparse (LARS)"
            alpha_ratio = None

        if st.button("Build Surrogate Model"):
            with st.spinner("Building polynomial chaos surrogate..."):
                chaos_result, training_size = pce_surrogate.build_pce_surrogate(
                    model,
                    problem,
                    estimation_method,
                    max_degree,
                    int(train_sample_size),
                    int(quadrature_order),
                    sparsity_strategy,
                )
                validation = pce_surrogate.compute_pce_validation(
                    chaos_result, model, problem
                )
                config = {
                    "estimation_method": estimation_method,
                    "degree": int(max_degree),
                    "sample_size": int(train_sample_size),
                    "quadrature_order": int(quadrature_order),
                    "sparsity_strategy": sparsity_strategy,
                    "training_sample_size": training_size,
                    "dimension": input_dimension,
                    "alpha_ratio": alpha_ratio,
                }
                set_pce_chaos_result(chaos_result)
                set_pce_build_results({"validation": validation, "config": config})
                st.success("Surrogate model built successfully.")

    chaos_result = get_pce_chaos_result()
    build_results = get_pce_build_results()
    diag_results = get_pce_diag_results()

    if chaos_result and build_results:
        st.divider()
        display_pce_results(
            chaos_result,
            build_results,
            diag_results,
            model,
            problem,
            build_results.get("config", {}),
        )

    render_sidebar_chat(current_code, selected_language_model)


if __name__ == "__main__":
    main()
