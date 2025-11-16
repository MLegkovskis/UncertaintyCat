"""Presentation helpers for the Streamlit application."""

from __future__ import annotations

import streamlit as st

from app.state import update_pce_diag_result
from modules import pce_surrogate
from modules.ancova_analysis import display_ancova_results
from modules.correlation_analysis import display_correlation_results
from modules.expectation_convergence_analysis import display_expectation_convergence_results
from modules.exploratory_data_analysis import display_exploratory_data_analysis_results
from modules.fast_analysis import display_fast_results
from modules.hsic_analysis import display_hsic_results
from modules.ml_analysis import display_ml_results
from modules.taylor_analysis import display_taylor_results


def display_model_understanding_view(results, _code, language_model):
    """Render the model understanding section using Streamlit widgets."""
    if not results:
        st.error("Model understanding results are unavailable.")
        return

    inputs_df = results.get('inputs_df')
    explanation = results.get('explanation', '')
    is_pce_used = results.get('is_pce_used', False)
    metamodel_str = results.get('metamodel_str')

    if explanation and 'model_understanding_response_markdown' not in st.session_state:
        st.session_state['model_understanding_response_markdown'] = explanation

    if explanation:
        st.markdown(explanation)
    else:
        st.info("No narrative explanation was generated for this analysis.")

    st.write("### Input Distributions")
    if inputs_df is not None:
        try:
            st.dataframe(
                inputs_df,
                column_config={
                    "Variable": st.column_config.TextColumn("Variable Name"),
                    "Distribution": st.column_config.TextColumn("Distribution Type"),
                    "Parameters": st.column_config.TextColumn("Parameters"),
                    "Bounds": st.column_config.TextColumn("Bounds"),
                    "Mean": st.column_config.NumberColumn("Mean", format="%.4f"),
                    "Std": st.column_config.NumberColumn("Std. Dev.", format="%.4f"),
                },
                width='stretch',
            )
        except Exception as exc:  # pragma: no cover - defensive UI guard
            st.error(f"Error displaying input distributions: {exc}")
    else:
        st.info("Input distribution summary is not available.")

    problem = results.get('problem')
    if problem is not None and hasattr(problem, '__repr_markdown__'):
        try:
            st.markdown(problem.__repr_markdown__(), unsafe_allow_html=True)
        except Exception as exc:  # pragma: no cover - optional rendering
            st.warning(f"Unable to render distribution markdown: {exc}")

    if is_pce_used and metamodel_str:
        st.write("### Polynomial Chaos Expansion Metamodel")
        st.info(
            "This analysis uses a polynomial chaos expansion (PCE) metamodel instead of the original model."
        )
        st.code(metamodel_str, language="python")


def display_exploratory_view(results, _code, _language_model):
    return display_exploratory_data_analysis_results(results)


def display_expectation_view(results, _code, _language_model):
    return display_expectation_convergence_results(results)


def display_correlation_view(results, _code, _language_model):
    return display_correlation_results(results)


def display_taylor_view(results, model_code, language_model):
    return display_taylor_results(results, model_code_str=model_code, language_model=language_model)


def display_hsic_view(results, model_code, language_model):
    return display_hsic_results(results, model_code_str=model_code, language_model=language_model)


def display_sobol_view(results, _code, _language_model):
    """Render Sobol sensitivity results."""
    if not results:
        st.error("Sobol analysis results are unavailable.")
        return

    indices_df = results.get('indices_df')
    sum_first_order = results.get('sum_first_order')
    sum_total_order = results.get('sum_total_order')
    interaction_effect = results.get('interaction_effect')
    fig_bar = results.get('fig_bar')
    fig_interaction = results.get('fig_interaction')
    fig_heatmap = results.get('fig_heatmap')
    S2_matrix = results.get('S2_matrix')
    dimension = results.get('dimension') or 0
    dist_df = results.get('dist_info')
    interactions_df = results.get('interactions_df')
    ai_insights = results.get('ai_insights')

    st.markdown("## Sobol Sensitivity Analysis")
    with st.container():
        st.subheader("Sensitivity Indices")
        if fig_bar:
            st.plotly_chart(fig_bar, width='stretch')
        if indices_df is not None:
            st.subheader("Sensitivity Indices Table")
            st.markdown(
                """
                This table summarizes first-order ($S_i$) and total-order ($T_i$) Sobol indices, the
                share of each variable due to interactions, and their 95% confidence intervals.
                """
            )
            st.dataframe(indices_df, width='stretch')

        st.markdown("#### Variance Decomposition Summary")
        col1, col2 = st.columns(2)
        with col1:
            if sum_first_order is not None:
                st.metric("Sum of First-Order Indices", f"{sum_first_order:.4f}")
                if sum_first_order > 1.0:
                    st.warning(
                        "Sum of first-order indices exceeds 1.0, indicating potential numerical issues."
                    )
        with col2:
            if sum_total_order is not None:
                st.metric("Sum of Total-Order Indices", f"{sum_total_order:.4f}")
                if sum_total_order is not None and sum_first_order is not None and sum_total_order < sum_first_order:
                    st.warning(
                        "Total-order indices sum is less than the first-order sum, which should not occur."
                    )

        if sum_first_order is not None and interaction_effect is not None:
            st.markdown(
                f"""
                - **Sum of First-Order Indices = {sum_first_order:.4f}:** values near 1 imply an additive model,
                  while lower values indicate interactions.
                - **Interaction Effect = {interaction_effect:.4f}:** represents the variance share
                  driven by cross-variable interactions.
                """
            )

        st.subheader("Interaction Effects")
        if fig_interaction:
            st.plotly_chart(fig_interaction, width='stretch')

        if dimension > 1 and S2_matrix is not None and fig_heatmap:
            st.subheader("Second-Order Interactions")
            st.plotly_chart(fig_heatmap, width='stretch')
            if interactions_df is not None:
                st.subheader("Top Interactions")
                st.dataframe(interactions_df, width='stretch')

        if dist_df is not None:
            st.subheader("Input Distributions")
            st.dataframe(dist_df, width='stretch')

    if ai_insights:
        st.subheader("AI-Generated Expert Analysis")
        st.markdown(ai_insights)


def display_fast_view(results, _code, _language_model):
    return display_fast_results(results)


def display_ancova_view(results, model_code, language_model):
    return display_ancova_results(results, language_model=language_model, model_code_str=model_code)


def display_shapley_view(results, model_code, language_model):
    return display_ml_results(results, model_code_str=model_code, language_model=language_model)


def display_overall_summary(results, _code, _language_model):
    """Render the overall LLM-generated synthesis."""
    summary_text = results.get("summary_text")
    sources = results.get("sources", [])

    if summary_text:
        st.markdown(summary_text)
    else:
        st.info("No summary text available yet.")

    if sources:
        st.caption(f"Summary based on: {', '.join(sources)}")


def display_pce_results(
    chaos_result,
    build_results,
    diag_results,
    model,
    problem,
    build_config,
):
    """Render surrogate diagnostics and trigger additional analyses."""
    if chaos_result is None or build_results is None:
        st.info("Build a surrogate model to access diagnostics.")
        return

    validation = build_results.get("validation")
    st.header("2. Diagnostics & Exploitation")
    tabs = st.tabs(
        [
            "📊 Validation",
            "📈 Degree Sensitivity",
            "🔍 Sobol' Indices",
            "📉 Conditional Expectation",
            "📄 Coefficients",
        ]
    )

    with tabs[0]:
        st.subheader("Metamodel Validation")
        if validation:
            st.metric("R² Score", f"{validation['r2_score']:.4f}")
            st.plotly_chart(validation["validation_plot"], width="stretch")
        else:
            st.info("Validation metrics will appear after building the surrogate.")

    with tabs[1]:
        st.subheader("R² Score vs. Polynomial Degree")
        st.caption(
            "Runs a K-fold cross-validation loop per degree. This computation can take a moment."
        )
        kfold_n = st.number_input("Sample Size", 100, 5000, 300, key="pce_kfold_n")
        kfold_k = st.slider("K-Folds", 3, 10, 5, key="pce_kfold_k")
        kfold_deg = st.slider("Max Degree", 3, 15, build_config.get("degree", 8), key="pce_kfold_deg")
        if st.button("Run Degree Sensitivity Analysis"):
            with st.spinner("Running cross-validation..."):
                fig = pce_surrogate.compute_degree_sensitivity_kfold(
                    model, problem, kfold_n, kfold_deg, kfold_k
                )
                update_pce_diag_result("degree_sensitivity_plot", fig)
        if diag_results.get("degree_sensitivity_plot") is not None:
            st.plotly_chart(diag_results["degree_sensitivity_plot"], width="stretch")

    with tabs[2]:
        st.subheader("Sobol' Sensitivity Indices")
        bootstrap = st.number_input(
            "Bootstrap Rebuilds (0 disables confidence intervals)",
            min_value=0,
            max_value=50,
            value=0,
            key="pce_sobol_bootstrap",
        )
        if st.button("Compute Sobol' Indices"):
            with st.spinner("Computing Sobol' indices..."):
                fig = pce_surrogate.compute_pce_sobol_indices(
                    chaos_result,
                    model,
                    problem,
                    build_config,
                    bootstrap,
                )
                update_pce_diag_result("sobol_plot", fig)
        if diag_results.get("sobol_plot") is not None:
            st.plotly_chart(diag_results["sobol_plot"], width="stretch")

    with tabs[3]:
        st.subheader("Conditional Expectation")
        st.caption("Generates scatter plots with PCE-based conditional expectations for each input.")
        if st.button("Generate Conditional Plots"):
            with st.spinner("Generating conditional expectation plots..."):
                try:
                    graphs = pce_surrogate.get_pce_conditional_expectation_plots(
                        chaos_result, model, problem
                    )
                except RuntimeError as err:
                    st.error(str(err))
                else:
                    update_pce_diag_result("conditional_plots", graphs)
        if diag_results.get("conditional_plots"):
            for fig in diag_results["conditional_plots"]:
                st.plotly_chart(fig, width="stretch")

    with tabs[4]:
        st.subheader("PCE Coefficients")
        if "coefficients_df" not in diag_results:
            df = pce_surrogate.get_pce_coefficients_table(chaos_result)
            update_pce_diag_result("coefficients_df", df)
        st.dataframe(diag_results.get("coefficients_df"), width="stretch")
