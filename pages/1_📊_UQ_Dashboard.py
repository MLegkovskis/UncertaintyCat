import streamlit as st

from app.components import render_app_shell, render_code_editor, render_sidebar_chat
from app.config import ANALYSIS_REGISTRY
from app.core import (
    ensure_simulation_data,
    get_compiled_model,
    run_all_analyses,
    run_single_analysis,
)
from app.state import (
    analyses_completed,
    get_all_results,
    get_simulation_data,
    init_session_state,
)
from utils.core_utils import check_code_safety


def main():
    init_session_state()

    current_code, selected_language_model = render_app_shell()
    current_code = render_code_editor(current_code)

    if not current_code:
        st.info("Please select or upload a model file to begin analysis.")
        return

    is_safe, safety_message = check_code_safety(current_code)
    if not is_safe:
        st.error(f"Security Error: {safety_message}")
        return

    try:
        model, problem = get_compiled_model(current_code)
    except Exception as exc:
        st.error(f"Error compiling model: {exc}")
        return

    if get_simulation_data() is None:
        with st.spinner("Running Initial Monte Carlo Simulation for EDA..."):
            ensure_simulation_data(model, problem, sample_size=1000)
    else:
        ensure_simulation_data(model, problem, sample_size=1000)

    st.markdown("---")
    st.header("🚀 Run All Analyses")
    if st.button(
        "Run Full UQ Suite",
        key="run_all_analyses_button",
        help="Runs all UQ analyses and generates a comprehensive report below.",
    ):
        status_text = st.empty()
        progress_bar = st.progress(0.0)

        def update_progress(name: str, index: int, total: int) -> None:
            status_text.markdown(
                f'<div class="running-indicator">Running {name}...</div>',
                unsafe_allow_html=True,
            )
            progress_bar.progress(index / total)

        with st.spinner("Running the full UQ suite..."):
            run_all_analyses(
                model,
                problem,
                current_code,
                selected_language_model,
                progress_callback=update_progress,
            )
        results = get_all_results()
        had_error = any(
            isinstance(data, dict) and data.get("error") for data in results.values()
        )
        if had_error:
            status_text.error(
                "One or more analyses reported errors. Please review the sections below."
            )
        else:
            status_text.success("All analyses completed! You can now view the comprehensive report below.")
        progress_bar.progress(1.0)

    st.markdown("---")
    st.header("🔬 Run Individual Analysis Modules")
    st.caption(
        "Click a button to run a specific analysis. Results will be stored and displayed in the report section below."
    )

    cols = []
    for idx, analysis in enumerate(ANALYSIS_REGISTRY):
        if idx % 3 == 0:
            cols = st.columns(3)
        col = cols[idx % 3]
        with col:
            if st.button(analysis["button_label"], key=f"run_{analysis['key']}"):
                with st.spinner(f"Running {analysis['name']}..."):
                    results = run_single_analysis(
                        analysis["key"],
                        model,
                        problem,
                        current_code,
                        selected_language_model,
                    )
                if isinstance(results, dict) and results.get("error"):
                    st.error(f"Error: {results['error']}")
                else:
                    st.success(f"{analysis['name']} completed.")

    st.markdown("---")
    results = get_all_results()
    if results:
        is_full_run_complete = analyses_completed()
        report_title = (
            "Comprehensive Analysis Report"
            if is_full_run_complete
            else "Stored Analysis Results"
        )
        report_caption = (
            "This report contains the results of all UQ analyses. Click on each section to expand."
            if is_full_run_complete
            else "This section shows results from individually run analyses or a previous full UQ run. Click 'Run Full UQ Suite' for a fresh comprehensive report. Expand sections for details."
        )
        st.header(f"📊 {report_title}")
        st.caption(report_caption)

        for analysis in ANALYSIS_REGISTRY:
            module_name = analysis["name"]
            module_results = results.get(module_name)
            if not module_results:
                continue
            is_error = isinstance(module_results, dict) and module_results.get("error")
            expander_title = (
                f"⚠️ {module_name} (Error)" if is_error else f"📄 {module_name}"
            )
            with st.expander(expander_title, expanded=bool(is_error)):
                if is_error:
                    st.error(
                        f"An error occurred during the '{module_name}' analysis: {module_results['error']}"
                    )
                else:
                    st.subheader(f"{module_name} Results")
                    analysis["display"](
                        module_results, current_code, selected_language_model
                    )
    else:
        st.info(
            "Run the 'Full UQ Suite' or individual analysis modules to generate and view reports."
        )

    render_sidebar_chat(current_code, selected_language_model)


if __name__ == "__main__":
    main()
