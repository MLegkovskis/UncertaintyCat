import hashlib

import streamlit as st
import numpy as np
import plotly.express as px

from app.components import render_app_shell, render_code_editor, render_sidebar_chat
from app.core import get_compiled_model
from app.state import (
    get_model_code,
    get_reliability_preview_context,
    get_reliability_results,
    init_session_state,
    set_reliability_preview_context,
    set_reliability_results,
)
from modules import reliability
from utils.core_utils import check_code_safety, call_groq_api

_METHOD_OPTIONS = [
    "FORM (First Order)",
    "SORM (Second Order)",
    "Monte Carlo Simulation",
    "Directional Sampling",
]
_OPERATOR_OPTIONS = [">", ">=", "<", "<="]


def _method_key(label: str) -> str:
    if label.startswith("FORM"):
        return "FORM"
    if label.startswith("SORM"):
        return "SORM"
    if label.startswith("Monte Carlo"):
        return "MONTE CARLO"
    if label.startswith("Directional"):
        return "DIRECTIONAL SAMPLING"
    return label.upper()


def _render_results(results: dict) -> None:
    if not results:
        return

    prob = results.get("probability")
    beta = results.get("reliability_index")
    cov = results.get("cov")
    ci = results.get("confidence_interval")
    friendly_pf = reliability.format_probability(prob)

    st.header("📊 Reliability Report")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Probability of Failure (Pf)",
        f"{prob:.4%}" if prob is not None else "—",
        help=f"Scientific: {prob:.3e}" if prob is not None else None,
    )
    c2.metric("Reliability Index (β)", f"{beta:.3f}" if beta is not None else "—")
    c3.metric("Odds", friendly_pf, delta_color="off")
    c4.metric("Method", results.get("method", "—"))

    if cov is not None or ci:
        extra = []
        if cov is not None:
            extra.append(f"CoV {cov:.3f}")
        if ci:
            extra.append(f"95% CI [{ci[0]:.3e}, {ci[1]:.3e}]")
        if extra:
            st.caption(" · ".join(extra))

    tab_imp, tab_design, tab_conv, tab_ai = st.tabs([
        "🔍 Importance Factors",
        "📍 Design Point",
        "📈 Convergence",
        "🤖 AI Insights",
    ])

    with tab_imp:
        df_imp = results.get("importance_factors")
        if df_imp is not None and not df_imp.empty:
            fig = px.bar(
                df_imp,
                x="Importance",
                y="Variable",
                orientation="h",
                title="Importance Factors",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Variables with larger importance contribute more to failure risk.")
        else:
            st.info("Importance factors not available for this method.")

    with tab_design:
        df_design = results.get("design_point")
        if df_design is not None and not df_design.empty:
            st.subheader("Most Probable Failure Point")
            st.caption("Physical-space coordinates of the design point.")
            st.dataframe(df_design, use_container_width=True, hide_index=True)
        else:
            st.info("Design point not available.")

    with tab_conv:
        df_conv = results.get("convergence_sample")
        if df_conv is not None and not df_conv.empty:
            x_col = df_conv.columns[0]
            y_col = df_conv.columns[-1]
            fig = px.line(df_conv, x=x_col, y=y_col, title="Probability Estimate vs. Iterations")
            fig.update_yaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Convergence history is only available for simulation-based methods.")

    with tab_ai:
        st.subheader("Automated Expert Assessment")
        if st.button("Generate Reliability Assessment"):
            pf_text = f"{prob:.4e}" if prob is not None else "unknown"
            beta_text = f"{beta:.3f}" if beta is not None else "unknown"
            prompt = f"""
You are a senior reliability engineer. Analyze the following results:
- Method: {results.get('method')}
- Probability of Failure (Pf): {pf_text} ({friendly_pf})
- Reliability Index (β): {beta_text}

Explain whether this safety level is acceptable (β≈3 is a common target),
highlight concerns when CoV/CI are large, and suggest next steps.
"""
            with st.spinner("Consulting Groq..."):
                try:
                    response = call_groq_api(prompt)
                except Exception as exc:
                    st.error(f"Groq API error: {exc}")
                else:
                    st.markdown(response)


def main() -> None:
    init_session_state()
    current_code, selected_language_model = render_app_shell()
    current_code = render_code_editor(current_code)

    st.title("🛡️ Reliability Analysis")
    st.caption("Estimate failure probabilities using FORM, SORM, and simulation techniques.")

    if not current_code:
        st.info("Select or write a model to begin reliability analysis.")
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

    code_hash = hashlib.sha1(current_code.encode("utf-8")).hexdigest()

    with st.expander("1. Configure Reliability Problem", expanded=True):
        conf_cols = st.columns(2)
        with conf_cols[0]:
            st.markdown("##### Failure Definition")
            op_col, thr_col = st.columns([1, 2])
            operator = op_col.selectbox("Operator", _OPERATOR_OPTIONS, label_visibility="collapsed")
            threshold = thr_col.number_input("Threshold", value=0.0, label_visibility="collapsed")
            st.caption(f"Failure occurs when **Model Output {operator} {threshold}**.")
        with conf_cols[1]:
            st.markdown("##### Analysis Method")
            method_label = st.selectbox("Method", _METHOD_OPTIONS, label_visibility="collapsed")

            if "Monte Carlo" in method_label or "Directional" in method_label:
                with st.popover("Simulation Settings"):
                    max_iter = st.number_input("Max Iterations", 1000, 1_000_000, 10_000)
                    target_cov = st.number_input("Target CoV", 0.01, 0.5, 0.05)
            else:
                max_iter, target_cov = 10_000, 0.05

        preview_button = st.button("Preview Output Distribution & Check Definition")
        stats_summary = None
        preview_context = get_reliability_preview_context()
        preview_matches_inputs = (
            preview_context is not None
            and preview_context.get("operator") == operator
            and preview_context.get("threshold") == threshold
            and preview_context.get("code_hash") == code_hash
        )
        if preview_button:
            set_reliability_preview_context(None)
            with st.spinner("Sampling model output..."):
                try:
                    input_sample = problem.getSample(1000)
                    output_sample = model(input_sample)
                    data = np.array(output_sample).flatten()
                    stats_summary = {
                        "min": float(data.min()),
                        "max": float(data.max()),
                        "mean": float(data.mean()),
                        "std": float(data.std()),
                    }
                    fig = reliability.compute_output_distribution_plot(
                        model, problem, threshold, operator
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    set_reliability_preview_context(
                        {
                            "operator": operator,
                            "threshold": threshold,
                            "code_hash": code_hash,
                        }
                    )
                    preview_matches_inputs = True
                except Exception as exc:
                    st.error(f"Preview failed: {exc}")
                    stats_summary = None

        if stats_summary:
            with st.spinner("Consulting AI on failure definition..."):
                code_context = get_model_code() or ""
                prompt = f"""
You are a reliability expert. The user defined a model and a failure condition.

Model code:
```python
{code_context}
```

Output statistics from a 1000-sample preview:
- Range: [{stats_summary['min']:.4f}, {stats_summary['max']:.4f}]
- Mean: {stats_summary['mean']:.4f}
- Std: {stats_summary['std']:.4f}

Failure condition: Output {operator} {threshold}

In 3-4 sentences:
1. Infer what the output likely represents (based on the code).
2. Judge whether the threshold is physically meaningful given the range.
3. Warn if the threshold seems impossible or ill-posed, and suggest a typical definition.
"""
                try:
                    advice = call_groq_api(prompt)
                except Exception as exc:
                    st.error(f"AI guidance failed: {exc}")
                else:
                    impossible = (operator in {">", ">="} and threshold > stats_summary["max"]) or (
                        operator in {"<", "<="} and threshold < stats_summary["min"]
                    )
                    box = st.warning if impossible else st.info
                    prefix = "⚠️ Potential issue:" if impossible else "💡 AI guidance:" 
                    box(f"{prefix} {advice}")

        run_disabled = not preview_matches_inputs
        if run_disabled:
            st.caption("Run the preview with the current failure definition to enable analysis.")

        if st.button(
            "Run Reliability Analysis",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
        ):
            method_key = _method_key(method_label)
            with st.spinner(f"Running {method_key} analysis..."):
                try:
                    res = reliability.run_reliability_analysis(
                        model,
                        problem,
                        method_key,
                        threshold,
                        operator,
                        max_iter=int(max_iter),
                        target_cov=float(target_cov),
                    )
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                else:
                    set_reliability_results(res.to_dict())
                    st.rerun()

    stored = get_reliability_results()
    if stored:
        st.divider()
        _render_results(stored)

    render_sidebar_chat(current_code, selected_language_model)


if __name__ == "__main__":
    main()
