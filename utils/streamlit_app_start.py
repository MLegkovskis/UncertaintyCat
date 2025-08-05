import streamlit as st
from utils.core_utils import get_model_options
from utils.css_styles import load_css

def streamlit_app_start():
    """
    Handles all Streamlit app-wide configuration, sidebar setup, and page selection.
    Returns (current_code, selected_language_model, selected_page, dropdown_items)
    """
    # Load and apply custom CSS
    st.markdown(load_css(), unsafe_allow_html=True)

    # Add CSS for the custom status indicator
    st.markdown("""
    <style>
    .running-indicator {
        background-color: #f0f2f6;
        border-left: 5px solid #ff9800;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
    }
    .running-indicator:before {
        content: '';
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #ff9800;
        margin-right: 10px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {
            opacity: 1;
        }
        50% {
            opacity: 0.4;
        }
        100% {
            opacity: 1;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Function to show immediate visual feedback
    def show_running_indicator(container, analysis_name):
        container.markdown(f'<div class="running-indicator">Running {analysis_name}...</div>', unsafe_allow_html=True)
    st.show_running_indicator = show_running_indicator  # Attach to st for use elsewhere if needed

    # Header with app title (centered, no logo column)
    st.header("UncertaintyCat | Version 5.21")
    st.caption("Advanced Uncertainty Quantification and Sensitivity Analysis Platform")

    # Fullscreen recommendation
    st.info("📌 **Tip:** This application works best in fullscreen mode to view all analysis tabs properly.")

    # Sidebar styling and navigation
    st.sidebar.image("logo.jpg", width=250)
    st.sidebar.header("Navigation Panel")

    # Create pages with icons
    pages = {
        "📊 UQ Dashboard":           "Comprehensive uncertainty quantification and sensitivity analysis",
        "📉 Dimensionality Reduction": "Reduce model complexity by identifying non-influential variables",
        # "📐 PCE Least-Squares":      "Generate a PCE surrogate via least-squares and inspect Sobol indices",
        "📈 Distribution Fitting":   "Fit probability distributions to your data for UQ analysis"
    }

    selected_page = st.sidebar.radio("", list(pages.keys()))
    st.sidebar.caption(pages[selected_page.strip()])

    # Sidebar divider
    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Model select/upload
    dropdown_items = ["(Select or define your own model)"] + get_model_options()
    current_code = ""
    simulation_results = None
    model = None
    problem = None
    code_snippet = None
    selected_model = st.sidebar.selectbox(
        "Select a Model File:",
        dropdown_items,
        index=0
    )
    if selected_model != "(Select or define your own model)":
        from utils.model_loader import load_model_code
        current_code = load_model_code(selected_model)
    uploaded_file = st.sidebar.file_uploader("or Upload a Python Model File")
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        if st.sidebar.button("Apply Uploaded File", key="apply_uploaded_file"):
            current_code = file_contents

    # LLM model selection in sidebar
    groq_model_options = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "qwen-2.5-32b",
        "deepseek-r1-distill-llama-70b"
    ]
    selected_language_model = st.sidebar.selectbox(
        "Select Language Model:",
        options=groq_model_options,
        index=0
    )
    return current_code, selected_language_model, selected_page, dropdown_items
