import os
import streamlit as st
import numpy as np
import openturns as ot
import pandas as pd
import plotly.express as px

# Import modules
from modules.monte_carlo import monte_carlo_simulation, create_monte_carlo_dataframe
from modules.model_understanding import model_understanding
from modules.exploratory_data_analysis import exploratory_data_analysis
from modules.expectation_convergence_analysis import expectation_convergence_analysis_joint
from modules.sobol_sensitivity_analysis import sobol_sensitivity_analysis
from modules.taylor_analysis import taylor_analysis
from modules.correlation_analysis import correlation_analysis
from modules.hsic_analysis import hsic_analysis
from modules.ml_analysis import ml_analysis
from modules.morris_analysis import morris_analysis
from modules.fast_ancova_analysis import fast_sensitivity_analysis, ancova_sensitivity_analysis

# Import utils
from utils.core_utils import (
    get_model_options,
    call_groq_api
)
from utils.model_utils import (
    validate_problem_structure,
    test_model,
    sample_inputs
)
from utils.constants import RETURN_INSTRUCTION

###############################################################################
# 2) LOAD MODEL CODE FROM EXAMPLES
###############################################################################
def load_model_code(selected_model_name: str) -> str:
    """
    Loads code from 'examples/' folder if a valid model is selected.
    Otherwise returns an empty string.
    """
    try:
        file_path = os.path.join('examples', selected_model_name)
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return ""

###############################################################################
# 3) STREAMLIT APP START
###############################################################################
# Set page configuration with custom theme
st.set_page_config(
    layout="wide",
    page_title="UncertaintyCat | Enterprise",
    page_icon="🐱",
    initial_sidebar_state="expanded"
)

# Custom CSS for enterprise look and feel
st.markdown("""
<style>
    .main-header {
        font-family: 'Arial', sans-serif;
        color: #2C3E50;
        padding-bottom: 15px;
        border-bottom: 2px solid #3498DB;
    }
    .sub-header {
        font-family: 'Arial', sans-serif;
        color: #34495E;
        padding: 10px 0;
        margin-top: 20px;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9FA;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #EBF5FB;
        border-left: 5px solid #3498DB;
    }
    .success-box {
        background-color: #E9F7EF;
        border-left: 5px solid #2ECC71;
    }
    .warning-box {
        background-color: #FEF9E7;
        border-left: 5px solid #F1C40F;
    }
    .error-box {
        background-color: #FDEDEC;
        border-left: 5px solid #E74C3C;
    }
    .stTextArea>div>div>textarea {
        font-family: 'Courier New', monospace;
        background-color: #F8F9FA;
    }
    .section-divider {
        height: 3px;
        background-color: #F0F3F4;
        margin: 30px 0;
        border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.jpg", width=100)
with col2:
    st.markdown('<h1 class="main-header">UncertaintyCat | Enterprise Edition</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #7F8C8D;">Advanced Uncertainty Quantification and Sensitivity Analysis Platform</p>', unsafe_allow_html=True)

# Sidebar styling and navigation
st.sidebar.markdown('<div style="text-align: center; padding: 10px 0;"><h3>Navigation Panel</h3></div>', unsafe_allow_html=True)

# Create pages with icons
pages = {
    "📊 Main Analysis": "Comprehensive uncertainty quantification and sensitivity analysis",
    "📉 Dimensionality Reduction": "Reduce model complexity by identifying non-influential variables"
}
selected_page = st.sidebar.radio("", list(pages.keys()))
st.sidebar.markdown(f'<div class="info-box status-box">{pages[selected_page.strip()]}</div>', unsafe_allow_html=True)

# Sidebar divider
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

###############################################################################
# 4) MODEL SELECT / UPLOAD
###############################################################################
st.sidebar.markdown('<h3>Model Configuration</h3>', unsafe_allow_html=True)

# Insert placeholder item at index 0 for "no model selected"
dropdown_items = ["(Select or define your own model)"] + get_model_options()

# Variables to store current state
current_code = ""
simulation_results = None
model = None
problem = None
code_snippet = None

# Model selection in sidebar
selected_model = st.sidebar.selectbox(
    "Select a Model File:",
    dropdown_items,
    index=0
)

if selected_model != "(Select or define your own model)":
    current_code = load_model_code(selected_model)

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("or Upload a Python Model File")
if uploaded_file is not None:
    file_contents = uploaded_file.read().decode("utf-8")
    if st.sidebar.button("Apply Uploaded File", key="apply_uploaded_file"):
        current_code = file_contents

# LLM model selection in sidebar
st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<h3>AI Configuration</h3>', unsafe_allow_html=True)

groq_model_options = [
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "qwen-2.5-32b",
    "deepseek-r1-distill-llama-70b"
]
selected_language_model = st.sidebar.selectbox(
    "Select Language Model:",
    options=groq_model_options,
    index=0
)

###############################################################################
# 5) CODE EDITOR & SYNTAX-PREVIEW
###############################################################################
st.markdown('<h2 class="sub-header">Model Definition</h2>', unsafe_allow_html=True)

with st.expander("Model Code Editor & Preview", expanded=True):
    col_code, col_preview = st.columns(2)

    with col_code:
        st.markdown('<p style="font-weight: bold; margin-bottom: 10px;">Model Code Editor</p>', unsafe_allow_html=True)
        code_area_value = st.text_area(
            label="",
            value=current_code,
            height=300
        )
        # Update current code if changed in the editor
        current_code = code_area_value

    with col_preview:
        st.markdown('<p style="font-weight: bold; margin-bottom: 10px;">Syntax-Highlighted Preview</p>', unsafe_allow_html=True)
        if current_code.strip():
            st.code(current_code, language="python")
        else:
            st.markdown('<div class="info-box status-box">No code to display. Please select or upload a model.</div>', unsafe_allow_html=True)

###############################################################################
# 7) PAGE-SPECIFIC CONTENT
###############################################################################
if "📊 Main Analysis" in selected_page:
    st.markdown('<h2 class="sub-header">Uncertainty Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    # Create a card for the run button
    st.markdown('<p style="font-weight: bold;">Run Comprehensive Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p>Click the button below to run a full suite of uncertainty and sensitivity analyses on your model.</p>', unsafe_allow_html=True)
    run_button = st.button("Run Analysis", key="run_main_simulation")
    st.markdown('</div>', unsafe_allow_html=True)

    ###############################################################################
    # 8) MAIN SIMULATION LOGIC
    ###############################################################################
    if run_button:
        if not current_code:
            st.markdown('<div class="error-box status-box">Please provide a model first.</div>', unsafe_allow_html=True)
        else:
            try:
                # Execute the model code in a fresh namespace
                local_namespace = {}
                exec(current_code, local_namespace)
                
                # Check that both model and problem are defined
                if 'model' not in local_namespace or 'problem' not in local_namespace:
                    st.markdown('<div class="error-box status-box">The model code must define both \'model\' and \'problem\'.</div>', unsafe_allow_html=True)
                else:
                    model = local_namespace['model']
                    problem = local_namespace['problem']

                    # Validate the problem structure
                    validate_problem_structure(problem)

                    # Get number of samples from UI
                    N = 2000
                    
                    # Run Monte Carlo simulation
                    with st.spinner("Running Monte Carlo Simulation..."):
                        st.markdown('<div class="info-box status-box">Running Monte Carlo simulation with 2000 samples...</div>', unsafe_allow_html=True)
                        results = monte_carlo_simulation(model, problem, N=N, seed=42)
                        data = create_monte_carlo_dataframe(results)
                        st.markdown('<div class="success-box status-box">Monte Carlo simulation completed successfully.</div>', unsafe_allow_html=True)
                    
                    # Store results for use in analyses
                    simulation_results = {
                        "data": data,
                        "model": model,
                        "problem": problem,
                        "code": current_code,
                        "selected_language_model": selected_language_model,
                        "N": N
                    }
                    
                    # Display results in styled sections
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">Model Understanding</h2>', unsafe_allow_html=True)
                    with st.spinner("Running Model Understanding..."):
                        model_understanding(
                            model,
                            problem,
                            current_code,
                            language_model=selected_language_model
                        )

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running Exploratory Data Analysis..."):
                        exploratory_data_analysis(
                            data, N, model, problem, current_code,
                            language_model=selected_language_model
                        )

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">Expectation Convergence Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running Expectation Convergence Analysis..."):
                        expectation_convergence_analysis_joint(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )

                    # Run all analyses by default (removed checkboxes)
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">Sobol Sensitivity Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running Sobol Sensitivity Analysis..."):
                        sobol_sensitivity_analysis(
                            1024, model, problem, current_code,
                            language_model=selected_language_model
                        )

                    # Add FAST Analysis
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">FAST Sensitivity Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running FAST Sensitivity Analysis..."):
                        fast_results = fast_sensitivity_analysis(
                            model, problem, size=400, model_code_str=current_code,
                            language_model=selected_language_model
                        )
                        
                        if fast_results:
                            # Display the explanation
                            st.markdown(fast_results['explanation'], unsafe_allow_html=True)
                            
                            # Display the indices table
                            st.markdown('<p style="font-weight: bold; margin-top: 20px;">FAST Sensitivity Indices</p>', unsafe_allow_html=True)
                            st.dataframe(fast_results['indices_df'][['Variable', 'First Order', 'Total Order', 'Interaction', 'Interaction %']], use_container_width=True)
                            
                            # Display the bar chart
                            st.plotly_chart(fast_results['fig_bar'], use_container_width=True)
                            
                            # Display pie charts in two columns
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fast_results['fig_pie_first'], use_container_width=True)
                            with col2:
                                st.plotly_chart(fast_results['fig_pie_total'], use_container_width=True)
                            
                            # Display LLM insights if available
                            if fast_results['llm_insights'] and selected_language_model:
                                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                                st.markdown('<h3 class="sub-header">FAST Analysis Insights</h3>', unsafe_allow_html=True)
                                st.markdown(fast_results['llm_insights'], unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add ANCOVA Analysis
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">ANCOVA Sensitivity Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running ANCOVA Sensitivity Analysis..."):
                        ancova_results = ancova_sensitivity_analysis(
                            model, problem, size=2000, model_code_str=current_code,
                            language_model=selected_language_model
                        )
                        
                        if ancova_results:
                            # Display the explanation
                            st.markdown(ancova_results['explanation'], unsafe_allow_html=True)
                            
                            # Display the indices table
                            st.markdown('<p style="font-weight: bold; margin-top: 20px;">ANCOVA Sensitivity Indices</p>', unsafe_allow_html=True)
                            st.dataframe(ancova_results['indices_df'][['Variable', 'ANCOVA Index', 'Uncorrelated Index', 'Correlated Index', 'Correlation %']], use_container_width=True)
                            
                            # Display the bar chart
                            st.plotly_chart(ancova_results['fig_bar'], use_container_width=True)
                            
                            # Display stacked bar chart
                            st.plotly_chart(ancova_results['fig_stacked'], use_container_width=True)
                            
                            # Display pie chart and heatmap in two columns
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(ancova_results['fig_pie'], use_container_width=True)
                            with col2:
                                st.plotly_chart(ancova_results['fig_heatmap'], use_container_width=True)
                            
                            # Display LLM insights if available
                            if ancova_results['llm_insights'] and selected_language_model:
                                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                                st.markdown('<h3 class="sub-header">ANCOVA Analysis Insights</h3>', unsafe_allow_html=True)
                                st.markdown(ancova_results['llm_insights'], unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">Taylor Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running Taylor Analysis..."):
                        taylor_analysis(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">Correlation Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running Correlation Analysis..."):
                        correlation_analysis(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">HSIC Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running HSIC Analysis..."):
                        hsic_analysis(
                            model, problem, current_code,
                            language_model=selected_language_model
                        )

                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h2 class="sub-header">SHAP Analysis</h2>', unsafe_allow_html=True)
                    with st.spinner("Running SHAP Analysis..."):
                        ml_analysis(
                            data, problem, current_code,
                            language_model=selected_language_model
                        )
                    
                    st.markdown('<div class="success-box status-box">All analyses completed successfully!</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown(f'<div class="error-box status-box">Error during simulation: {str(e)}</div>', unsafe_allow_html=True)

elif "📉 Dimensionality Reduction" in selected_page:
    st.markdown('<h2 class="sub-header">Dimensionality Reduction with Morris Method</h2>', unsafe_allow_html=True)
    
    # First, check if we have a model from the code editor
    if not current_code and model is None:
        st.markdown('<div class="info-box status-box">Please define your model in the Model Definition section first.</div>', unsafe_allow_html=True)
        
        st.markdown('<p style="font-weight: bold;">How to use the Morris Analysis:</p>', unsafe_allow_html=True)
        st.markdown("""
        1. First, define or load your model in the **Model Definition** section
        2. Click the **Run Analysis** button to perform the Morris analysis
        
        The Morris method helps identify which input variables have minimal impact on your model's output.
        This allows you to create simplified models by fixing non-influential variables at nominal values.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # If we have code but no model yet, try to execute it
        if current_code and model is None:
            try:
                # Execute the model code in a fresh namespace
                local_namespace = {}
                exec(current_code, local_namespace)
                
                # Check that both model and problem are defined
                if 'model' in local_namespace and 'problem' in local_namespace:
                    model = local_namespace['model']
                    problem = local_namespace['problem']
            except Exception as e:
                st.markdown(f'<div class="error-box status-box">Error executing model code: {str(e)}</div>', unsafe_allow_html=True)
        
        # Create a card for the configuration
        st.markdown('<p style="font-weight: bold;">Morris Analysis Configuration</p>', unsafe_allow_html=True)
        
        # Configuration controls
        col1, col2 = st.columns(2)
        with col1:
            n_trajectories = st.slider("Number of trajectories", min_value=5, max_value=50, value=10, 
                                      help="More trajectories provide better estimates but increase computation time")
        with col2:
            n_levels = st.slider("Number of levels", min_value=3, max_value=10, value=5,
                                help="Number of levels in the Morris grid design")
        
        threshold = st.slider("Threshold for non-influential variables (%)", min_value=1, max_value=20, value=5,
                             help="Variables with effects below this percentage of the maximum effect are considered non-influential")
        
        run_button = st.button("Run Morris Analysis", key="run_morris_analysis")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if run_button and model is not None and problem is not None:
            with st.spinner("Running Morris analysis..."):
                try:
                    # Run Morris analysis directly
                    from modules.morris_analysis import run_morris_analysis, plot_morris_results_plotly, identify_non_influential_variables, get_recommended_fixed_values
                    
                    # Run Morris analysis
                    results = run_morris_analysis(model, problem, n_trajectories, n_levels)
                    
                    # Plot results using Plotly
                    fig1, fig2 = plot_morris_results_plotly(results)
                    
                    # Display plots in styled sections
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h3 class="sub-header">Morris Mean Absolute Elementary Effects</h3>', unsafe_allow_html=True)
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    st.markdown('<h3 class="sub-header">Morris Analysis: μ* vs σ</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="info-box status-box">Variables in the top-right corner have high influence and non-linear effects or interactions. Variables near the origin have low influence on the output.</div>', unsafe_allow_html=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Identify non-influential variables
                    non_influential = identify_non_influential_variables(results, threshold)
                    
                    if non_influential:
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        st.markdown('<h3 class="sub-header">Non-influential Variables</h3>', unsafe_allow_html=True)
                        st.markdown(f'<div class="info-box status-box">The following variables have effects below {threshold}% of the maximum effect and can potentially be fixed:</div>', unsafe_allow_html=True)
                        
                        # Create a DataFrame for display
                        non_infl_df = pd.DataFrame(non_influential, columns=['Variable', 'Index', 'Effect'])
                        non_infl_df = non_infl_df.sort_values('Effect')
                        
                        # Calculate max effect for reference
                        max_effect = max(results['mean_abs_effects'])
                        
                        # Create an enhanced DataFrame with more information
                        enhanced_df = pd.DataFrame({
                            'Variable': non_infl_df['Variable'],
                            'Effect Value': non_infl_df['Effect'].round(6),
                            'Relative Effect (%)': (non_infl_df['Effect'] / max_effect * 100).round(2)
                        })
                        
                        # Display enhanced table
                        st.dataframe(enhanced_df, use_container_width=True)
                        
                        # Create a visual representation of the effects
                        st.markdown('<p style="font-weight: bold; margin-top: 20px;">Visual Comparison of Non-influential Variables</p>', unsafe_allow_html=True)
                        
                        # Create a horizontal bar chart with Plotly for the non-influential variables
                        fig_non_infl = px.bar(
                            enhanced_df.sort_values('Effect Value', ascending=True), 
                            x='Effect Value',
                            y='Variable',
                            orientation='h',
                            text='Relative Effect (%)',
                            color='Effect Value',
                            color_continuous_scale='Blues',
                            labels={'Effect Value': 'Morris Effect', 'Variable': 'Input Variables'},
                            title='Relative Influence of Non-influential Variables'
                        )
                        
                        # Improve the layout
                        fig_non_infl.update_traces(
                            texttemplate='%{text}%',
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>Effect: %{x:.6f}<br>Relative: %{text}%<extra></extra>'
                        )
                        
                        fig_non_infl.update_layout(
                            font=dict(size=12),
                            height=max(300, len(non_influential) * 50),  # Dynamic height based on number of variables
                            margin=dict(l=20, r=20, t=50, b=20),
                            plot_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='lightgray'),
                            yaxis=dict(showgrid=True, gridcolor='lightgray', autorange="reversed")
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig_non_infl, use_container_width=True)
                        
                        # Add contextual information
                        with st.expander("Understanding Morris Effects", expanded=False):
                            st.markdown("""
                            ### Interpreting Morris Effects
                            
                            **What are Morris Effects?**  
                            Morris effects measure how much each input variable influences the model output. 
                            
                            **How to interpret the values:**
                            - **Effect Value**: The absolute mean elementary effect, which quantifies the average impact of changing the variable.
                            - **Relative Effect (%)**: The effect as a percentage of the maximum effect observed across all variables.
                            
                            **Decision making:**
                            - Variables with very low effects (below the threshold) can often be fixed at nominal values without significantly affecting model outputs.
                            - This simplifies the model by reducing its dimensionality, making it easier to understand and computationally more efficient.
                            """)
                        
                        # Get and display recommended fixed values
                        var_indices = [idx for _, idx, _ in non_influential]
                        recommendations = get_recommended_fixed_values(problem, var_indices)
                        
                        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                        st.markdown('<h3 class="sub-header">Recommended Fixed Values</h3>', unsafe_allow_html=True)
                        st.markdown('<div class="info-box status-box">You can fix these variables at the following values in your model:</div>', unsafe_allow_html=True)
                        st.dataframe(recommendations)
                        
                        # Add explanation with LLM if available
                        if selected_language_model:
                            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                            st.markdown('<h3 class="sub-header">Model Simplification Insights</h3>', unsafe_allow_html=True)
                            
                            # Create a placeholder for the insights
                            insights_placeholder = st.empty()
                            
                            with st.spinner("Generating insights..."):
                                from utils.core_utils import call_groq_api
                                
                                # Prepare the prompt
                                prompt = f"""
                                Based on Morris sensitivity analysis, these variables have minimal impact on the model output:
                                {', '.join([name for name, _, _ in non_influential])}
                                
                                The model code is:
                                ```python
                                {current_code}
                                ```
                                
                                Explain in 2-3 paragraphs:
                                1. Why these variables might have low influence on this specific model
                                2. How the model could be simplified by fixing these variables
                                3. What physical or mathematical insights this provides about the system
                                """
                                
                                # Call the LLM
                                response = call_groq_api(prompt, model_name=selected_language_model)
                                
                                # Display the response in the placeholder
                                insights_placeholder.markdown('<div class="card" style="margin-top: 10px;">' + response + '</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="info-box status-box">No non-influential variables were identified. All variables appear to have significant effects on the output.</div>', unsafe_allow_html=True)
                        
                    st.markdown('<div class="success-box status-box">Morris analysis completed successfully!</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box status-box">Error in Morris analysis: {str(e)}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="warning-box status-box">Please try with different parameters or check your model implementation.</div>', unsafe_allow_html=True)
