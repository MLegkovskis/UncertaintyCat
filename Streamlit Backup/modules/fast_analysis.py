import numpy as np
import pandas as pd
import openturns as ot
from utils.core_utils import call_groq_api # create_chat_interface not used here
from utils.constants import RETURN_INSTRUCTION # Assuming this is correctly defined
import streamlit as st
import plotly.graph_objects as go
# import plotly.express as px # Not used directly in this snippet
# from plotly.subplots import make_subplots # Not used directly but good for future
from typing import Optional # Added for type hinting

def fast_sensitivity_analysis(model: ot.Function, 
                                      problem: ot.Distribution, 
                                      size: int = 400, 
                                      model_code_str: Optional[str] = None, 
                                      language_model: Optional[str] = None) -> dict:
    """
    Perform FAST sensitivity analysis computation.
    This version is based on your provided original working code for LLM interaction.
    """
    # Verify input types
    if not isinstance(model, ot.Function):
        raise TypeError("Model must be an OpenTURNS Function")
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise TypeError("Problem must be an OpenTURNS Distribution")
            
    # Get dimension from the model's input dimension
    dimension = model.getInputDimension()
    
    # Get variable names robustly
    variable_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        desc = marginal.getDescription()
        name = desc[0] if desc and desc[0] else f"X{i+1}" # Use X{i+1} if no description
        variable_names.append(name)
    
    # Create independent distribution for FAST
    marginals = [problem.getMarginal(i) for i in range(dimension)]
    independent_dist = ot.JointDistribution(marginals)
    
    sensitivityAnalysis = ot.FAST(model, independent_dist, size)
    firstOrderIndices = sensitivityAnalysis.getFirstOrderIndices()
    totalOrderIndices = sensitivityAnalysis.getTotalOrderIndices()
    
    indices_data = []
    for i, name in enumerate(variable_names):
        s_i = float(firstOrderIndices[i])
        t_i = float(totalOrderIndices[i])
        interaction_val = t_i - s_i
        # Ensure t_i is not zero or very close to zero before division
        interaction_pct = (interaction_val / t_i) * 100 if abs(t_i) > 1e-9 else 0.0
        
        indices_data.append({
            'Variable': name,
            'First Order (S_i)': s_i,
            'Total Order (T_i)': t_i,
            'Interaction (T_i - S_i)': interaction_val,
            'Interaction %': interaction_pct
        })
    
    indices_df = pd.DataFrame(indices_data)
    if not indices_df.empty: # Ensure DataFrame is not empty before sorting
        indices_df = indices_df.sort_values('Total Order (T_i)', ascending=False).reset_index(drop=True)
    
    # --- Plotting ---
    fig_bar = go.Figure()
    if not indices_df.empty:
        fig_bar.add_trace(go.Bar(
            x=indices_df['Variable'], y=indices_df['First Order (S_i)'],
            name='First Order ($S_i$)', marker_color='rgba(31, 119, 180, 0.8)',
            hovertemplate="<b>%{x}</b><br>First Order ($S_i$): %{y:.4f}<extra></extra>"
        ))
        fig_bar.add_trace(go.Bar(
            x=indices_df['Variable'], y=indices_df['Total Order (T_i)'],
            name='Total Order ($T_i$)', marker_color='rgba(214, 39, 40, 0.8)',
            hovertemplate="<b>%{x}</b><br>Total Order ($T_i$): %{y:.4f}<extra></extra>"
        ))
    fig_bar.update_layout(
        title_text='FAST Sensitivity Indices: First Order vs. Total Order',
        xaxis_title='Input Variables', yaxis_title='Sensitivity Index Value',
        barmode='group', template='plotly_white', height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig_breakdown = go.Figure()
    if not indices_df.empty:
        fig_breakdown.add_trace(go.Bar(
            x=indices_df['Variable'], y=indices_df['First Order (S_i)'],
            name='Direct Effect ($S_i$)', marker_color='rgba(31, 119, 180, 0.8)',
            hovertemplate="<b>%{x}</b><br>Direct Effect ($S_i$): %{y:.4f}<extra></extra>"
        ))
        fig_breakdown.add_trace(go.Bar(
            x=indices_df['Variable'], y=indices_df['Interaction (T_i - S_i)'],
            name='Interaction Effect ($T_i$ - $S_i$)', marker_color='rgba(255, 127, 14, 0.8)',
            hovertemplate="<b>%{x}</b><br>Interaction ($T_i$-$S_i$): %{y:.4f}<extra></extra>"
        ))
    fig_breakdown.update_layout(
        title_text='FAST Sensitivity Indices: Total Order Breakdown',
        xaxis_title='Input Variables', yaxis_title='Sensitivity Index Value',
        barmode='stack', template='plotly_white', height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # Generate LLM insights and explanation if requested
    llm_insights = None
    explanation = None # Initialize explanation
    default_explanation = """### FAST Sensitivity Analysis
The FAST method quantifies how each input variable contributes to the model output variance.
First order indices measure direct effects, while total order indices include interactions.
FAST requires input variables to be independent for its standard interpretation.
"""

    if model_code_str and language_model: # model_code_str is required for good insights
        # Prepare the prompt for insights (using your original correct f-string format)
        # Ensure firstOrderIndices and totalOrderIndices are available for the list comprehensions
        first_order_summary_str = ', '.join([f"{name}: {float(firstOrderIndices[i]):.4f}" for i, name in enumerate(variable_names)])
        total_order_summary_str = ', '.join([f"{name}: {float(totalOrderIndices[i]):.4f}" for i, name in enumerate(variable_names)])

        insights_prompt = f"""{RETURN_INSTRUCTION}
I've performed a FAST sensitivity analysis on the following model:
```python
{model_code_str}
```

The results show these first order indices:
{first_order_summary_str}

And these total order indices:
{total_order_summary_str}

Please provide 2-3 paragraphs of insights about:
1. Which variables have the most influence on the model output and why
2. The significance of any interaction effects observed (difference between Total and First order)
3. How these results could inform model simplification or further analysis
"""
        
        explanation_prompt = f"""{RETURN_INSTRUCTION}
Please provide a concise explanation of the FAST (Fourier Amplitude Sensitivity Test) method for sensitivity analysis.

Here is the technical information about the FAST method from OpenTURNS documentation:

FAST is a sensitivity analysis method based on the ANOVA decomposition of variance using Fourier expansion.
It works with a random vector X of independent components and recasts the representation as a function of a scalar parameter s.

The Fourier expansion of the model response is:
f(s) = Σ A_k cos(ks) + B_k sin(ks)

The first order indices are estimated by:
S_i = D_i/D = Σ(A_p^2 + B_p^2) / Σ(A_n^2 + B_n^2) 
(where summation for D_i is over multiples of fundamental frequency for X_i)

The total order indices are estimated by:
T_i = 1 - D_(-i)/D = 1 - Σ(A_k^2 + B_k^2) / Σ(A_n^2 + B_n^2)
(where summation for D_(-i) is over frequencies NOT multiples of fundamental frequency for X_i)

Where:
- D is the total variance (sum of all A_n^2 + B_n^2)
- D_i is the portion of variance from the i-th input
- D_(-i) is the variance due to all inputs except the i-th

Format your response in markdown with appropriate headers and bullet points. Keep it educational but accessible to non-experts. Explain what S_i and T_i mean practically and how to interpret their difference.
"""
        
        model_name_for_api = language_model
        if not language_model or language_model.lower() == 'groq':
            model_name_for_api = "llama3-70b-8192" # Your preferred default
        try:
            llm_insights = call_groq_api(insights_prompt, model_name=model_name_for_api)
        except Exception as e:
            llm_insights = f"Error generating AI insights: {str(e)}"
        try:
            explanation = call_groq_api(explanation_prompt, model_name=model_name_for_api)
        except Exception as e:
            # If AI explanation fails, append the error to the default static explanation
            explanation = f"{default_explanation}\n\nError generating detailed AI explanation: {str(e)}"
    else:
        explanation = default_explanation # Use default if no model_code_str or language_model
    
    return {
        'model_code_str': model_code_str, # Pass through for consistency
        'indices_df': indices_df,
        'fig_bar_indices': fig_bar, # Renamed for clarity in returned dict
        'fig_stacked_breakdown': fig_breakdown, # Renamed for clarity
        'explanation_text': explanation, # Use 'explanation_text' consistently
        'llm_insights_text': llm_insights # Use 'llm_insights_text' consistently
    }

# --- Main Entry Point Function (Corrected call to fast_sensitivity_analysis) ---
def fast_analysis(model: ot.Function, problem: ot.Distribution, size: int = 400, 
                  model_code_str: Optional[str] = None, language_model: Optional[str] = None, 
                  display_results: bool = True) -> dict:
    """
    Perform and optionally display FAST sensitivity analysis.
    This is the primary entry point for the FAST module.
    """
    results_placeholder = None
    if display_results:
        results_placeholder = st.empty()
        with results_placeholder.container(): # Use container to manage spinner
            st.info("🚀 Starting FAST Sensitivity Analysis...")
    
    analysis_results_dict = {} 
    try:
        # Corrected: Call fast_sensitivity_analysis directly
        analysis_results_dict = fast_sensitivity_analysis(
            model=model, problem=problem, size=size, 
            model_code_str=model_code_str, language_model=language_model
        )
        
        if display_results and results_placeholder:
            with results_placeholder.container():
                st.success("✅ FAST Sensitivity Analysis Completed!")
        
        st.session_state.fast_analysis_results = analysis_results_dict
        
        if display_results:
            display_fast_results(analysis_results_dict) 
            
        return analysis_results_dict
    
    except Exception as e:
        error_message = f"Critical error in FAST analysis workflow: {str(e)}"
        if display_results:
            if results_placeholder: 
                with results_placeholder.container(): st.error(error_message)
            else: st.error(error_message)
        else:
            print(error_message) # Log if not in Streamlit display
        
        # Populate a minimal error dictionary
        analysis_results_dict['error'] = error_message
        analysis_results_dict.setdefault('llm_insights_text', "AI insights skipped due to critical analysis error.")
        analysis_results_dict.setdefault('explanation_text', "Method explanation skipped due to critical analysis error.")
        analysis_results_dict.setdefault('indices_df', pd.DataFrame())
        analysis_results_dict.setdefault('fig_bar_indices', go.Figure())
        analysis_results_dict.setdefault('fig_stacked_breakdown', go.Figure())
        return analysis_results_dict

# --- Display Function (Robustified) ---
def display_fast_results(fast_results_data: dict):
    """
    Display FAST sensitivity analysis results in the Streamlit interface.
    """
    if not fast_results_data: # Handles if None was returned
        st.error("FAST analysis data is missing.")
        return
    
    # If error occurred during compute, it might be in the dict
    if 'error' in fast_results_data and fast_results_data['error']:
        # Error is usually displayed by the calling function's placeholder.
        # If critical data for display is missing, we might want to show a more specific error here
        # or just let the warnings for missing data below handle it.
        # For now, we'll proceed and let individual sections handle missing data.
        pass # The error in placeholder is primary.

    st.header("💨 FAST Sensitivity Analysis Results")

    explanation = fast_results_data.get('explanation_text')
    if explanation:
        st.markdown(explanation)
    else:
        st.markdown("FAST method explanation not available.") # Fallback
    
    indices_df_display = fast_results_data.get('indices_df')
    if indices_df_display is None or indices_df_display.empty:
        st.warning("Sensitivity indices data is not available for display.")
        # Attempt to display AI insights even if table data is missing, if insights exist
        llm_insights_err_disp = fast_results_data.get('llm_insights_text')
        if llm_insights_err_disp:
            st.subheader("🤖 AI-Generated Expert Analysis")
            st.markdown(llm_insights_err_disp)
        return # Stop further display if no indices

    st.subheader("📊 Sensitivity Indices & Metrics")
    most_influential_var_display = "N/A"
    most_influential_idx_display = np.nan
    if not indices_df_display.empty: # Check again after initial guard
        most_influential_var_display = indices_df_display.iloc[0]['Variable']
        most_influential_idx_display = indices_df_display.iloc[0]['Total Order (T_i)']
    
    sum_first_order_display = indices_df_display['First Order (S_i)'].sum()
    sum_total_order_display = indices_df_display['Total Order (T_i)'].sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("👑 Most Influential (by $T_i$)", most_influential_var_display, f"{most_influential_idx_display:.3f}")
    with col2:
        st.metric("∑ $S_i$ (Sum of First Order)", f"{sum_first_order_display:.3f}", help="If ΣSᵢ is much less than 1 (and less than ΣTᵢ), interactions are significant.")
    with col3:
        st.metric("∑ $T_i$ (Sum of Total Order)", f"{sum_total_order_display:.3f}", help="Can be > 1. Each Tᵢ includes all interactions for that variable.")

    st.markdown("#### Detailed Numerical Results")
    required_cols_disp_table = ['Variable', 'First Order (S_i)', 'Total Order (T_i)', 'Interaction (T_i - S_i)', 'Interaction %']
    # Check if all required columns exist in the DataFrame
    if all(col in indices_df_display.columns for col in required_cols_disp_table):
        df_to_show_disp_table = indices_df_display[required_cols_disp_table].copy()
        for col_fmt_table in ['First Order (S_i)', 'Total Order (T_i)', 'Interaction (T_i - S_i)']:
            df_to_show_disp_table[col_fmt_table] = pd.to_numeric(df_to_show_disp_table[col_fmt_table], errors='coerce').map('{:.4f}'.format)
        df_to_show_disp_table['Interaction %'] = pd.to_numeric(df_to_show_disp_table['Interaction %'], errors='coerce').map('{:.2f}%'.format)
        st.dataframe(df_to_show_disp_table, hide_index=True, width='stretch')
    else:
        st.warning("Detailed numerical results table is missing some expected columns. Displaying available data.")
        st.dataframe(indices_df_display, hide_index=True, width='stretch')


    st.subheader("🎨 Sensitivity Visualizations")
    fig_bar_display_val_plot = fast_results_data.get('fig_bar_indices')
    if isinstance(fig_bar_display_val_plot, go.Figure) and fig_bar_display_val_plot.data:
        st.markdown("##### First Order vs. Total Order Indices")
        st.markdown("Compares direct effects (First Order, $S_i$) vs. total effects including all interactions (Total Order, $T_i$) for each input variable.")
        st.plotly_chart(fig_bar_display_val_plot, width='stretch')
    else: st.warning("Bar chart for FAST indices is not available.")
    
    fig_breakdown_display_val_plot = fast_results_data.get('fig_stacked_breakdown')
    if isinstance(fig_breakdown_display_val_plot, go.Figure) and fig_breakdown_display_val_plot.data:
        st.markdown("##### Total Order Index Breakdown")
        st.markdown("Shows how Total Order indices ($T_i$) are composed of Direct Effects (First Order, $S_i$) and Interaction Effects ($T_i$ - $S_i$). The sum of these two bars for each variable equals its $T_i$.")
        st.plotly_chart(fig_breakdown_display_val_plot, width='stretch')
    else: st.warning("Breakdown chart for FAST indices is not available.")
        
    if pd.notna(sum_first_order_display): # Check if sum is a valid number
        if sum_first_order_display < 0.7 and sum_first_order_display > 0.01 : 
            st.info("ℹ️ **Significant Interaction Effects Likely:** The sum of First-Order indices (ΣSᵢ) is noticeably less than 1. This suggests that interactions between input variables account for a substantial portion of the model output's variance. The model is likely non-additive.")
        elif sum_first_order_display > 0.95 :
            st.success("✅ **Low Interaction Effects Likely:** The sum of First-Order indices (ΣSᵢ) is close to 1. This implies that the model's output variance is primarily driven by the direct, additive effects of individual variables, with minimal contributions from interactions.")
        else: 
            st.markdown(f"The sum of First-Order indices (ΣSᵢ = {sum_first_order_display:.3f}) provides an indication of the importance of interactions. Values much less than 1 point to significant interactions.")

    llm_insights_display_final_val = fast_results_data.get('llm_insights_text')
    if llm_insights_display_final_val:
        st.subheader("🤖 AI-Generated Expert Analysis")
        st.markdown(llm_insights_display_final_val)
    # Check if AI was expected but some prerequisite like model_code_str might have been missing when compute was called
    elif fast_results_data.get("model_code_str") and fast_results_data.get("language_model") and not llm_insights_display_final_val \
         and not ('error' in fast_results_data and fast_results_data['error']): # Only show if no major error and insights were expected
        st.warning("AI insights were expected but not generated. This might be due to an API error during generation or if the model code was not available.")

# --- Chat Context Function ---
def get_fast_context_for_chat(fast_results_data: dict) -> str:
    """
    Generate a formatted string of FAST analysis results for chat context.
    """
    if not fast_results_data or ('error' in fast_results_data and fast_results_data['error']):
        return f"FAST analysis results are not available due to error: {fast_results_data.get('error', 'Unknown error')}"

    context = "\n\n### FAST Sensitivity Analysis Summary:\n"
    indices_df_chat_val_ctx = fast_results_data.get("indices_df")
    if indices_df_chat_val_ctx is not None and not indices_df_chat_val_ctx.empty:
        context += "Top 3 influential variables (by Total Order Index Tᵢ):\n"
        for _, row in indices_df_chat_val_ctx.head(3).iterrows():
            context += (
                f"- **{row['Variable']}**: Tᵢ={row['Total Order (T_i)']:.3f} "
                f"(Sᵢ={row['First Order (S_i)']:.3f}, Interaction={row['Interaction (T_i - S_i)']:.3f})\n"
            )
        sum_s_i_chat_val_ctx = indices_df_chat_val_ctx['First Order (S_i)'].sum()
        context += f"Sum of First-Order Indices (ΣSᵢ): {sum_s_i_chat_val_ctx:.3f} (If << 1, strong interactions).\n"
    else:
        context += "Sensitivity indices data not available.\n"
    
    ai_insights_chat_val_ctx_val = fast_results_data.get('llm_insights_text')
    if ai_insights_chat_val_ctx_val and isinstance(ai_insights_chat_val_ctx_val, str) and "Error" not in ai_insights_chat_val_ctx_val :
        insight_lines_chat_val_ctx_val = ai_insights_chat_val_ctx_val.split('\n')
        # Try to get a more meaningful snippet
        snippet_lines_chat = []
        collecting = False
        for line in insight_lines_chat_val_ctx_val:
            stripped_line = line.strip()
            if stripped_line.lower().startswith("1.") or "influential variables" in stripped_line.lower() or "key findings" in stripped_line.lower() or "summary" in stripped_line.lower():
                collecting = True
            if collecting and stripped_line:
                snippet_lines_chat.append(stripped_line)
                if len(snippet_lines_chat) >= 3: # Grab up to 3 lines of the identified section
                    break
        
        if snippet_lines_chat:
            context += f"\n**AI Insights Snippet:**\n" + "\n".join(snippet_lines_chat) + "...\n"
        elif len(insight_lines_chat_val_ctx_val) > 1 : # Fallback to first few non-empty lines
             non_empty_lines_chat_ctx_val = [line.strip() for line in insight_lines_chat_val_ctx_val if line.strip()][:2]
             if non_empty_lines_chat_ctx_val:
                 context += f"\n**AI Insights Snippet:**\n" + "\n".join(non_empty_lines_chat_ctx_val) + "...\n"
        else: 
            context += f"\n**AI Insights Snippet:**\n{ai_insights_chat_val_ctx_val[:250]}...\n"
        context += "(Refer to full analysis for complete AI interpretation)\n"
        
    return context
