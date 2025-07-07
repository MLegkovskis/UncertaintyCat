import numpy as np
import pandas as pd
import openturns as ot
# import matplotlib.pyplot as plt # Not directly used, seaborn/plotly are
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff # Not directly used in provided snippet, consider removal if not needed elsewhere
import plotly.subplots as sp 
import scipy.stats as stats
from utils.core_utils import call_groq_api # create_chat_interface not used here
from utils.constants import RETURN_INSTRUCTION
# import seaborn as sns # Not directly used in provided snippet, consider removal if not needed elsewhere
from typing import Dict, List, Tuple, Any, Optional, Union
# Unused imports: uuid, json, os, from groq import Groq

def compute_exploratory_data_analysis(data: pd.DataFrame, 
                                      N: int, # Number of samples, often len(data)
                                      model: callable, # The original model function
                                      problem: ot.Distribution, # For nominal points, input names, ranges
                                      model_code_str: str) -> dict:
    """
    Perform exploratory data analysis calculations.
    Includes model_code_str in the returned dictionary.
    """
    if data is None or data.empty:
        raise ValueError("No data available for Exploratory Data Analysis.")
    
    # Ensure N matches data if provided, otherwise use len(data)
    if N != len(data):
        st.warning(f"Provided N ({N}) does not match data length ({len(data)}). Using data length for N.")
        N = len(data)

    # Identify input and output columns
    # Assuming 'Y' or 'Y*' are outputs, others are inputs for EDA context.
    # If problem.getDescription() provides names, those are canonical for inputs.
    input_ot_names = problem.getDescription()
    if not input_ot_names or len(input_ot_names) != problem.getDimension():
        input_ot_names = [f"X{i+1}" for i in range(problem.getDimension())]

    # Match DataFrame columns to OpenTURNS input names if possible
    # This assumes data columns are named X1, X2... Y or similar
    input_columns_df = []
    for i in range(problem.getDimension()):
        # Try to find corresponding column in DataFrame (e.g., X1, X2 or full name)
        df_col_found = None
        if input_ot_names[i] in data.columns:
            df_col_found = input_ot_names[i]
        elif f"X{i+1}" in data.columns: # Fallback to X1, X2 pattern
            df_col_found = f"X{i+1}"
        
        if df_col_found:
            input_columns_df.append(df_col_found)
        # If no match, this input might not be in the EDA dataframe, or naming mismatch
        
    if not input_columns_df: # If we couldn't map any OT inputs to df columns
        input_columns_df = [col for col in data.columns if col not in ['Y'] and not col.startswith('Y')]
        if not input_columns_df:
             raise ValueError("Could not identify input columns in the provided DataFrame for EDA.")


    output_columns_df = [col for col in data.columns if col == 'Y' or col.startswith('Y')]
    if not output_columns_df:
        raise ValueError("No output columns (expected 'Y' or 'Y1', 'Y2', etc.) found in the data for EDA.")

    # Create display names for better readability in plots/tables
    display_names = {}
    # For inputs, use OpenTURNS description if available and map to df columns
    for i, ot_name in enumerate(input_ot_names):
        if i < len(input_columns_df): # Ensure we don't go out of bounds for df_cols
            df_col = input_columns_df[i]
            display_names[df_col] = ot_name if ot_name else f"Input {i+1}" # Use OT name or fallback
    # For any remaining df input columns not mapped from OT problem
    for col in input_columns_df:
        if col not in display_names:
            display_names[col] = col # Default to df column name

    for col in output_columns_df:
        if col == 'Y':
            display_names[col] = "Output"
        elif col.startswith('Y') and col[1:].isdigit():
            display_names[col] = f"Output {col[1:]}"
        else:
            display_names[col] = col # Default for unusually named output columns

    corr = data.corr()
    corr_viz = corr.copy()
    np.fill_diagonal(corr_viz.values, np.nan) # Use NaN for diagonal in heatmap visual

    # For text display on heatmap, round values and handle NaN diagonal
    corr_text_display = corr.copy()
    for r_idx in range(len(corr_text_display.index)):
        for c_idx in range(len(corr_text_display.columns)):
            if r_idx == c_idx:
                corr_text_display.iloc[r_idx, c_idx] = "" # Empty string for diagonal text
            else:
                corr_text_display.iloc[r_idx, c_idx] = f"{corr_text_display.iloc[r_idx, c_idx]:.2f}"
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_viz.values,
        x=[display_names.get(col, col) for col in corr.columns],
        y=[display_names.get(col, col) for col in corr.index],
        colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
        text=corr_text_display.values,
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate="Corr(%{y}, %{x}): %{z:.3f}<extra></extra>"
    ))
    fig_corr.update_layout(title="Correlation Matrix of All Variables", height=max(400, 30 * len(corr.columns)), width=max(500, 35 * len(corr.columns)), xaxis_showgrid=False, yaxis_showgrid=False, yaxis_autorange='reversed')

    combined_plots_data = {} # Store data for plots, not figures directly
    regression_stats_all = {}
    
    nominal_point = problem.getMean() # Used for centering cross-cuts

    for output_col_df in output_columns_df:
        output_display_name = display_names.get(output_col_df, output_col_df)
        current_output_plots_data = {}
        current_output_regression_stats = []

        for i, input_col_df in enumerate(input_columns_df):
            # Ensure index 'i' corresponds to the correct input in 'problem' if names differ
            # This requires a robust mapping from input_col_df to its index in problem.getDescription()
            try:
                # Assuming input_ot_names are the canonical names from problem.getDescription()
                # and input_columns_df are the corresponding names in the DataFrame.
                # This part needs careful alignment if names are not perfectly matched (e.g. X1 vs. actual name)
                # For now, assume 'i' aligns with problem's i-th input if input_columns_df came from it.
                # If input_columns_df was a generic scrape, problem_idx needs explicit mapping.
                # Simplified assumption: input_col_df corresponds to problem.getMarginal(i)
                # A more robust mapping would be needed if column names in `data` don't directly map to order in `problem`
                problem_idx = i # This is an assumption that order matches!
                if problem_idx >= problem.getDimension(): continue # Should not happen if input_columns_df is correct

                input_ot_name_current = input_ot_names[problem_idx] # Canonical name from problem
                input_display_name_current = display_names.get(input_col_df, input_col_df)
            except IndexError:
                continue # Skip if problem_idx is out of bounds for input_ot_names


            # 1D Cross Cut Data
            marginal_dist = problem.getMarginal(problem_idx)
            mean_val = marginal_dist.getMean()[0]
            std_val = marginal_dist.getStandardDeviation()[0]
            # Define range robustly, considering distribution bounds
            lower_bound_dist = marginal_dist.getRange().getLowerBound()[0] if marginal_dist.getRange().getLowerBound() else -np.inf
            upper_bound_dist = marginal_dist.getRange().getUpperBound()[0] if marginal_dist.getRange().getUpperBound() else np.inf
            
            x_min_cc = max(mean_val - 3 * std_val, lower_bound_dist) if not np.isinf(lower_bound_dist) else mean_val - 3 * std_val
            x_max_cc = min(mean_val + 3 * std_val, upper_bound_dist) if not np.isinf(upper_bound_dist) else mean_val + 3 * std_val
            if x_min_cc >= x_max_cc : # Handle cases like constant distributions
                 x_min_cc = mean_val - 1 if std_val == 0 else mean_val - abs(mean_val * 0.1) if mean_val !=0 else -1
                 x_max_cc = mean_val + 1 if std_val == 0 else mean_val + abs(mean_val * 0.1) if mean_val !=0 else 1
                 if x_min_cc == x_max_cc: x_max_cc +=1 # Ensure a range

            x_values_cc = np.linspace(x_min_cc, x_max_cc, 50) # Reduced points for faster cross-cuts
            y_values_cc = []
            for x_val_cc in x_values_cc:
                point_cc = ot.Point(nominal_point) # Start with nominal values
                point_cc[problem_idx] = x_val_cc   # Vary the current input
                
                model_output_cc = model(point_cc)
                # Handle model output to get the specific component for output_col_df
                # This assumes model output structure matches how output_columns_df were identified
                try:
                    if isinstance(model_output_cc, (ot.Point, list, tuple, np.ndarray)):
                        if len(output_columns_df) > 1: # If multiple Y columns in original df
                            output_df_idx = output_columns_df.index(output_col_df)
                            y_val_cc = model_output_cc[output_df_idx]
                        else: # Single 'Y' output expected from model
                            y_val_cc = model_output_cc[0] if hasattr(model_output_cc, '__getitem__') else model_output_cc
                    else: # Scalar
                        y_val_cc = float(model_output_cc)
                    y_values_cc.append(y_val_cc)
                except (IndexError, TypeError, ValueError):
                    y_values_cc.append(np.nan) # Or handle error more explicitly

            # Linear Regression Data from Monte Carlo samples
            slope_reg, intercept_reg, r_val_reg, p_val_reg, stderr_reg = stats.linregress(
                data[input_col_df], data[output_col_df]
            )
            current_output_regression_stats.append({
                'input_variable': input_display_name_current, 'output_variable': output_display_name,
                'slope': slope_reg, 'intercept': intercept_reg, 'r_value': r_val_reg,
                'r_squared': r_val_reg**2, 'p_value': p_val_reg, 'std_error': stderr_reg
            })
            
            current_output_plots_data[input_col_df] = { # Use df col name as key
                'input_display_name': input_display_name_current,
                'output_display_name': output_display_name,
                'cross_cut_x': x_values_cc.tolist(),
                'cross_cut_y': y_values_cc,
                'scatter_x': data[input_col_df].tolist(),
                'scatter_y': data[output_col_df].tolist(),
                'regression_line_x': [data[input_col_df].min(), data[input_col_df].max()],
                'regression_line_y': [slope_reg * data[input_col_df].min() + intercept_reg, slope_reg * data[input_col_df].max() + intercept_reg],
                'r_squared': r_val_reg**2
            }
        combined_plots_data[output_col_df] = current_output_plots_data
        regression_stats_all[output_col_df] = current_output_regression_stats

    # 2D Cross Cuts (Contour Plots) - Simplified: always for the first output column
    # and only if more than one input.
    cross_cuts_2d_data = {} # Store data for plots
    primary_output_col_df = output_columns_df[0] # Focus on the first output for 2D cuts
    primary_output_display_name = display_names.get(primary_output_col_df, primary_output_col_df)

    if problem.getDimension() >= 2:
        for i in range(problem.getDimension()):
            for j in range(i + 1, problem.getDimension()):
                input_ot_name_i = input_ot_names[i]
                input_ot_name_j = input_ot_names[j]

                marginal_i_cc2d = problem.getMarginal(i)
                marginal_j_cc2d = problem.getMarginal(j)

                mean_i_cc2d, std_i_cc2d = marginal_i_cc2d.getMean()[0], marginal_i_cc2d.getStandardDeviation()[0]
                lower_i_cc2d, upper_i_cc2d = (marginal_i_cc2d.getRange().getLowerBound() or [-np.inf])[0], (marginal_i_cc2d.getRange().getUpperBound() or [np.inf])[0]
                x_min_i_cc2d = max(mean_i_cc2d - 2*std_i_cc2d, lower_i_cc2d) if not np.isinf(lower_i_cc2d) else mean_i_cc2d - 2*std_i_cc2d
                x_max_i_cc2d = min(mean_i_cc2d + 2*std_i_cc2d, upper_i_cc2d) if not np.isinf(upper_i_cc2d) else mean_i_cc2d + 2*std_i_cc2d
                if x_min_i_cc2d >= x_max_i_cc2d: x_min_i_cc2d, x_max_i_cc2d = (mean_i_cc2d - 1 if std_i_cc2d ==0 else mean_i_cc2d*0.9 if mean_i_cc2d!=0 else -1), (mean_i_cc2d + 1 if std_i_cc2d ==0 else mean_i_cc2d*1.1 if mean_i_cc2d!=0 else 1)
                if x_min_i_cc2d == x_max_i_cc2d : x_max_i_cc2d+=1


                mean_j_cc2d, std_j_cc2d = marginal_j_cc2d.getMean()[0], marginal_j_cc2d.getStandardDeviation()[0]
                lower_j_cc2d, upper_j_cc2d = (marginal_j_cc2d.getRange().getLowerBound() or [-np.inf])[0], (marginal_j_cc2d.getRange().getUpperBound() or [np.inf])[0]
                x_min_j_cc2d = max(mean_j_cc2d - 2*std_j_cc2d, lower_j_cc2d) if not np.isinf(lower_j_cc2d) else mean_j_cc2d - 2*std_j_cc2d
                x_max_j_cc2d = min(mean_j_cc2d + 2*std_j_cc2d, upper_j_cc2d) if not np.isinf(upper_j_cc2d) else mean_j_cc2d + 2*std_j_cc2d
                if x_min_j_cc2d >= x_max_j_cc2d: x_min_j_cc2d, x_max_j_cc2d = (mean_j_cc2d - 1 if std_j_cc2d ==0 else mean_j_cc2d*0.9 if mean_j_cc2d!=0 else -1), (mean_j_cc2d + 1 if std_j_cc2d ==0 else mean_j_cc2d*1.1 if mean_j_cc2d!=0 else 1)
                if x_min_j_cc2d == x_max_j_cc2d : x_max_j_cc2d+=1


                n_points_cc2d = 20 # Reduced for performance
                x_vals_i_cc2d = np.linspace(x_min_i_cc2d, x_max_i_cc2d, n_points_cc2d)
                x_vals_j_cc2d = np.linspace(x_min_j_cc2d, x_max_j_cc2d, n_points_cc2d)
                
                X_grid_cc2d, Y_grid_cc2d = np.meshgrid(x_vals_i_cc2d, x_vals_j_cc2d)
                Z_grid_cc2d = np.full_like(X_grid_cc2d, np.nan)

                for r_idx in range(n_points_cc2d): # y-axis of grid (input j)
                    for c_idx in range(n_points_cc2d): # x-axis of grid (input i)
                        point_cc2d = ot.Point(nominal_point)
                        point_cc2d[i] = X_grid_cc2d[r_idx, c_idx] # Input i varies along columns (x-axis)
                        point_cc2d[j] = Y_grid_cc2d[r_idx, c_idx] # Input j varies along rows (y-axis)
                        
                        model_output_cc2d = model(point_cc2d)
                        try: # Extract first component for Z value
                            z_val = model_output_cc2d[0] if hasattr(model_output_cc2d, '__getitem__') else float(model_output_cc2d)
                            Z_grid_cc2d[r_idx, c_idx] = z_val
                        except (IndexError, TypeError, ValueError):
                            Z_grid_cc2d[r_idx, c_idx] = np.nan # Mark as NaN if extraction fails

                cross_cuts_2d_data[f"{input_ot_name_i}_vs_{input_ot_name_j}"] = {
                    'x_values': x_vals_i_cc2d.tolist(), 
                    'y_values': x_vals_j_cc2d.tolist(), 
                    'z_values': Z_grid_cc2d.tolist(),
                    'input_name_i': input_ot_name_i,
                    'input_name_j': input_ot_name_j,
                    'output_name': primary_output_display_name
                }
    
    summary_stats_df = data.describe().transpose()
    # For display names in summary_stats_df index
    summary_stats_df.index = summary_stats_df.index.map(lambda x: display_names.get(x,x))


    return {
        'model_code_str': model_code_str, # Pass through for AI insights
        'eda_dataframe': data, # Original data used for EDA
        'input_columns_df': input_columns_df, # Column names from DF identified as inputs
        'output_columns_df': output_columns_df, # Column names from DF identified as outputs
        'display_names_map': display_names, # Map from df col name to display name
        'correlation_matrix_raw': corr, # Raw pandas correlation matrix
        'fig_corr_heatmap_data': {'z': corr_viz.values.tolist(), 
                                  'x': [display_names.get(col, col) for col in corr.columns], 
                                  'y': [display_names.get(col, col) for col in corr.index],
                                  'text': corr_text_display.values.tolist()},
        'combined_plots_data': combined_plots_data, # Data for 1D cross-cuts & regressions
        'regression_stats_all': regression_stats_all, # Detailed regression stats
        'cross_cuts_2d_data': cross_cuts_2d_data, # Data for 2D contour plots
        'summary_statistics_df': summary_stats_df # df.describe()
    }

# --- Main EDA Function ---
def exploratory_data_analysis(data: pd.DataFrame, 
                              N: int, # Original number of MC samples, can be inferred from data
                              model: callable, # Model function
                              problem: ot.Distribution, # Problem definition
                              model_code_str: str, # For AI context
                              language_model: str = 'groq', 
                              display_results: bool = True) -> dict:
    """
    Perform and display exploratory data analysis.
    """
    results_placeholder = None
    if display_results:
        results_placeholder = st.empty()
        with results_placeholder.container():
            st.info("🚀 Starting Exploratory Data Analysis...")

    analysis_results = {}
    try:
        analysis_results = compute_exploratory_data_analysis(data, N, model, problem, model_code_str)
        
        llm_insights = None
        if language_model and model_code_str and analysis_results: # Check if model_code_str is available
            if display_results:
                with results_placeholder.container():
                    st.info("🧠 Generating AI insights for EDA... This may take a moment.")
            
            # generate_ai_insights will retrieve model_code_str from analysis_results
            llm_insights = generate_ai_insights_eda(analysis_results, language_model=language_model)
        
        analysis_results['ai_insights'] = llm_insights

        if display_results and results_placeholder:
            with results_placeholder.container():
                st.success("✅ Exploratory Data Analysis Completed!")
        
        st.session_state.exploratory_data_analysis_results = analysis_results
        
        if display_results:
            display_exploratory_data_analysis_results(analysis_results) # Pass language_model if display func uses it
            
        return analysis_results
    except Exception as e:
        error_message = f"Error in exploratory_data_analysis: {str(e)}"
        if display_results:
            if results_placeholder: 
                with results_placeholder.container(): st.error(error_message)
            else: st.error(error_message)
        else:
            print(error_message)
        
        analysis_results['error'] = error_message
        analysis_results['ai_insights'] = "AI insights skipped due to EDA error."
        return analysis_results


# --- AI Insight Generation (Specific to EDA) ---
def generate_ai_insights_eda(analysis_results: dict, language_model: str = 'groq') -> str:
    """
    Generate AI insights for exploratory data analysis results.
    Retrieves model_code_str from analysis_results.
    """
    try:
        model_code_str = analysis_results.get('model_code_str')
        # model_code_str can be optional for EDA insights, but good to include if available
        # If not essential, the prompt can be adapted. For now, let's assume it's preferred.

        data_df = analysis_results.get('eda_dataframe')
        summary_stats_df = analysis_results.get('summary_statistics_df')
        corr_matrix = analysis_results.get('correlation_matrix_raw')
        regression_stats_all = analysis_results.get('regression_stats_all', {}) # Dict by output_col
        input_cols = analysis_results.get('input_columns_df', [])
        output_cols = analysis_results.get('output_columns_df', [])
        display_names = analysis_results.get('display_names_map', {})

        summary_stats_md = summary_stats_df.to_markdown(floatfmt=".3g") if summary_stats_df is not None else "Summary statistics not available."
        
        # Prepare correlation insights (e.g., top correlations with outputs)
        corr_insights_str = "Key Correlations:\n"
        if corr_matrix is not None:
            for out_col in output_cols:
                if out_col in corr_matrix.index:
                    output_corr = corr_matrix.loc[input_cols, out_col].abs().sort_values(ascending=False)
                    corr_insights_str += f"- Top correlations with {display_names.get(out_col, out_col)}:\n"
                    for i_col, val in output_corr.head(3).items():
                        corr_insights_str += f"  - {display_names.get(i_col, i_col)}: {corr_matrix.loc[i_col, out_col]:.3f}\n"
        else:
            corr_insights_str = "Correlation matrix not available.\n"

        # Prepare regression insights (e.g., R-squared values)
        regression_insights_str = "Linear Regression (Input vs. Output) R-squared values:\n"
        for out_col, reg_list in regression_stats_all.items():
            regression_insights_str += f"- For Output '{display_names.get(out_col, out_col)}':\n"
            if reg_list:
                for reg_item in reg_list:
                    regression_insights_str += f"  - vs. Input '{reg_item.get('input_variable', 'Unknown')}': R² = {reg_item.get('r_squared', 0):.3f}\n"
            else:
                regression_insights_str += "  No regression data available for this output.\n"
        
        model_code_md = f"```python\n{model_code_str.strip()}\n```" if model_code_str else "Model code not provided for context."

        prompt = f"""{RETURN_INSTRUCTION}
You are an expert data scientist performing an Exploratory Data Analysis (EDA).
The analysis is based on Monte Carlo simulation data from the following model:
{model_code_md}

Input variables considered in EDA: {', '.join([display_names.get(c,c) for c in input_cols])}
Output variable(s) considered in EDA: {', '.join([display_names.get(c,c) for c in output_cols])}

**Summary Statistics of all variables (Inputs & Outputs):**
{summary_stats_md}

**{corr_insights_str}**

**{regression_insights_str}**

**Analysis Request:**
Please provide a concise yet comprehensive EDA report covering:

1.  **Data Overview & Quality:**
    * Briefly comment on the central tendency (e.g., mean, median from summary stats) and dispersion (e.g., std, IQR) for key input and output variables.
    * Are there any immediate signs of potential data quality issues, extreme outliers, or unusual distributions from the summary statistics?

2.  **Input-Output Relationships:**
    * Based on the provided key correlations and R-squared values from linear regressions, which input variables appear to have the strongest linear relationships with the output(s)?
    * Mention any inputs that show weak linear relationships (low correlation, low R²).
    * Cross-cut plots (not directly shown to you) visualize these relationships by varying one input while others are at nominals. How would you generally interpret the combination of scatter plots (from MC data) and these 1D cross-cut lines?

3.  **Input Inter-correlations:**
    * Briefly examine the correlation matrix (implicitly from `corr_insights_str` or by inferring from a full matrix if it were provided) for any strong correlations *between input variables themselves*. What could such inter-correlations imply for more advanced sensitivity analyses?

4.  **Potential Non-linearities or Interactions:**
    * If linear R-squared values are low for some inputs despite those inputs being varied in cross-cuts, what might this suggest about the nature of their influence (e.g., non-linearity)?
    * 2D cross-cut contour plots (not shown to you) explore interactions between pairs of inputs. What kind of patterns in such plots would typically indicate interaction effects?

5.  **Key Findings & Next Steps:**
    * Summarize 2-3 main findings from this EDA.
    * What further investigations or more specialized UQ analyses (e.g., variance-based sensitivity, distribution fitting on outputs) would you recommend based on these initial exploratory results?

Focus on actionable insights. Be specific where data allows.
"""
        model_name_for_api = language_model
        if not language_model or language_model.lower() == 'groq':
            model_name_for_api = "llama3-70b-8192"

        insights = call_groq_api(prompt, model_name=model_name_for_api)
        return insights

    except Exception as e:
        st.error(f"Error in generate_ai_insights_eda: {str(e)}")
        return "Error: AI insights for EDA could not be generated."

# --- Display Function ---
def display_exploratory_data_analysis_results(analysis_results: dict): # Removed language_model from signature
    """
    Display exploratory data analysis results.
    Figures are now generated on-the-fly from data in analysis_results.
    """
    st.header("📊 Exploratory Data Analysis")
    
    if not analysis_results or 'error' in analysis_results and analysis_results['error']:
        st.error(f"Cannot display EDA results: {analysis_results.get('error', 'Unknown error')}")
        # Display AI insights if they contain an error message
        if 'ai_insights' in analysis_results and "Error" in str(analysis_results.get('ai_insights')):
             st.subheader("🧠 AI-Generated Insights & Interpretation")
             st.warning(analysis_results['ai_insights'])
        return

    # --- Correlation Matrix Display ---
    st.subheader("Correlation Matrix")
    fig_corr_data = analysis_results.get('fig_corr_heatmap_data')
    if fig_corr_data:
        fig_corr = go.Figure(data=go.Heatmap(
            z=fig_corr_data['z'], x=fig_corr_data['x'], y=fig_corr_data['y'],
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            text=fig_corr_data['text'], texttemplate="%{text}", textfont={"size": 9},
            hovertemplate="Corr(%{y}, %{x}): %{z:.3f}<extra></extra>"
        ))
        fig_corr.update_layout(
            title="Correlation Matrix of All Variables", 
            height=max(400, 30 * len(fig_corr_data['x'])), # Dynamic height
            width=max(500, 45 * len(fig_corr_data['x'])),  # Dynamic width
            xaxis_showgrid=False, yaxis_showgrid=False, 
            yaxis_autorange='reversed',
            margin=dict(l=100, r=50, t=50, b=50) # Adjust margins
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Correlation matrix data not available.")

    # --- 1D Cross Cuts & Regression Plots ---
    st.subheader("Input-Output Relationships (1D Cross-Cuts & Regression)")
    st.markdown("Top plot: Output response when varying one input (others at nominal). Bottom plot: Scatter of Monte Carlo samples with linear regression.")

    combined_plots_data = analysis_results.get('combined_plots_data', {})
    output_columns_df_display = analysis_results.get('output_columns_df', [])
    
    if not combined_plots_data or not output_columns_df_display:
        st.info("No 1D cross-cut or regression plot data available.")
    elif len(output_columns_df_display) == 1:
        output_col_key = output_columns_df_display[0]
        plots_for_output = combined_plots_data.get(output_col_key, {})
        input_df_cols_for_output = list(plots_for_output.keys())
        if input_df_cols_for_output:
            input_tabs = st.tabs([plots_for_output[input_df_col]['input_display_name'] for input_df_col in input_df_cols_for_output])
            for i, input_df_col_tab in enumerate(input_df_cols_for_output):
                with input_tabs[i]:
                    plot_data = plots_for_output[input_df_col_tab]
                    fig_1d = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                              subplot_titles=[f"Cross Cut: {plot_data['input_display_name']} vs {plot_data['output_display_name']}",
                                                              f"Regression: {plot_data['input_display_name']} vs {plot_data['output_display_name']}"])
                    fig_1d.add_trace(go.Scatter(x=plot_data['cross_cut_x'], y=plot_data['cross_cut_y'], mode='lines', name='Cross Cut', line=dict(color='royalblue')), row=1, col=1)
                    fig_1d.add_trace(go.Scatter(x=plot_data['scatter_x'], y=plot_data['scatter_y'], mode='markers', name='MC Samples', marker=dict(opacity=0.5, color='grey')), row=2, col=1)
                    fig_1d.add_trace(go.Scatter(x=plot_data['regression_line_x'], y=plot_data['regression_line_y'], mode='lines', name=f'Lin. Reg. (R²={plot_data["r_squared"]:.2f})', line=dict(color='firebrick')), row=2, col=1)
                    fig_1d.update_layout(height=600, margin=dict(t=60,b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                    fig_1d.update_yaxes(title_text=plot_data['output_display_name'], row=1, col=1); fig_1d.update_yaxes(title_text=plot_data['output_display_name'], row=2, col=1)
                    fig_1d.update_xaxes(title_text=plot_data['input_display_name'], row=2, col=1)
                    st.plotly_chart(fig_1d, use_container_width=True)
    else: # Multiple outputs
        output_tabs_display = st.tabs([analysis_results['display_names_map'].get(col,col) for col in output_columns_df_display])
        for i, output_col_key_tab in enumerate(output_columns_df_display):
            with output_tabs_display[i]:
                plots_for_output_tab = combined_plots_data.get(output_col_key_tab, {})
                input_df_cols_for_output_tab = list(plots_for_output_tab.keys())
                if input_df_cols_for_output_tab:
                    input_tabs_inner = st.tabs([plots_for_output_tab[input_df_col]['input_display_name'] for input_df_col in input_df_cols_for_output_tab])
                    for j, input_df_col_tab_inner in enumerate(input_df_cols_for_output_tab):
                        with input_tabs_inner[j]:
                            plot_data_inner = plots_for_output_tab[input_df_col_tab_inner]
                            # (Reconstruct fig_1d as above for this specific input/output tab)
                            fig_1d_multi = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                                      subplot_titles=[f"Cross Cut: {plot_data_inner['input_display_name']} vs {plot_data_inner['output_display_name']}",
                                                                      f"Regression: {plot_data_inner['input_display_name']} vs {plot_data_inner['output_display_name']}"])
                            fig_1d_multi.add_trace(go.Scatter(x=plot_data_inner['cross_cut_x'], y=plot_data_inner['cross_cut_y'], mode='lines', name='Cross Cut', line=dict(color='royalblue')), row=1, col=1)
                            fig_1d_multi.add_trace(go.Scatter(x=plot_data_inner['scatter_x'], y=plot_data_inner['scatter_y'], mode='markers', name='MC Samples', marker=dict(opacity=0.5, color='grey')), row=2, col=1)
                            fig_1d_multi.add_trace(go.Scatter(x=plot_data_inner['regression_line_x'], y=plot_data_inner['regression_line_y'], mode='lines', name=f'Lin. Reg. (R²={plot_data_inner["r_squared"]:.2f})', line=dict(color='firebrick')), row=2, col=1)
                            fig_1d_multi.update_layout(height=600, margin=dict(t=60,b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
                            fig_1d_multi.update_yaxes(title_text=plot_data_inner['output_display_name'], row=1, col=1); fig_1d_multi.update_yaxes(title_text=plot_data_inner['output_display_name'], row=2, col=1)
                            fig_1d_multi.update_xaxes(title_text=plot_data_inner['input_display_name'], row=2, col=1)
                            st.plotly_chart(fig_1d_multi, use_container_width=True)


    # --- 2D Cross Cuts (Contour Plots) ---
    st.subheader("2D Cross-Cuts (Contour Plots for Primary Output)")
    st.markdown("Shows how the primary output changes when varying two inputs (others at nominals). Helps identify interaction effects.")
    cross_cuts_2d_data_display = analysis_results.get('cross_cuts_2d_data', {})
    if cross_cuts_2d_data_display:
        input_pairs = list(cross_cuts_2d_data_display.keys())
        if input_pairs:
            pair_tabs = st.tabs(input_pairs)
            for i, pair_key in enumerate(input_pairs):
                with pair_tabs[i]:
                    plot_data_2d = cross_cuts_2d_data_display[pair_key]
                    fig_2d = go.Figure(data=go.Contour(
                        z=plot_data_2d['z_values'], x=plot_data_2d['x_values'], y=plot_data_2d['y_values'],
                        colorscale='Viridis', contours=dict(showlabels=True, labelfont=dict(size=10)),
                        colorbar=dict(title=plot_data_2d['output_name'])
                    ))
                    fig_2d.update_layout(title=f"Output vs {plot_data_2d['input_name_i']} and {plot_data_2d['input_name_j']}",
                                         xaxis_title=plot_data_2d['input_name_i'], yaxis_title=plot_data_2d['input_name_j'],
                                         height=500, width=600, margin=dict(t=50))
                    st.plotly_chart(fig_2d, use_container_width=True)
        else:
            st.info("No 2D cross-cut data available (requires at least 2 input variables).")
    else:
        st.info("No 2D cross-cut data available.")

    # --- AI Insights ---
    ai_insights_display = analysis_results.get('ai_insights')
    if ai_insights_display:
        st.subheader("🧠 AI-Generated Insights & Interpretation")
        st.markdown(ai_insights_display)
    # Check if AI insights were expected but failed (e.g., model_code_str was missing from compute's return)
    elif analysis_results.get("model_code_str") is None and analysis_results.get("language_model") and not ai_insights_display and not ('error' in analysis_results and analysis_results['error']):
         st.warning("AI insights could not be generated because the model code was not available to the insight generation step.")
    elif analysis_results.get("language_model") and not ai_insights_display and not ('error' in analysis_results and analysis_results['error']):
         st.warning("AI insights were expected but not generated for EDA. This might be due to an API error or other issue.")


# --- Chat Context Function ---
def get_eda_context_for_chat(eda_results: dict) -> str:
    """
    Generate a formatted string of EDA results for chat context.
    """
    if not eda_results or ('error' in eda_results and eda_results['error']):
        return f"Exploratory Data Analysis results are not available due to an error: {eda_results.get('error', 'Unknown error')}"

    context = "\n\n### Exploratory Data Analysis Summary:\n"
    
    summary_df = eda_results.get('summary_statistics_df')
    if summary_df is not None and not summary_df.empty:
        context += "**Key Variable Statistics:**\n"
        # Provide a snippet of summary stats, e.g., for output columns
        output_cols_chat = eda_results.get('output_columns_df', [])
        for out_col_chat in output_cols_chat:
            out_disp_name_chat = eda_results.get('display_names_map', {}).get(out_col_chat, out_col_chat)
            if out_disp_name_chat in summary_df.index: # Check if display name is in index
                 stats_series = summary_df.loc[out_disp_name_chat]
                 context += f"- Output '{out_disp_name_chat}': Mean={stats_series.get('mean', 'N/A'):.3g}, Std={stats_series.get('std', 'N/A'):.3g}, Min={stats_series.get('min', 'N/A'):.3g}, Max={stats_series.get('max', 'N/A'):.3g}\n"
            elif out_col_chat in summary_df.index: # Fallback to original column name if display name not in index
                 stats_series = summary_df.loc[out_col_chat]
                 context += f"- Output '{out_disp_name_chat}': Mean={stats_series.get('mean', 'N/A'):.3g}, Std={stats_series.get('std', 'N/A'):.3g}, Min={stats_series.get('min', 'N/A'):.3g}, Max={stats_series.get('max', 'N/A'):.3g}\n"


    corr_matrix_chat = eda_results.get("correlation_matrix_raw")
    input_cols_chat = eda_results.get('input_columns_df', [])
    output_cols_chat = eda_results.get('output_columns_df', [])
    display_names_chat = eda_results.get('display_names_map', {})

    if corr_matrix_chat is not None and not corr_matrix_chat.empty and input_cols_chat and output_cols_chat:
        context += "\n**Top Input-Output Correlations (Pearson):**\n"
        for out_col_c in output_cols_chat:
            out_disp_c = display_names_chat.get(out_col_c, out_col_c)
            if out_col_c in corr_matrix_chat.index: # Check if output column exists in corr matrix index
                # Ensure input_cols_chat are also in corr_matrix.columns
                valid_input_cols_for_corr = [ic for ic in input_cols_chat if ic in corr_matrix_chat.columns]
                if valid_input_cols_for_corr:
                    correlations_with_output = corr_matrix_chat.loc[valid_input_cols_for_corr, out_col_c].abs().sort_values(ascending=False)
                    context += f"- With Output '{out_disp_c}':\n"
                    for i_col_c, val_c in correlations_with_output.head(2).items(): # Top 2
                        context += f"  - vs. {display_names_chat.get(i_col_c, i_col_c)}: {corr_matrix_chat.loc[i_col_c, out_col_c]:.3f}\n"
    
    ai_insights_context = eda_results.get("ai_insights")
    if ai_insights_context and isinstance(ai_insights_context, str):
        insight_lines_context = ai_insights_context.split('\n')
        summary_hint_context = [line for line in insight_lines_context if "summary:" in line.lower() or "key findings:" in line.lower() or "## Executive Summary" in line]
        if summary_hint_context:
            context += f"\n**AI Insights Snippet:**\n{summary_hint_context[0]}\n"
        elif len(insight_lines_context) > 2 and insight_lines_context[0].startswith("#"):
             non_header_lines_context = [line.strip() for line in insight_lines_context if line.strip() and not line.strip().startswith("#")]
             if non_header_lines_context: context += f"\n**AI Insights Snippet:**\n{non_header_lines_context[0]}...\n"
        else: 
            context += f"\n**AI Insights Snippet:**\n{ai_insights_context[:200]}...\n"
        context += "(Refer to full analysis for details)\n"
    
    return context
