import numpy as np
import pandas as pd
import openturns as ot
import streamlit as st
from utils.constants import RETURN_INSTRUCTION # Assuming this is correctly defined
from utils.core_utils import call_groq_api # Assuming this is correctly defined
from utils.model_utils import get_ot_model # Assuming this is correctly defined
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Helper Functions ---
def get_distribution_code_string(dist: ot.Distribution) -> str:
    """
    Generate a string representing the OpenTURNS distribution code.
    """
    try:
        ot_class_name = dist.getName()
        if not ot_class_name or ot_class_name == "Distribution": # Fallback
            ot_class_name = dist.getClassName().replace("Implementation", "").replace("Factory", "")
    except Exception:
        ot_class_name = dist.getClassName().replace("Implementation", "").replace("Factory", "")
    
    params = dist.getParameter()
    param_str = ", ".join([repr(p) for p in params])
    formula_str = f"ot.{ot_class_name}({param_str})"
    return formula_str

# --- Core Computation Function ---
def compute_expectation_convergence_analysis(model: callable, problem: ot.Distribution, 
                                             model_code_str: str = None, 
                                             N_samples: int = 10000) -> dict:
    """
    Perform expectation convergence analysis calculations for a univariate model output.
    Includes Q-Q plot summary statistics.
    """
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Input 'problem' must be an OpenTURNS Distribution object.")
    
    ot_model = get_ot_model(model) 
    input_random_vector = ot.RandomVector(problem)
    output_random_vector = ot.CompositeRandomVector(ot_model, input_random_vector)

    if output_random_vector.getDimension() != 1:
        raise ValueError(
            f"The model for expectation analysis must be univariate (single output). "
            f"Detected output dimension: {output_random_vector.getDimension()}."
        )

    expectation_algo = ot.ExpectationSimulationAlgorithm(output_random_vector)
    expectation_algo.setMaximumOuterSampling(N_samples)
    expectation_algo.setBlockSize(1) 
    expectation_algo.setCoefficientOfVariationCriterionType("NONE") 
    
    expectation_algo.run()
    result = expectation_algo.getResult()

    graph_mean_conv = expectation_algo.drawExpectationConvergence()
    data_mean_conv = graph_mean_conv.getDrawable(0).getData()
    
    sample_sizes_algo = np.array([s[0] for s in data_mean_conv[:, 0]])
    mean_estimates_algo = np.array([m[0] for m in data_mean_conv[:, 1]])

    std_error_of_final_mean_algo = result.getStandardDeviation()[0] if result.getStandardDeviation() else np.nan
    N_final_samples_algo = sample_sizes_algo[-1] if sample_sizes_algo.size > 0 else 0
    
    standard_errors_at_k = np.full_like(sample_sizes_algo, np.nan, dtype=float)
    if N_final_samples_algo > 0 and np.all(sample_sizes_algo > 0) and not np.isnan(std_error_of_final_mean_algo):
        standard_errors_at_k = std_error_of_final_mean_algo * np.sqrt(N_final_samples_algo / sample_sizes_algo)

    z_value = 1.96
    lower_bounds_mean_algo = mean_estimates_algo - z_value * standard_errors_at_k
    upper_bounds_mean_algo = mean_estimates_algo + z_value * standard_errors_at_k
    
    actual_samples_run_by_algo = int(N_final_samples_algo) 
    
    Y_values = np.array([])
    if actual_samples_run_by_algo > 0:
        input_sample_dist_analysis = problem.getSample(actual_samples_run_by_algo)
        output_sample_ot_dist_analysis = ot_model(input_sample_dist_analysis)
        Y_values = np.array(output_sample_ot_dist_analysis).flatten()

    mean_Y_sample, std_Y_sample, se_Y_sample_mean, skewness_Y_sample, kurtosis_Y_sample, q1, q3, iqr_Y_sample = [np.nan] * 8
    conf_int_Y_sample = [np.nan, np.nan]

    if len(Y_values) > 0:
        mean_Y_sample = np.mean(Y_values)
        if len(Y_values) >= 2:
            std_Y_sample = np.std(Y_values, ddof=1)
            se_Y_sample_mean = std_Y_sample / np.sqrt(len(Y_values))
            conf_int_Y_sample = [mean_Y_sample - z_value * se_Y_sample_mean, mean_Y_sample + z_value * se_Y_sample_mean]
        skewness_Y_sample = stats.skew(Y_values)
        kurtosis_Y_sample = stats.kurtosis(Y_values, fisher=True)
        q1, q3 = np.percentile(Y_values, [25, 75])
        iqr_Y_sample = q3 - q1
    
    std_dev_estimates_cumulative = np.full_like(sample_sizes_algo, np.nan, dtype=float)
    min_samples_for_std = 2
    if len(Y_values) > 0:
        for i, n_k_from_algo in enumerate(sample_sizes_algo):
            n_k_int = min(int(n_k_from_algo), len(Y_values))
            if n_k_int >= min_samples_for_std:
                std_dev_estimates_cumulative[i] = np.std(Y_values[:n_k_int], ddof=1)
            elif i > 0 and not np.isnan(std_dev_estimates_cumulative[i-1]):
                std_dev_estimates_cumulative[i] = std_dev_estimates_cumulative[i-1]
    
    fit_df = pd.DataFrame()
    best_fitted_distribution_obj = None
    best_distribution_name_fitted = "None"
    ot_code_best_distribution = "None"
    best_params_fitted = []

    if len(Y_values) >= min_samples_for_std :
        try:
            from modules.distribution_fitting import get_distribution_factories, fit_distribution
            factories = get_distribution_factories().get("Continuous Univariate", [])
            Y_values_for_fitting = Y_values
            target_fitting_samples = 1000 
            if len(Y_values_for_fitting) < target_fitting_samples and len(Y_values_for_fitting) > 0:
                additional_needed = target_fitting_samples - len(Y_values_for_fitting)
                try:
                    additional_input_sample = problem.getSample(additional_needed)
                    additional_output_sample = ot_model(additional_input_sample)
                    additional_Y = np.array(additional_output_sample).flatten()
                    Y_values_for_fitting = np.concatenate([Y_values_for_fitting, additional_Y])
                except Exception as e_aug:
                    st.warning(f"Could not augment samples for distribution fitting: {str(e_aug)}.")
            
            if len(Y_values_for_fitting) >= min_samples_for_std:
                fitting_results_list = []
                for factory in factories:
                    try:
                        dist_name = factory.getClassName().replace("Factory", "")
                        fitted_dist, stats_dict = fit_distribution(Y_values_for_fitting, factory)
                        if fitted_dist and stats_dict:
                            fitting_results_list.append({
                                'Distribution': dist_name, 'Parameters': list(fitted_dist.getParameter()),
                                'AIC': stats_dict.get("AIC", float('inf')), 'BIC': stats_dict.get("BIC", float('inf')),
                                'KS_Statistic': stats_dict.get("KS_statistic", float('nan')),
                                'KS_pvalue': stats_dict.get("KS_pvalue", float('nan')),
                                'OT_Fitted_Distribution': fitted_dist
                            })
                    except Exception: continue
                if fitting_results_list:
                    fit_df = pd.DataFrame(fitting_results_list)
                    if not fit_df.empty and 'AIC' in fit_df.columns and fit_df['AIC'].notna().any():
                        fit_df = fit_df.sort_values('AIC', ascending=True).reset_index(drop=True)
                        if not fit_df.empty:
                            best_fit_series = fit_df.iloc[0]
                            best_distribution_name_fitted = best_fit_series['Distribution']
                            best_fitted_distribution_obj = best_fit_series['OT_Fitted_Distribution']
                            ot_code_best_distribution = get_distribution_code_string(best_fitted_distribution_obj)
                            best_params_fitted = best_fit_series['Parameters']
        except ImportError: st.warning("Distribution fitting module not found. Skipping.")
        except Exception as e_fit: st.warning(f"An error occurred during distribution fitting: {str(e_fit)}")

    # --- Q-Q Plot Summary Statistics Calculation ---
    qq_plot_summary = {
        'correlation': np.nan,
        'lower_tail_deviation_avg': np.nan,
        'middle_section_deviation_avg': np.nan,
        'upper_tail_deviation_avg': np.nan,
        'representative_points': [] 
    }
    osm_qq, osr_qq = np.array([]), np.array([]) 
    if len(Y_values) >= 20: 
        try:
            osm_qq, osr_qq = stats.probplot(Y_values, dist=stats.norm, fit=False)
            if len(osm_qq) >= 2 and len(osr_qq) >=2:
                 qq_plot_summary['correlation'] = np.corrcoef(osm_qq, osr_qq)[0, 1]

            deviations_qq = osr_qq - osm_qq
            n_qq_pts = len(osr_qq)
            idx_25_qq = int(n_qq_pts * 0.25)
            idx_75_qq = int(n_qq_pts * 0.75)

            if idx_25_qq > 0:
                qq_plot_summary['lower_tail_deviation_avg'] = np.mean(deviations_qq[:idx_25_qq])
            if (idx_75_qq - idx_25_qq) > 0:
                qq_plot_summary['middle_section_deviation_avg'] = np.mean(deviations_qq[idx_25_qq:idx_75_qq])
            if (n_qq_pts - idx_75_qq) > 0:
                qq_plot_summary['upper_tail_deviation_avg'] = np.mean(deviations_qq[idx_75_qq:])
            
            percentiles_to_sample_qq = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            sample_q_values_qq = np.percentile(Y_values, [p * 100 for p in percentiles_to_sample_qq])
            theoretical_q_values_for_percentiles_qq = stats.norm.ppf(percentiles_to_sample_qq)
            
            qq_plot_summary['representative_points'] = [
                (f"{p*100:.0f}%", float(t_q), float(s_q)) 
                for p, t_q, s_q in zip(percentiles_to_sample_qq, theoretical_q_values_for_percentiles_qq, sample_q_values_qq)
            ]
        except Exception as e_qq_stats:
            st.warning(f"Could not compute detailed Q-Q statistics: {e_qq_stats}")

    # --- Plotting ---
    fig_mean_convergence = make_subplots(rows=1, cols=2, subplot_titles=("Mean Convergence (Linear X-axis)", "Mean Convergence (Log X-axis)"), horizontal_spacing=0.15)
    if sample_sizes_algo.size > 0:
        fig_mean_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=mean_estimates_algo, mode='lines', name='Mean Estimate', line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig_mean_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=upper_bounds_mean_algo, mode='lines', line=dict(width=0), showlegend=False, name='Upper CI'), row=1, col=1)
        fig_mean_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=lower_bounds_mean_algo, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31,119,180,0.2)', name='95% CI'), row=1, col=1)
        fig_mean_convergence.add_trace(go.Scatter(x=[sample_sizes_algo[-1]], y=[mean_estimates_algo[-1]], mode='markers', name='Final Algo. Mean', marker=dict(color='red', size=8, symbol='x')), row=1, col=1)
        fig_mean_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=mean_estimates_algo, mode='lines', line=dict(color='#1f77b4', width=2), showlegend=False), row=1, col=2)
        fig_mean_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=upper_bounds_mean_algo, mode='lines', line=dict(width=0), showlegend=False), row=1, col=2)
        fig_mean_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=lower_bounds_mean_algo, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(31,119,180,0.2)', showlegend=False), row=1, col=2)
        fig_mean_convergence.add_trace(go.Scatter(x=[sample_sizes_algo[-1]], y=[mean_estimates_algo[-1]], mode='markers', marker=dict(color='red', size=8, symbol='x'), showlegend=False), row=1, col=2)
    fig_mean_convergence.update_layout(title_text="Monte Carlo Convergence of the Mean", height=450, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(b=50, t=60))
    fig_mean_convergence.update_xaxes(title_text="Number of Samples", type="linear", row=1, col=1); fig_mean_convergence.update_xaxes(title_text="Number of Samples", type="log", row=1, col=2)
    fig_mean_convergence.update_yaxes(title_text="Mean Estimate", row=1, col=1); fig_mean_convergence.update_yaxes(title_text="Mean Estimate", row=1, col=2)

    fig_std_convergence = make_subplots(rows=1, cols=2, subplot_titles=("Std Dev Convergence (Linear X-axis)", "Std Dev Convergence (Log X-axis)"), horizontal_spacing=0.15)
    if sample_sizes_algo.size > 0 and std_dev_estimates_cumulative.size == sample_sizes_algo.size:
        fig_std_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=std_dev_estimates_cumulative, mode='lines', name='Std Dev Estimate (Cumulative)', line=dict(color='#ff7f0e', width=2)), row=1, col=1)
        final_std_dev_val = std_dev_estimates_cumulative[-1] if std_dev_estimates_cumulative.size > 0 and not np.isnan(std_dev_estimates_cumulative[-1]) else np.nan
        fig_std_convergence.add_trace(go.Scatter(x=[sample_sizes_algo[-1]], y=[final_std_dev_val], mode='markers', name='Final Std Dev (Cumulative)', marker=dict(color='darkred', size=8, symbol='x')), row=1, col=1)
        fig_std_convergence.add_trace(go.Scatter(x=sample_sizes_algo, y=std_dev_estimates_cumulative, mode='lines', line=dict(color='#ff7f0e', width=2), showlegend=False), row=1, col=2)
        fig_std_convergence.add_trace(go.Scatter(x=[sample_sizes_algo[-1]], y=[final_std_dev_val], mode='markers', marker=dict(color='darkred', size=8, symbol='x'), showlegend=False), row=1, col=2)
    fig_std_convergence.update_layout(title_text="Monte Carlo Convergence of Standard Deviation (Cumulative from Sample)", height=450, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(b=50, t=60))
    fig_std_convergence.update_xaxes(title_text="Number of Samples", type="linear", row=1, col=1); fig_std_convergence.update_xaxes(title_text="Number of Samples", type="log", row=1, col=2)
    fig_std_convergence.update_yaxes(title_text="Standard Deviation Estimate", row=1, col=1); fig_std_convergence.update_yaxes(title_text="Standard Deviation Estimate", row=1, col=2)

    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("Output Distribution & Best Fit", "Q-Q Plot (vs. Normal)"))
    if len(Y_values) > 0: # osm_qq, osr_qq are defined in Q-Q stats section
        fig_dist.add_trace(go.Histogram(x=Y_values, histnorm='probability density', name='Empirical PDF', marker_color='lightblue', opacity=0.7), row=1, col=1)
        display_name_for_plot_legend = "None Fitted"
        if best_fitted_distribution_obj:
            display_name_for_plot_legend = best_distribution_name_fitted
            x_pdf_plot = np.linspace(Y_values.min(), Y_values.max(), 200)
            try:
                y_pdf_plot = np.array([best_fitted_distribution_obj.computePDF(ot.Point([val])) for val in x_pdf_plot])
                fig_dist.add_trace(go.Scatter(x=x_pdf_plot, y=y_pdf_plot, mode='lines', name=f'Fit: {display_name_for_plot_legend}', line=dict(color='red', width=2)), row=1, col=1)
            except Exception: pass
        
        if osm_qq.size > 0 and osr_qq.size > 0 : 
            fig_dist.add_trace(go.Scatter(x=osm_qq, y=osr_qq, mode='markers', name='Data Quantiles', marker_color='blue'), row=1, col=2)
            min_qq_plot, max_qq_plot = min(osm_qq.min(), osr_qq.min()), max(osm_qq.max(), osr_qq.max())
            fig_dist.add_trace(go.Scatter(x=[min_qq_plot, max_qq_plot], y=[min_qq_plot, max_qq_plot], mode='lines', name='y=x (Normal Ref.)', line=dict(color='red', dash='dash')), row=1, col=2)
    fig_dist.update_layout(title_text="Analysis of Output Distribution Shape", height=450, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(b=50, t=60))
    fig_dist.update_xaxes(title_text="Output Value", row=1, col=1); fig_dist.update_yaxes(title_text="Density", row=1, col=1)
    fig_dist.update_xaxes(title_text="Theoretical Quantiles (Normal)", row=1, col=2); fig_dist.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    final_algo_mean_val = result.getExpectationEstimate()[0] if result.getExpectationEstimate() else np.nan
    expectation_dist_algo = result.getExpectationDistribution()
    final_algo_ci_lower = expectation_dist_algo.computeQuantile(0.025)[0] if expectation_dist_algo else np.nan
    final_algo_ci_upper = expectation_dist_algo.computeQuantile(0.975)[0] if expectation_dist_algo else np.nan
    variance_estimate_algo = result.getVarianceEstimate()[0] if result.getVarianceEstimate() else np.nan
    estimated_output_std_dev_algo = np.sqrt(variance_estimate_algo) if not np.isnan(variance_estimate_algo) else np.nan

    summary_convergence_df = pd.DataFrame({
        'Metric': ['Final Mean Estimate (Algo.)', 'Std Err of Mean (Algo.)', '95% CI for Mean (Algo.)', 'Samples Run by Algo', 'Est. Output Std Dev (σY_hat Algo.)'],
        'Value': [
            f"{final_algo_mean_val:.5g}", f"{std_error_of_final_mean_algo:.5g}",
            f"[{final_algo_ci_lower:.5g}, {final_algo_ci_upper:.5g}]",
            f"{actual_samples_run_by_algo}", f"{estimated_output_std_dev_algo:.5g}"
        ]})
    summary_distribution_df = pd.DataFrame({
        'Statistic': ['Mean (Sample)', 'Std Dev (Sample, ddof=1)', '95% CI (Sample Mean)', 'Skewness (Sample)', 'Kurtosis (Fisher, Sample)', 'IQR (Sample)', 'Best Fit (AIC)'],
        'Value': [
            f"{mean_Y_sample:.5g}" if not np.isnan(mean_Y_sample) else "N/A", 
            f"{std_Y_sample:.5g}" if not np.isnan(std_Y_sample) else "N/A", 
            f"[{conf_int_Y_sample[0]:.5g}, {conf_int_Y_sample[1]:.5g}]" if not np.isnan(conf_int_Y_sample[0]) else "N/A",
            f"{skewness_Y_sample:.3f}" if not np.isnan(skewness_Y_sample) else "N/A", 
            f"{kurtosis_Y_sample:.3f}" if not np.isnan(kurtosis_Y_sample) else "N/A", 
            f"{iqr_Y_sample:.5g}" if not np.isnan(iqr_Y_sample) else "N/A",
            f"{best_distribution_name_fitted if best_distribution_name_fitted != 'None' else 'Not fitted'}"
        ]})
    quantiles_list_summary = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantile_values_summary = np.quantile(Y_values, quantiles_list_summary) if len(Y_values) > 0 else [np.nan]*len(quantiles_list_summary)
    quantiles_df_summary = pd.DataFrame({'Quantile': [f"{q*100:.0f}%" for q in quantiles_list_summary], 'Value': [f"{val:.5g}" if not np.isnan(val) else "N/A" for val in quantile_values_summary]})

    input_params_list_summary = []
    if problem:
        for i in range(problem.getDimension()):
            marginal = problem.getMarginal(i)
            desc = marginal.getDescription()
            var_name = desc[0] if desc and desc[0] else f"X{i+1}"
            input_params_list_summary.append({
                'Variable': var_name, 'Distribution': marginal.getClassName().replace("Implementation",""), 
                'Parameters': [f"{p:.4g}" for p in marginal.getParameter()]
            })
    inputs_summary_df_prompt = pd.DataFrame(input_params_list_summary)

    return {
        'model_code_str': model_code_str, 
        'Y_values': Y_values, 'sample_sizes_algo': sample_sizes_algo, 'mean_estimates_algo': mean_estimates_algo,
        'lower_bounds_mean_algo': lower_bounds_mean_algo, 'upper_bounds_mean_algo': upper_bounds_mean_algo,
        'std_dev_estimates_cumulative': std_dev_estimates_cumulative,
        'final_algo_mean': final_algo_mean_val,
        'final_algo_std_error_of_mean': std_error_of_final_mean_algo,
        'final_algo_ci_mean': [final_algo_ci_lower, final_algo_ci_upper],
        'estimated_output_std_dev_algo': estimated_output_std_dev_algo,
        'samples_run_by_algo': actual_samples_run_by_algo,
        'output_sample_mean': mean_Y_sample, 'output_sample_std_dev': std_Y_sample,
        'output_sample_skewness': skewness_Y_sample, 'output_sample_kurtosis': kurtosis_Y_sample,
        'qq_plot_summary_stats': qq_plot_summary,
        'best_fit_distribution_name': best_distribution_name_fitted,
        'best_fit_distribution_code': ot_code_best_distribution,
        'best_fit_distribution_object': best_fitted_distribution_obj,
        'best_fit_params': best_params_fitted,
        'distribution_fit_results_df': fit_df,
        'inputs_summary_df': inputs_summary_df_prompt,
        'fig_mean_convergence': fig_mean_convergence, 'fig_std_dev_convergence': fig_std_convergence,
        'fig_output_distribution': fig_dist,
        'summary_convergence_df': summary_convergence_df,
        'summary_distribution_df': summary_distribution_df,
        'summary_quantiles_df': quantiles_df_summary,
    }

# --- AI Insight Generation Function (Signature changed, retrieves model_code_str from analysis_data) ---
def generate_ai_insights(analysis_data: dict, # model_code_str is NOT in signature here
                         language_model: str = 'groq') -> str: 
    """
    Generate AI-powered insights for expectation convergence analysis.
    model_code_str is now expected to be within analysis_data.
    """
    try:
        model_code_str = analysis_data.get('model_code_str') # Retrieve model_code_str
        if not model_code_str: # Essential for the prompt
            return "AI insights cannot be generated: Model code string was not found in the analysis data."

        final_mean_prompt = analysis_data.get('final_algo_mean', 'N/A')
        std_err_mean_algo_prompt_val = analysis_data.get('final_algo_std_error_of_mean', 'N/A')
        ci_mean_algo_list_prompt_val = analysis_data.get('final_algo_ci_mean', ['N/A', 'N/A'])
        ci_mean_algo_lower_prompt_val = ci_mean_algo_list_prompt_val[0] if isinstance(ci_mean_algo_list_prompt_val, list) and len(ci_mean_algo_list_prompt_val) > 0 else 'N/A'
        ci_mean_algo_upper_prompt_val = ci_mean_algo_list_prompt_val[1] if isinstance(ci_mean_algo_list_prompt_val, list) and len(ci_mean_algo_list_prompt_val) > 1 else 'N/A'
        
        output_std_dev_algo_prompt_val = analysis_data.get('estimated_output_std_dev_algo', 'N/A')
        samples_run_prompt_val = analysis_data.get('samples_run_by_algo', 'N/A')
        
        skew_sample_prompt_val = analysis_data.get('output_sample_skewness', 'N/A')
        kurt_sample_prompt_val = analysis_data.get('output_sample_kurtosis', 'N/A')
        
        best_fit_name_prompt_val = analysis_data.get('best_fit_distribution_name', 'None')
        best_fit_code_prompt_val = analysis_data.get('best_fit_distribution_code', 'N/A')
        
        inputs_df_prompt_val = analysis_data.get('inputs_summary_df')
        inputs_md_prompt_val = inputs_df_prompt_val.to_markdown(index=False, floatfmt=".4g") if inputs_df_prompt_val is not None and not inputs_df_prompt_val.empty else "Input distribution data not available."

        model_code_formatted_prompt_val = f"```python\n{model_code_str.strip()}\n```"

        # --- Enhanced Q-Q Plot Information for LLM ---
        qq_summary = analysis_data.get('qq_plot_summary_stats', {})
        qq_correlation_prompt = qq_summary.get('correlation', np.nan)
        qq_lower_dev_prompt = qq_summary.get('lower_tail_deviation_avg', np.nan)
        qq_mid_dev_prompt = qq_summary.get('middle_section_deviation_avg', np.nan)
        qq_upper_dev_prompt = qq_summary.get('upper_tail_deviation_avg', np.nan)
        qq_points_prompt = qq_summary.get('representative_points', [])

        qq_interpretation_prompt = "Q-Q Plot Summary (vs. Normal Distribution):\n"
        if not np.isnan(qq_correlation_prompt):
            qq_interpretation_prompt += f"- Linearity (Correlation of Q-Q points): {qq_correlation_prompt:.3f} (A value of 1.0 suggests perfect linear alignment of points on the Q-Q plot. Deviations from 1 indicate non-linearity in the Q-Q plot, suggesting non-normal features).\n"
        else:
             qq_interpretation_prompt += "- Linearity (Correlation of Q-Q points): Not available.\n"

        if not np.isnan(qq_lower_dev_prompt):
            qq_interpretation_prompt += f"- Lower Tail (approx. bottom 25%): Sample quantiles are on average {abs(qq_lower_dev_prompt):.3f} units {'above' if qq_lower_dev_prompt > 0 else 'below'} theoretical normal quantiles. This suggests the sample's lower tail is {'heavier' if qq_lower_dev_prompt > 0 else 'lighter'} than a normal distribution's lower tail.\n"
        else:
            qq_interpretation_prompt += "- Lower Tail deviation: Not available.\n"

        if not np.isnan(qq_mid_dev_prompt):
            qq_interpretation_prompt += f"- Middle Section (approx. middle 50%): Sample quantiles are on average {abs(qq_mid_dev_prompt):.3f} units {'above' if qq_mid_dev_prompt > 0 else 'below'} theoretical normal quantiles. This reflects how the body of the distribution compares to normal.\n"
        else:
            qq_interpretation_prompt += "- Middle Section deviation: Not available.\n"

        if not np.isnan(qq_upper_dev_prompt):
            qq_interpretation_prompt += f"- Upper Tail (approx. top 25%): Sample quantiles are on average {abs(qq_upper_dev_prompt):.3f} units {'above' if qq_upper_dev_prompt > 0 else 'below'} theoretical normal quantiles. This suggests the sample's upper tail is {'heavier' if qq_upper_dev_prompt > 0 else 'lighter'} than a normal distribution's upper tail.\n"
        else:
            qq_interpretation_prompt += "- Upper Tail deviation: Not available.\n"
        
        if qq_points_prompt:
            qq_interpretation_prompt += "- Representative Q-Q Points (Theoretical Normal vs. Sample Quantiles):\n"
            for perc_str, t_q, s_q in qq_points_prompt:
                qq_interpretation_prompt += f"  - {perc_str}: Theoretical={t_q:.3f}, Sample={s_q:.3f} (Difference: {s_q - t_q:.3f})\n"
        else:
            qq_interpretation_prompt += "- Detailed Q-Q representative point data not available.\n"
        qq_interpretation_prompt += "Based on these Q-Q characteristics, describe the output distribution's shape relative to a normal distribution (e.g., overall fit, skewness, tail weight/behavior)."

        prompt = f"""{RETURN_INSTRUCTION}
You are an expert in uncertainty quantification and statistical simulation. I have performed a Monte Carlo Expectation Convergence Analysis for a computational model with a single output. Please provide a detailed interpretation of the results.

**Computational Model Code:**
{model_code_formatted_prompt_val}

**Input Variable Distributions Summary:**
{inputs_md_prompt_val}

**Key Results from ExpectationSimulationAlgorithm (OpenTURNS):**
- Final Estimated Mean of Output: {final_mean_prompt:.5g}
- Final Estimated Standard Error of the Sample Mean: {std_err_mean_algo_prompt_val:.5g}
- 95% Confidence Interval for the Mean: [{ci_mean_algo_lower_prompt_val:.5g}, {ci_mean_algo_upper_prompt_val:.5g}]
- Total Samples Used by Algorithm (N_samples * BlockSize, as COV criterion was 'NONE'): {samples_run_prompt_val}

**Characteristics of the Output Sample (based on {samples_run_prompt_val} model evaluations):**
- Estimated Standard Deviation of Output (σY_hat from Algo): {output_std_dev_algo_prompt_val:.5g}
- Skewness (Sample): {skew_sample_prompt_val:.3f}
- Kurtosis (Fisher, normal=0, Sample): {kurt_sample_prompt_val:.3f}
- Best-Fit Distribution (by AIC, if available): {best_fit_name_prompt_val} (OpenTURNS Code: `{best_fit_code_prompt_val}`)

**Q-Q Plot Summary & Interpretation Guidance (vs. Normal Distribution):**
{qq_interpretation_prompt}

**Analysis Request:**
Please provide insights on the following, using precise scientific language:

1.  **Convergence Quality & Sufficiency of Samples:**
    * Comment on the stability of the mean estimate as seen in its convergence plot (not shown to you, but infer from CI). Based on the 95% CI width ([{ci_mean_algo_lower_prompt_val:.5g}, {ci_mean_algo_upper_prompt_val:.5g}]) relative to the mean ({final_mean_prompt:.5g}), how precise is this estimate?
    * Similarly, discuss the convergence of the standard deviation estimate (plot not shown, infer general stability if possible).
    * Is the number of samples ({samples_run_prompt_val}) likely sufficient for reliable estimates of these moments, given the model's nature (if inferable) and output variability?

2.  **Output Uncertainty & Variability:**
    * Interpret the final estimated mean ({final_mean_prompt:.5g}).
    * What does the estimated standard deviation of the output (σY_hat ≈ {output_std_dev_algo_prompt_val:.5g}) indicate about the inherent variability or spread of the model's output?

3.  **Output Distribution Shape & Characteristics:**
    * Based on the provided Q-Q plot summary statistics (linearity correlation, tail deviations, representative points), describe how the output distribution deviates from a normal distribution.
    * Discuss the implications of the output sample's skewness ({skew_sample_prompt_val:.3f}) and kurtosis ({kurt_sample_prompt_val:.3f}), and how these align with the Q-Q plot findings.
    * If a best-fit distribution ({best_fit_name_prompt_val}) was identified, what does this specific type of distribution typically imply about underlying processes? How does its shape (e.g., parameters in `{best_fit_code_prompt_val}`) align with the observed skewness, kurtosis, and Q-Q characteristics?

4.  **Practical Implications & Recommendations:**
    * What are the key takeaways from this analysis for someone using this model's output for decision-making or risk assessment?
    * Are there any particular cautions if, for example, the distribution is heavily skewed or has extreme tails (as might be indicated by the Q-Q summary and kurtosis)?
    * Suggest potential next steps: Should more samples be run? Is the current precision adequate? Could the fitted distribution be used in further analyses (e.g., reliability studies)?

Structure your response clearly. Refer to the provided numerical values. The algorithm was run with `CoefficientOfVariationCriterionType = 'NONE'`, meaning it likely completed all `MaximumOuterSampling` iterations.
"""
        model_name_for_api = language_model
        if not language_model or language_model.lower() == 'groq':
            model_name_for_api = "llama3-70b-8192" 

        insights = call_groq_api(prompt, model_name=model_name_for_api)
        return insights
        
    except Exception as e:
        st.error(f"Error in generate_ai_insights: {str(e)}")
        return "Error: AI insights could not be generated due to an internal error."

# --- Main Entry Point (if your app calls this directly) ---
def expectation_convergence_analysis(model: callable, problem: ot.Distribution, model_code_str: str = None, 
                                     N_samples: int = 10000, language_model: str = 'groq', 
                                     display_results: bool = True) -> dict:
    """
    Perform and display expectation convergence analysis for a univariate model.
    This function is the primary wrapper if your main_app calls a single function for this module.
    """
    results_placeholder = None
    if display_results:
        results_placeholder = st.empty()
        with results_placeholder.container():
             st.info("🚀 Starting Expectation Convergence Analysis...")
    
    analysis_data = {}
    try:
        # Pass model_code_str to compute_... so it's in analysis_data
        analysis_data = compute_expectation_convergence_analysis(
            model, problem, model_code_str, N_samples
        )
        
        llm_insights = None
        # generate_ai_insights will retrieve model_code_str from analysis_data dictionary
        # It's crucial that model_code_str was passed to compute_... and stored in analysis_data
        if language_model and analysis_data.get('model_code_str') and analysis_data.get('final_algo_mean') is not None:
            if display_results:
                with results_placeholder.container():
                    st.info("🧠 Generating AI insights... This may take a moment.")
            
            llm_insights = generate_ai_insights(analysis_data, language_model=language_model) 
        
        analysis_data['ai_insights'] = llm_insights

        if display_results and results_placeholder:
            with results_placeholder.container():
                st.success("✅ Expectation Convergence Analysis Completed!")
            
        st.session_state.expectation_convergence_results = analysis_data
        
        if display_results:
            display_expectation_convergence_results(analysis_data)
            
        return analysis_data
    
    except Exception as e:
        error_message = f"Critical error in expectation_convergence_analysis wrapper: {str(e)}"
        if display_results:
            if results_placeholder: 
                with results_placeholder.container(): st.error(error_message)
            else: st.error(error_message)
        else:
            print(error_message) # Log if not displaying
        
        analysis_data['error'] = error_message
        analysis_data['ai_insights'] = "AI insights skipped due to critical analysis error."
        return analysis_data

# --- Display Function ---
def display_expectation_convergence_results(analysis_data: dict):
    """
    Display expectation convergence analysis results in the Streamlit interface.
    """
    if not analysis_data:
        st.error("Analysis data is missing, cannot display results.")
        return
    # Check if the analysis itself had an error, but still try to display what we can
    if 'error' in analysis_data and analysis_data['error'] and not analysis_data.get('fig_mean_convergence'):
        st.error(f"Analysis Error: {analysis_data.get('error')}")
        # Display AI insights if they contain an error message from AI generation step
        if 'ai_insights' in analysis_data and "Error" in str(analysis_data['ai_insights']):
             st.subheader("🧠 AI-Generated Insights & Interpretation")
             st.warning(analysis_data['ai_insights'])
        return

    st.header("📈 Expectation Convergence Analysis")
    st.markdown("""
    This analysis examines how the estimated mean and standard deviation of the model output 
    converge as the number of Monte Carlo samples increases. It also characterizes 
    the output distribution. This helps assess the reliability of these statistical estimates.
    """)

    st.subheader("Convergence Plots")
    # Check if figures are actual Plotly Figure objects and have data
    mean_fig = analysis_data.get('fig_mean_convergence')
    if isinstance(mean_fig, go.Figure) and mean_fig.data:
        st.plotly_chart(mean_fig, use_container_width=True)
    else: st.warning("Mean convergence plot data is unavailable or empty.")

    std_fig = analysis_data.get('fig_std_dev_convergence')
    if isinstance(std_fig, go.Figure) and std_fig.data:
        st.plotly_chart(std_fig, use_container_width=True)
    else: st.warning("Standard deviation convergence plot data is unavailable or empty.")

    st.subheader("Output Distribution Analysis")
    dist_fig = analysis_data.get('fig_output_distribution')
    if isinstance(dist_fig, go.Figure) and dist_fig.data:
        st.plotly_chart(dist_fig, use_container_width=True)
    else: st.warning("Output distribution plot data is unavailable or empty.")

    st.subheader("Summary Statistics")
    col1, col2 = st.columns(2)
    with col1:
        df_conv_summary = analysis_data.get('summary_convergence_df')
        if isinstance(df_conv_summary, pd.DataFrame) and not df_conv_summary.empty:
            st.markdown("##### Algorithm Convergence Summary")
            st.dataframe(df_conv_summary, hide_index=True, use_container_width=True)
        else: st.markdown("Algorithm convergence summary not available.")
    with col2:
        df_dist_summary = analysis_data.get('summary_distribution_df')
        if isinstance(df_dist_summary, pd.DataFrame) and not df_dist_summary.empty:
            st.markdown("##### Output Sample Statistics")
            st.dataframe(df_dist_summary, hide_index=True, use_container_width=True)
        else: st.markdown("Output sample statistics not available.")
    
    df_quantiles = analysis_data.get('summary_quantiles_df')
    if isinstance(df_quantiles, pd.DataFrame) and not df_quantiles.empty:
        st.markdown("##### Output Sample Quantiles")
        st.dataframe(df_quantiles, hide_index=True, use_container_width=True)
    else: st.markdown("Output sample quantiles not available.")

    st.subheader("Distribution Fitting")
    fit_df_display = analysis_data.get('distribution_fit_results_df')
    if isinstance(fit_df_display, pd.DataFrame) and not fit_df_display.empty:
        best_fit_code_display_val = analysis_data.get('best_fit_distribution_code', 'N/A')
        st.markdown(f"**Best Fit (by AIC): {analysis_data.get('best_fit_distribution_name', 'N/A')}**")
        st.markdown(f"OpenTURNS Code: `{best_fit_code_display_val}`")
        
        cols_to_show_fit_display = ['Distribution', 'AIC', 'BIC', 'KS_Statistic', 'KS_pvalue']
        actual_cols_fit_display = [col for col in cols_to_show_fit_display if col in fit_df_display.columns]
        display_fit_df_st_val = fit_df_display[actual_cols_fit_display].copy()

        for col_format_fit_val in ['AIC', 'BIC']:
            if col_format_fit_val in display_fit_df_st_val: 
                try: display_fit_df_st_val[col_format_fit_val] = pd.to_numeric(display_fit_df_st_val[col_format_fit_val], errors='coerce').map('{:.2f}'.format)
                except: pass # Ignore formatting errors if conversion fails
        for col_format_fit_val in ['KS_Statistic', 'KS_pvalue']:
             if col_format_fit_val in display_fit_df_st_val: 
                 try: display_fit_df_st_val[col_format_fit_val] = pd.to_numeric(display_fit_df_st_val[col_format_fit_val], errors='coerce').map('{:.4f}'.format)
                 except: pass
        
        st.dataframe(display_fit_df_st_val.head(), hide_index=True, use_container_width=True)
    else:
        st.info("No distribution fitting performed or no distributions provided a suitable fit.")

    ai_insights_text_display_val = analysis_data.get('ai_insights')
    if ai_insights_text_display_val:
        st.subheader("🧠 AI-Generated Insights & Interpretation")
        st.markdown(ai_insights_text_display_val)
    elif analysis_data.get("model_code_str") is None and analysis_data.get("language_model") and not ai_insights_text_display_val and not ('error' in analysis_data and analysis_data['error']):
         st.warning("AI insights could not be generated because the model code was not available to the insight generation step.")
    elif analysis_data.get("language_model") and not ai_insights_text_display_val and not ('error' in analysis_data and analysis_data['error']):
         st.warning("AI insights were expected but not generated. This might be due to an API error or other issue during insight generation.")


# --- Chat Context Function ---
def get_expectation_convergence_context_for_chat(analysis_data: dict) -> str:
    """
    Generate a formatted string of key expectation convergence results for chat context.
    """
    if not analysis_data or ('error' in analysis_data and analysis_data['error']):
        return f"Expectation convergence analysis results are not available or an error occurred: {analysis_data.get('error', 'Unknown error')}"

    context = "\n\n### Expectation Convergence Analysis Summary:\n"
    context += f"- Final Mean Estimate (Algo): {analysis_data.get('final_algo_mean', 'N/A'):.4g}\n"
    context += f"- Est. Output Std Dev (Algo σY_hat): {analysis_data.get('estimated_output_std_dev_algo', 'N/A'):.4g}\n"
    ci_mean_chat_val = analysis_data.get('final_algo_ci_mean', ['N/A', 'N/A'])
    # Ensure ci_mean_chat_val is a list with at least two elements before indexing
    ci_lower_chat = ci_mean_chat_val[0] if isinstance(ci_mean_chat_val, list) and len(ci_mean_chat_val) > 0 else 'N/A'
    ci_upper_chat = ci_mean_chat_val[1] if isinstance(ci_mean_chat_val, list) and len(ci_mean_chat_val) > 1 else 'N/A'
    context += f"- 95% CI for Mean (Algo): [{ci_lower_chat:.4g}, {ci_upper_chat:.4g}]\n"
    
    context += f"- Samples Run by Algo: {analysis_data.get('samples_run_by_algo', 'N/A')}\n"
    context += f"- Output Sample Skewness: {analysis_data.get('output_sample_skewness', 'N/A'):.3f}\n"
    context += f"- Output Sample Kurtosis: {analysis_data.get('output_sample_kurtosis', 'N/A'):.3f}\n"
    context += f"- Best Fit Dist. (AIC): {analysis_data.get('best_fit_distribution_name', 'None')}\n"
    
    qq_summary_chat = analysis_data.get('qq_plot_summary_stats', {})
    qq_corr_chat = qq_summary_chat.get('correlation', np.nan)
    if not np.isnan(qq_corr_chat):
        context += f"- Q-Q Plot (vs Normal) Correlation: {qq_corr_chat:.3f}\n"
            
    ai_insights_chat_val = analysis_data.get('ai_insights')
    if ai_insights_chat_val and isinstance(ai_insights_chat_val, str):
        # Create a more targeted snippet for chat
        insight_lines_chat_val = ai_insights_chat_val.split('\n')
        executive_summary_lines = [line for line in insight_lines_chat_val if "executive summary" in line.lower() or "key findings" in line.lower()]
        if executive_summary_lines:
            # Try to find the start of the summary and take a few lines
            try:
                start_index = insight_lines_chat_val.index(executive_summary_lines[0])
                snippet = "\n".join(insight_lines_chat_val[start_index : min(len(insight_lines_chat_val), start_index + 4)])
                context += f"\n**AI Insights Snippet:**\n{snippet}...\n"
            except ValueError: # Should not happen if executive_summary_lines is populated
                context += f"\n**AI Insights Snippet:**\n{ai_insights_chat_val[:250]}...\n"
        elif len(insight_lines_chat_val) > 2 :
             non_header_lines_chat = [line.strip() for line in insight_lines_chat_val if line.strip() and not line.strip().startswith("#")]
             if len(non_header_lines_chat) >=1:
                 context += f"\n**AI Insights Snippet:**\n{non_header_lines_chat[0]}"
                 if len(non_header_lines_chat) >=2: context += f"\n{non_header_lines_chat[1]}"
                 context += "...\n"
        else: 
            context += f"\n**AI Insights Snippet:**\n{ai_insights_chat_val[:250]}...\n"
        context += "(Refer to full analysis for complete AI interpretation)\n"
    
    return context
