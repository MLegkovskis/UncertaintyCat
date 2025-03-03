import numpy as np
import pandas as pd
import openturns as ot
import matplotlib.pyplot as plt
import streamlit as st
from utils.core_utils import call_groq_api
from utils.markdown_utils import RETURN_INSTRUCTION
from utils.model_utils import get_ot_distribution, get_ot_model, get_distribution_names


def expectation_convergence_analysis(model, problem, model_code_str, N_samples=8000, language_model='groq'):
    # Create distributions
    distribution = get_ot_distribution(problem)

    # Create OpenTURNS model
    ot_model = get_ot_model(model, problem)

    # Define the input random vector and the output random vector
    input_random_vector = ot.RandomVector(distribution)
    output_random_vector = ot.CompositeRandomVector(ot_model, input_random_vector)

    # Run expectation simulation algorithm
    expectation_algo = ot.ExpectationSimulationAlgorithm(output_random_vector)
    expectation_algo.setMaximumOuterSampling(N_samples)
    expectation_algo.setBlockSize(1)
    expectation_algo.setCoefficientOfVariationCriterionType("NONE")

    expectation_algo.run()
    result = expectation_algo.getResult()

    # Extract convergence data
    graph = expectation_algo.drawExpectationConvergence()
    data = graph.getDrawable(0).getData()

    # Convert OpenTURNS Samples to numpy arrays
    sample_sizes = np.array([s[0] for s in data[:, 0]])
    mean_estimates = np.array([m[0] for m in data[:, 1]])

    # Calculate 95% confidence intervals using the final standard deviation
    final_std_dev = result.getStandardDeviation()[0]
    initial_sample_size = sample_sizes[-1]
    standard_errors = final_std_dev * np.sqrt(initial_sample_size / sample_sizes)
    z_value = 1.96  # 95% confidence
    lower_bounds = mean_estimates - z_value * standard_errors
    upper_bounds = mean_estimates + z_value * standard_errors
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=120)

    # --- Mean Estimate Convergence Plot (Linear Scale) ---
    axes[0].plot(sample_sizes, mean_estimates, label='Mean Estimate', color='#1f77b4', linewidth=2)
    axes[0].fill_between(sample_sizes, lower_bounds, upper_bounds, color='#1f77b4', alpha=0.2, label='95% Confidence Interval')

    # Highlight the point of convergence
    convergence_point = (sample_sizes[-1], mean_estimates[-1])
    axes[0].scatter(*convergence_point, color='red', s=100, zorder=5, label='Converged Mean')
    axes[0].annotate(f'Converged Mean: {mean_estimates[-1]:.4f}',
                     xy=convergence_point,
                     xytext=(-100, 20),  # Adjust position within the plot
                     textcoords='offset points',
                     fontsize=12,
                     arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

    axes[0].set_title('Convergence of Mean Estimate (Linear Scale)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Sample Size', fontsize=14)
    axes[0].set_ylabel('Mean Estimate of Output Y', fontsize=14)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].legend(fontsize=12, loc='best')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # --- Mean Estimate Convergence Plot (Logarithmic Scale) ---
    axes[1].plot(sample_sizes, mean_estimates, label='Mean Estimate', color='#1f77b4', linewidth=2)
    axes[1].fill_between(sample_sizes, lower_bounds, upper_bounds, color='#1f77b4', alpha=0.2, label='95% Confidence Interval')

    # Highlight the point of convergence within the plot
    axes[1].scatter(*convergence_point, color='red', s=100, zorder=5, label='Converged Mean')
    axes[1].annotate(f'Converged Mean: {mean_estimates[-1]:.4f}',
                     xy=convergence_point,
                     xytext=(-100, 20),  # Adjust position within the plot
                     textcoords='offset points',
                     fontsize=12,
                     arrowprops=dict(arrowstyle='->', color='red', linewidth=2))

    axes[1].set_title('Convergence of Mean Estimate (Log Scale)', fontsize=16, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Sample Size (Log Scale)', fontsize=14)
    axes[1].set_ylabel('Mean Estimate of Output Y', fontsize=14)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].legend(fontsize=12, loc='best')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    # --- Distribution of Y Plot ---

    # Generate samples of Y at convergence
    sample_size = 5000
    input_sample = distribution.getSample(sample_size)
    output_sample = ot_model(input_sample)

    Y_values = np.array(output_sample).flatten()

    # Plot histogram with density curve
    import seaborn as sns
    sns.histplot(Y_values, bins=50, kde=True, color='#2ca02c', edgecolor='black', alpha=0.7, ax=axes[2])

    axes[2].set_title('Distribution of Model Output Y at Convergence', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('Output Y', fontsize=14)
    axes[2].set_ylabel('Frequency', fontsize=14)
    axes[2].tick_params(axis='both', which='major', labelsize=12)

    # Calculate statistics
    mean_Y = np.mean(Y_values)
    std_Y = np.std(Y_values)
    conf_int = [mean_Y - 1.96 * std_Y, mean_Y + 1.96 * std_Y]

    # Add vertical lines for mean and confidence intervals
    axes[2].axvline(mean_Y, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_Y:.4f}')
    axes[2].axvline(conf_int[0], color='purple', linestyle='--', linewidth=2, label=f'95% CI Lower: {conf_int[0]:.4f}')
    axes[2].axvline(conf_int[1], color='purple', linestyle='--', linewidth=2, label=f'95% CI Upper: {conf_int[1]:.4f}')

    # Add annotations for statistics
    # axes[2].annotate(f'Mean: {mean_Y:.4f}',
    #                  xy=(mean_Y, axes[2].get_ylim()[1]*0.9),
    #                  xytext=(10, 0),
    #                  textcoords='offset points',
    #                  fontsize=12,
    #                  color='blue',
    #                  rotation=90)

    # axes[2].annotate(f'95% CI Lower: {conf_int[0]:.4f}',
    #                  xy=(conf_int[0], axes[2].get_ylim()[1]*0.7),
    #                  xytext=(10, 0),
    #                  textcoords='offset points',
    #                  fontsize=12,
    #                  color='purple',
    #                  rotation=90)

    # axes[2].annotate(f'95% CI Upper: {conf_int[1]:.4f}',
    #                  xy=(conf_int[1], axes[2].get_ylim()[1]*0.7),
    #                  xytext=(10, 0),
    #                  textcoords='offset points',
    #                  fontsize=12,
    #                  color='purple',
    #                  rotation=90)

    axes[2].legend(fontsize=12, loc='best')
    axes[2].grid(True, linestyle='--', alpha=0.7)

    # --- Overall Figure Adjustments ---
    plt.suptitle('Expectation Convergence and Output Distribution', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect to fit the suptitle
    # Prepare data for the prompt
    convergence_summary = f"""
- The convergence algorithm determined that the model statistics converge at a sample size of {int(initial_sample_size)}.
- The final mean estimate is {mean_estimates[-1]:.6f}.
- The standard deviation at convergence is {final_std_dev:.6f}.
- The 95% confidence interval for the mean estimate is from {lower_bounds[-1]:.6f} to {upper_bounds[-1]:.6f}.
"""

    # Prepare the input parameters and their uncertainties
    input_parameters = []
    for name, dist_info in zip(problem['names'], problem['distributions']):
        input_parameters.append({
            'Variable': name,
            'Distribution': dist_info['type'],
            'Parameters': dist_info['params']
        })

    inputs_df = pd.DataFrame(input_parameters)
    inputs_md_table = inputs_df.to_markdown(index=False)

    # Use the provided model_code_str directly
    model_code = model_code_str

    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code.strip().split('\n')])

    # Prepare the prompt
    prompt = f"""
{RETURN_INSTRUCTION}

Given the following user-defined model defined in Python code:

```python
{model_code_formatted}
```

and the following uncertain input distributions:

{inputs_df}

Given the following summary of an expectation convergence analysis (remember you must convert everything given to you into a valid Markdown):

{convergence_summary}

Please:
  - Provide a detailed summary and interpretation of the results.
  - Focus on the likely operating range of the model under the specified uncertainty.
  - Explain the significance of the convergence of the mean estimate, referencing the Central Limit Theorem and statistical concepts.
  - Discuss how the confidence interval helps in understanding the risk and operational range of the model when deployed in reality.
"""

    response_key = 'expectation_response_markdown'
    fig_key = 'expectation_fig'

    # Check if the results are already in session state
    if response_key not in st.session_state:
        response_markdown = call_groq_api(prompt, model_name=language_model)
        st.session_state[response_key] = response_markdown
    else:
        response_markdown = st.session_state[response_key]

    if fig_key not in st.session_state:
        st.session_state[fig_key] = fig
    else:
        fig = st.session_state[fig_key]

    # Display the results
    st.markdown(response_markdown)
    st.pyplot(fig)


def expectation_convergence_analysis_joint(model, problem, model_code_str, N_samples=8000, language_model='groq'):
    """Analyze convergence of Monte Carlo estimation."""
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")
    
    # Use the distribution directly
    distribution = problem
    
    # Get variable names
    dimension = distribution.getDimension()
    variable_names = []
    for i in range(dimension):
        marginal = distribution.getMarginal(i)
        variable_names.append(marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}")
    
    # Set up sample sizes
    N_values = np.unique(np.logspace(1, np.log10(N_samples), 20).astype(int))
    
    # Initialize arrays to store results
    means = np.zeros((len(N_values), 20))
    stds = np.zeros((len(N_values), 20))
    
    # Run Monte Carlo simulations
    for i, N in enumerate(N_values):
        for j in range(20):
            # Generate samples - ensure N is an integer
            X = np.array(distribution.getSample(int(N)))
            Y = np.array([model(x) for x in X])
            
            # Calculate statistics
            means[i, j] = np.mean(Y)
            stds[i, j] = np.std(Y)
    
    # Calculate statistics across replicates
    mean_of_means = np.mean(means, axis=1)
    std_of_means = np.std(means, axis=1)
    mean_of_stds = np.mean(stds, axis=1)
    std_of_stds = np.std(stds, axis=1)
    
    # Create confidence intervals
    ci_means = 1.96 * std_of_means
    ci_stds = 1.96 * std_of_stds
    
    # Plot results
    st.write("### Convergence Analysis")
    st.write("This analysis shows how the estimated mean and standard deviation converge as the sample size increases.")
    
    # Mean convergence plot
    fig_mean, ax_mean = plt.subplots(figsize=(10, 6))
    ax_mean.semilogx(N_values, mean_of_means, 'b-', label='Mean estimate')
    ax_mean.fill_between(N_values, 
                        mean_of_means - ci_means,
                        mean_of_means + ci_means,
                        color='b', alpha=0.2,
                        label='95% Confidence interval')
    ax_mean.grid(True)
    ax_mean.set_xlabel('Number of samples')
    ax_mean.set_ylabel('Estimated mean')
    ax_mean.set_title('Convergence of Mean Estimate')
    ax_mean.legend()
    st.pyplot(fig_mean)
    plt.close()
    
    # Standard deviation convergence plot
    fig_std, ax_std = plt.subplots(figsize=(10, 6))
    ax_std.semilogx(N_values, mean_of_stds, 'r-', label='Std. dev. estimate')
    ax_std.fill_between(N_values,
                       mean_of_stds - ci_stds,
                       mean_of_stds + ci_stds,
                       color='r', alpha=0.2,
                       label='95% Confidence interval')
    ax_std.grid(True)
    ax_std.set_xlabel('Number of samples')
    ax_std.set_ylabel('Estimated standard deviation')
    ax_std.set_title('Convergence of Standard Deviation Estimate')
    ax_std.legend()
    st.pyplot(fig_std)
    plt.close()
    
    # Calculate convergence metrics
    relative_mean_change = np.abs(np.diff(mean_of_means) / mean_of_means[:-1])
    relative_std_change = np.abs(np.diff(mean_of_stds) / mean_of_stds[:-1])
    
    # Find where convergence criteria are met
    mean_converged = np.where(relative_mean_change < 0.01)[0]
    std_converged = np.where(relative_std_change < 0.01)[0]
    
    # Report convergence findings
    st.write("### Convergence Analysis Results")
    
    if len(mean_converged) > 0:
        n_mean = N_values[mean_converged[0]]
        st.write(f"- Mean estimate converges (< 1% change) at approximately {n_mean} samples")
    else:
        st.write("- Mean estimate has not fully converged with the given sample sizes")
        
    if len(std_converged) > 0:
        n_std = N_values[std_converged[0]]
        st.write(f"- Standard deviation estimate converges (< 1% change) at approximately {n_std} samples")
    else:
        st.write("- Standard deviation estimate has not fully converged with the given sample sizes")
    
    # Final estimates
    st.write("\n### Final Estimates (using maximum sample size)")
    st.write(f"- Mean: {mean_of_means[-1]:.4f} ± {ci_means[-1]:.4f}")
    st.write(f"- Standard Deviation: {mean_of_stds[-1]:.4f} ± {ci_stds[-1]:.4f}")
    
    # Recommendations
    st.write("\n### Recommendations")
    if len(mean_converged) > 0 and len(std_converged) > 0:
        recommended_n = max(n_mean, n_std)
        st.write(f"Based on the convergence analysis, using {recommended_n} samples should provide stable estimates.")
    else:
        st.write("Consider increasing the maximum sample size to achieve better convergence.")