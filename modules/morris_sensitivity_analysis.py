import numpy as np
import pandas as pd
import plotly.express as px
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
from modules.statistical_utils import get_bounds_for_salib

def run_morris_analysis_for_dimensionality_reduction(N, model, problem):
    # Generate bounds for SALib
    problem_for_salib = get_bounds_for_salib(problem)

    # Generate samples
    param_values = morris.sample(problem_for_salib, N, num_levels=4, optimal_trajectories=10)

    # Evaluate the model
    Y = np.array([model(X) for X in param_values]).flatten()

    # Perform Morris analysis
    Si = morris_analyze.analyze(
        problem_for_salib, param_values, Y, conf_level=0.95, print_to_console=False, num_levels=4
    )

    # Extract sensitivity indices
    mu_star = Si['mu_star']
    sigma = Si['sigma']
    names = problem['names']

    # Calculate thresholds based on mean and standard deviation
    mean_mu_star = np.mean(mu_star)
    std_mu_star = np.std(mu_star)
    threshold_mu_star = mean_mu_star - std_mu_star

    mean_sigma = np.mean(sigma)
    std_sigma = np.std(sigma)
    threshold_sigma = mean_sigma - std_sigma

    # Determine non-influential variables based on thresholds
    non_influential_indices = [
        i for i in range(len(mu_star))
        if mu_star[i] < threshold_mu_star and sigma[i] < threshold_sigma
    ]

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'mu_star': mu_star,
        'sigma': sigma,
        'Variable': names,
        'Influential': ['Non-influential' if i in non_influential_indices else 'Influential' for i in range(len(names))]
    })

    # Create interactive plot using Plotly
    fig = px.scatter(
        df,
        x='mu_star',
        y='sigma',
        color='Influential',
        hover_name='Variable',
        labels={
            'mu_star': 'μ* (Mean of Absolute Elementary Effects)',
            'sigma': 'σ (Standard Deviation of EE)'
        },
        title=f'Morris Sensitivity Analysis (N = {N})',
        width=800,
        height=600
    )

    # Add threshold lines
    fig.add_vline(x=threshold_mu_star, line_dash='dash', line_color='grey',
                  annotation_text='Threshold μ*', annotation_position='top left')
    fig.add_hline(y=threshold_sigma, line_dash='dash', line_color='grey',
                  annotation_text='Threshold σ', annotation_position='top right')

    # Update layout for better readability
    fig.update_layout(
        legend_title_text='Variable Influence',
        xaxis_title='μ* (Mean of Absolute Elementary Effects)',
        yaxis_title='σ (Standard Deviation of EE)',
        hovermode='closest'
    )

    return non_influential_indices, mu_star, sigma, fig