import numpy as np
import pandas as pd
import plotly.express as px
import openturns as ot
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
from modules.statistical_utils import get_bounds_for_salib

def run_morris_analysis_for_dimensionality_reduction(N, model, problem):
    """Run Morris analysis for dimensionality reduction."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")

    # Get input names and bounds
    dimension = problem.getDimension()
    names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        names.append(marginal.getDescription()[0])

    # Generate bounds for SALib
    bounds = get_bounds_for_salib(problem)

    # Create SALib problem definition
    problem_for_salib = {
        'num_vars': dimension,
        'names': names,
        'bounds': bounds
    }

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

    # Calculate thresholds based on percentiles
    threshold_mu_star = np.percentile(mu_star, 25)  # Lower quartile
    threshold_sigma = np.percentile(sigma, 25)     # Lower quartile

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
        'Influential': ['Yes' if i not in non_influential_indices else 'No' for i in range(len(names))]
    })

    # Create Morris plot
    fig = px.scatter(
        df,
        x='sigma',
        y='mu_star',
        text='Variable',
        color='Influential',
        title='Morris Sensitivity Plot',
        labels={
            'sigma': 'σ (Interaction Effects)',
            'mu_star': 'μ* (Overall Effect)',
            'Variable': 'Input Variable'
        }
    )

    # Update text position and plot layout
    fig.update_traces(textposition='top center')
    fig.update_layout(
        showlegend=True,
        width=800,
        height=600
    )

    return non_influential_indices, mu_star, sigma, fig