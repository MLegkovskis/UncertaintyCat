# modules/monte_carlo.py

import numpy as np
import pandas as pd
import openturns as ot
from .statistical_utils import sample_inputs

def monte_carlo_simulation(model, problem, N=1000, seed=42):
    """Run Monte Carlo simulation."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")

    # Get input names from marginals
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        input_names.append(marginal.getDescription()[0])

    # Generate samples
    X = sample_inputs(N, problem, seed)
    Y = np.array([model(x) for x in X])

    # Compute statistics
    mean = np.mean(Y)
    std = np.std(Y)
    quantiles = np.percentile(Y, [2.5, 97.5])

    # Create results dictionary
    results = {
        'samples': X,
        'responses': Y,
        'mean': mean,
        'std': std,
        'quantiles': quantiles,
        'input_names': input_names
    }

    return results

def create_monte_carlo_dataframe(results):
    """Create a DataFrame with Monte Carlo results."""
    X = results['samples']
    Y = results['responses']
    input_names = results['input_names']

    # Create DataFrame
    data = pd.DataFrame(X, columns=input_names)
    data['Y'] = Y

    return data

def print_monte_carlo_summary(results):
    """Print summary of Monte Carlo results."""
    print("\nMonte Carlo Simulation Results:")
    print(f"Mean: {results['mean']:.4f}")
    print(f"Standard Deviation: {results['std']:.4f}")
    print(f"95% Confidence Interval: [{results['quantiles'][0]:.4f}, {results['quantiles'][1]:.4f}]")
