# modules/monte_carlo.py

import numpy as np
import pandas as pd
import openturns as ot
from utils.model_utils import sample_inputs

def monte_carlo_simulation(model, problem, N=1000, seed=42):
    """Run Monte Carlo simulation."""
    # Ensure problem is an OpenTURNS distribution
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        raise ValueError("Problem must be an OpenTURNS distribution")

    # Get input names from marginals
    dimension = problem.getDimension()
    input_names = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        input_names.append(marginal.getDescription()[0] if marginal.getDescription()[0] != "" else f"X{i+1}")

    # Generate samples
    X = sample_inputs(N, problem, seed)
    
    # Evaluate model for each sample
    # Handle both scalar and vector outputs
    Y_list = []
    for x in X:
        y = model(x)
        # Convert scalar or list to numpy array
        if isinstance(y, (int, float)):
            y = np.array([y])
        elif isinstance(y, list):
            y = np.array(y)
        Y_list.append(y)
    
    Y = np.vstack(Y_list)
    
    # Compute statistics
    if Y.shape[1] == 1:
        Y = Y.flatten()
        mean = np.mean(Y)
        std = np.std(Y)
        quantiles = np.percentile(Y, [2.5, 97.5])
    else:
        mean = np.mean(Y, axis=0)
        std = np.std(Y, axis=0)
        quantiles = np.array([np.percentile(Y[:, i], [2.5, 97.5]) for i in range(Y.shape[1])])

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
    """Create a DataFrame from Monte Carlo simulation results."""
    # Extract data from results
    X = results['samples']
    Y = results['responses']
    input_names = results['input_names']
    
    # Create DataFrame with input variables
    data = pd.DataFrame(X, columns=input_names)
    
    # Add response column(s)
    if isinstance(Y, np.ndarray):
        if Y.ndim == 1:
            # Single output
            data['Y'] = Y
        else:
            # Multiple outputs
            for i in range(Y.shape[1]):
                data[f'Y{i+1}'] = Y[:, i]
    else:
        # Fallback for any other case
        data['Y'] = Y
    
    return data

def print_monte_carlo_summary(results):
    """Print summary of Monte Carlo results."""
    print("\nMonte Carlo Simulation Results:")
    print(f"Mean: {results['mean']:.4f}")
    print(f"Standard Deviation: {results['std']:.4f}")
    print(f"95% Confidence Interval: [{results['quantiles'][0]:.4f}, {results['quantiles'][1]:.4f}]")
