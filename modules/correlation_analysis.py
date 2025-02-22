import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openturns as ot
from modules.monte_carlo import monte_carlo_simulation, create_monte_carlo_dataframe
from modules.statistical_utils import get_bounds

def correlation_analysis(model, problem, code_snippet, language_model="llama-3.3-70b-versatile"):
    """Perform correlation analysis."""
    
    # Run Monte Carlo simulation to get samples
    N = 1000
    results = monte_carlo_simulation(model, problem, N=N)
    data = create_monte_carlo_dataframe(results)
    
    # Calculate correlations
    corr_pearson = data.corr(method='pearson')
    corr_spearman = data.corr(method='spearman')
    
    # Create a dictionary of correlation methods
    methods = {
        'Pearson': corr_pearson['Y'].drop('Y'),
        'Spearman': corr_spearman['Y'].drop('Y')
    }
    
    # Get variable names
    if isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        dimension = problem.getDimension()
        names = []
        for i in range(dimension):
            marginal = problem.getMarginal(i)
            names.append(marginal.getDescription()[0])
    else:
        names = problem['names']
    
    # Create DataFrame with correlation results
    df = pd.DataFrame(methods, index=names)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    plt.title('Correlation Analysis')
    plt.xlabel('Input Variables')
    plt.ylabel('Correlation Coefficient')
    ax.set_xticklabels(names, rotation=45, ha='right')
    plt.legend(title='Method')
    plt.tight_layout()
    
    # Generate description of input distributions
    if isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        inputs_description = ""
        for i in range(problem.getDimension()):
            marginal = problem.getMarginal(i)
            param_name = marginal.getDescription()[0]
            inputs_description += f"{param_name}: {marginal.__class__.__name__}, parameters {list(marginal.getParameter())}\n"
    else:
        inputs_description = ""
        for name, dist_info in zip(names, problem['distributions']):
            inputs_description += f"{name}: {dist_info}\n"
    
    # Store results
    results = {
        "correlations": df,
        "inputs_description": inputs_description,
        "figure": fig
    }
    
    # Close the figure to prevent memory leaks
    plt.close()
    
    return results
