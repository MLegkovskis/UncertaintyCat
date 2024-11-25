# modules/monte_carlo.py

import numpy as np
import pandas as pd
from modules.statistical_utils import sample_inputs

# Function to perform Monte Carlo simulation
def monte_carlo_simulation(N, model, problem):
    X_samples = sample_inputs(N, problem)
    Y = np.array([model(X) for X in X_samples]).flatten()
    data = pd.DataFrame(X_samples, columns=problem['names'])
    data['Y'] = Y
    return data
