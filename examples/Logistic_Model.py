import numpy as np


# The logistic model of population growth
def function_of_interest(X):
    y0, a, c = X
    t0 = 0.0  # The initial time (years)
    t = 2000.0  # The final time (years)
    y0 *= 1.0e6  # Convert into millions
    b = np.exp(c)
    y = a * y0 / (b * y0 + (a - b * y0) * np.exp(-a * (t - t0)))
    y /= 1.0e6  # Convert from millions
    return [y]


# Problem definition for the deflection model
problem = {
    "num_vars": 3,
    "names": ["Y0", "A", "C"],
    "distributions": [
        {"type": "Normal", "params": [3.9, 1.0]},  # Y0 (initial population)
        {"type": "Normal", "params": [0.03, 0.01]},  # A
        {"type": "Normal", "params": [-22.6, 0.2]},  # C
    ],
}

model = function_of_interest
