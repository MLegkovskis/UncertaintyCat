import numpy as np


# Deflection of a tube model
def function_of_interest(X):
    F, L, a, De, di, E = X
    I = np.pi * (De ^ 4 - di ^ 4) / 32
    b = L - a
    deflection = -F * a ^ 2 * (L - a) ^ 2 / (3 * E * L * I)
    return [deflection]


# Problem definition for the deflection model
problem = {
    "num_vars": 6,
    "names": ["F", "L", "a", "De", "di", "E"],
    "distributions": [
        {"type": "Normal", "params": [1.0, 0.1]},  # F
        {"type": "Normal", "params": [1.5, 0.01]},  # L
        {"type": "Uniform", "params": [0.7, 1.2]},  # a
        {"type": "Uniform", "params": [0.75, 0.85]},  # De
        {"type": "Uniform", "params": [0.09, 0.11]},  # di
        {"type": "Normal", "params": [200000, 2000]},  # E
    ],
}

model = function_of_interest
