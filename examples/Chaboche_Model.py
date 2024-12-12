import numpy as np


# The Chaboche mechanical model of a stress-strain curve
def function_of_interest(X):
    Strain, R, C, Gamma = X
    R *= 1.0e6  # Scale R into MPa
    C *= 1.0e6  # Scale C into MPa
    sigma = R - C * np.expm1(-Gamma * Strain) / Gamma
    sigma /= 1.0e6  # Un-scale sigma from MPa
    return [sigma]


# Problem definition for the deflection model
problem = {
    "num_vars": 4,
    "names": ["Strain", "R", "C", "Gamma"],
    "distributions": [
        {"type": "Uniform", "params": [0.0, 0.07]},  # Strain
        {"type": "LogNormalMuSigma", "params": [750.0, 11.0, 0.0]},  # R
        {"type": "Normal", "params": [2750.0, 250.0]},  # C
        {"type": "Normal", "params": [10.0, 2.0]},  # Gamma
    ],
}

model = function_of_interest
