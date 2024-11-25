import numpy as np

def function_of_interest(X):
    k0, E, R, T, CA0, V = X  # Arrhenius parameters and reactor conditions
    # Calculate the rate constant using Arrhenius equation
    k = k0 * np.exp(-E / (R * T))
    # Assume first-order reaction: A → Products
    # Use the design equation for a continuous stirred-tank reactor (CSTR)
    Q = 1.0  # Flow rate in m³/s
    CA = CA0 / (1 + k * V / Q)
    # Conversion X = (CA0 - CA) / CA0
    conversion = (CA0 - CA) / CA0
    return [conversion]

# Problem definition
problem = {
    'num_vars': 6,
    'names': ['k0', 'E', 'R', 'T', 'CA0', 'V'],
    'distributions': [
        {'type': 'LogNormalMuSigma', 'params': [1e10, 1e9, 0]},   # k0: Pre-exponential factor
        {'type': 'Uniform', 'params': [80000, 120000]},           # E: Activation energy (J/mol)
        {'type': 'Uniform', 'params': [8.314, 8.514]},            # R: Gas constant (J/(mol·K))
        {'type': 'Normal', 'params': [350, 5]},                   # T: Temperature (K)
        {'type': 'Uniform', 'params': [1.0, 2.0]},                # CA0: Initial concentration (mol/m³)
        {'type': 'Uniform', 'params': [1.0, 5.0]}                 # V: Reactor volume (m³)
    ]
}

model = function_of_interest
