import numpy as np

# Define your function here
def function_of_interest(X):
    gamma, phi, Rs, G, M = X
    b = 2.54e-9  # Constant value of b
    
    # Ensure sqrt_term is always non-negative by taking the max with zero
    sqrt_term = max((8.0 * gamma * phi * Rs) / (np.pi * G * pow(b, 2)), 0)
    
    # Compute sigma_c ensuring the value is always physical
    sigma_c = abs(((M * gamma) / (2.0 * b)) * (np.sqrt(sqrt_term) - phi) / 1e6)
    
    # Avoid negative or zero sigma_c in MPa values by adding a small positive constant if needed
    sigma_c = max(sigma_c, 1e-6)
    
    return [sigma_c]

# Problem definition
problem = {
    'num_vars': 5,
    'names': ['gamma', 'phi', 'Rs', 'G', 'M'],
    'distributions': [
        {'type': 'Uniform', 'params': [0.15, 0.25]},        # gamma
        {'type': 'Uniform', 'params': [0.30, 0.45]},        # phi
        {'type': 'Uniform', 'params': [1e-8, 3e-8]},        # Rs
        {'type': 'Uniform', 'params': [6e10, 8e10]},        # G
        {'type': 'Normal', 'params': [3.05, 0.15]}          # M
    ]
}

model = function_of_interest
