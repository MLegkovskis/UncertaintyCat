import numpy as np

def function_of_interest(X):
    G, A, eta, T_cell, T_ref, beta = X  # Solar irradiance, area, efficiency, cell temperature, reference temperature, temperature coefficient
    # Adjust efficiency based on temperature
    eta_adjusted = eta * (1 + beta * (T_cell - T_ref))
    # Power output
    P = G * A * eta_adjusted
    return [P]

# Problem definition
problem = {
    'num_vars': 6,
    'names': ['G', 'A', 'eta', 'T_cell', 'T_ref', 'beta'],
    'distributions': [
        {'type': 'Uniform', 'params': [800, 1000]},       # G: Solar irradiance (W/m²)
        {'type': 'Uniform', 'params': [1.5, 2.0]},        # A: Panel area (m²)
        {'type': 'Uniform', 'params': [0.15, 0.20]},      # eta: Efficiency at T_ref (dimensionless)
        {'type': 'Normal', 'params': [45, 5]},            # T_cell: Cell temperature (°C)
        {'type': 'Uniform', 'params': [25, 30]},          # T_ref: Reference temperature (°C)
        {'type': 'Uniform', 'params': [-0.005, -0.003]}   # beta: Temperature coefficient (1/°C)
    ]
}

model = function_of_interest
