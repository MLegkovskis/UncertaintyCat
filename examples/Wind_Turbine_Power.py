import numpy as np

def function_of_interest(X):
    rho_air, A, Cp, v_wind, eta, beta = X  # Air density, rotor area, power coefficient, wind speed, efficiency, blade pitch angle
    # Adjust power coefficient based on blade pitch angle
    Cp_adjusted = Cp * np.cos(np.radians(beta))
    # Power extracted from wind
    P = 0.5 * rho_air * A * Cp_adjusted * v_wind ** 3 * eta
    return [P]

# Problem definition
problem = {
    'num_vars': 6,
    'names': ['rho_air', 'A', 'Cp', 'v_wind', 'eta', 'beta'],
    'distributions': [
        {'type': 'Uniform', 'params': [1.225, 1.25]},     # rho_air: Air density (kg/m³)
        {'type': 'Uniform', 'params': [1000, 2000]},      # A: Rotor swept area (m²)
        {'type': 'Uniform', 'params': [0.4, 0.5]},        # Cp: Power coefficient (dimensionless)
        {'type': 'Normal', 'params': [8, 2]},             # v_wind: Wind speed (m/s), replacing Weibull with Normal distribution
        {'type': 'Uniform', 'params': [0.9, 0.95]},       # eta: Efficiency (dimensionless)
        {'type': 'Uniform', 'params': [-5, 5]}            # beta: Blade pitch angle (degrees)
    ]
}

model = function_of_interest
