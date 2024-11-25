import numpy as np
from scipy.integrate import solve_ivp

# Define the function of interest for the heat transfer model
def function_of_interest(X):
    radius, length, k, h, Q = X  # Inputs: radius, length, thermal conductivity, convective heat transfer coefficient, heat generation rate

    # Adjust heat generation rate by volume according to the length
    Q_adjusted = Q / length

    # Differential equation for radial temperature distribution
    def heat_eq(r, T):
        # Here, T[0] is the temperature and T[1] is its radial derivative dT/dr
        dTdr = T[1]
        d2Tdr2 = -h * (T[0] - 300) / (k * r) + Q_adjusted / k  # 300 K as the boundary surface temperature
        return [dTdr, d2Tdr2]

    # Initial conditions at the center (r=0 can be singular, so start slightly off-center)
    r_min = 1e-5  # start slightly off-center to avoid singularity at r=0
    initial_conditions = [300, 0]  # assume the center starts at the boundary temperature, gradient is zero

    # Solve the differential equation from near the center of the cylinder to its surface
    sol = solve_ivp(heat_eq, [r_min, radius], initial_conditions, method='BDF')

    # Return the maximum temperature, which is typically at the center
    return [max(sol.y[0])]

# Problem definition for the thermal model
problem = {
    'num_vars': 5,
    'names': ['radius', 'length', 'k', 'h', 'Q'],
    'distributions': [
        {'type': 'Uniform', 'params': [0.01, 0.1]},      # radius in meters
        {'type': 'Uniform', 'params': [0.1, 1.0]},       # length in meters
        {'type': 'Uniform', 'params': [15, 50]},         # thermal conductivity in W/(m·K)
        {'type': 'Uniform', 'params': [5, 25]},          # convective heat transfer coefficient in W/(m²·K)
        {'type': 'Uniform', 'params': [1000, 5000]}      # heat generation rate per volume in W/m³
    ]
}

model = function_of_interest