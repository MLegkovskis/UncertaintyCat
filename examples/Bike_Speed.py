import numpy as np
from scipy.optimize import fsolve

# Define the function that represents the non-linear equation
def bike_speed_eq(speed, *params):
    rider_power, air_density, drag_coefficient, frontal_area, rolling_resistance, bike_mass, gravity_earth = params
    return rider_power - (
        0.5 * air_density * drag_coefficient * frontal_area * speed**3 +
        rolling_resistance * bike_mass * gravity_earth * speed
    )

# Wrapper function to solve for the speed
def function_of_interest(inputs):
    tire_radius, bike_mass, rider_power, air_density, rolling_resistance, drag_coefficient, gravity_earth, frontal_area = inputs
    initial_guess = 10  # Initial guess for the speed (m/s)
    params = (rider_power, air_density, drag_coefficient, frontal_area, rolling_resistance, bike_mass, gravity_earth)
    speed_solution = fsolve(bike_speed_eq, initial_guess, args=params)
    return [speed_solution[0]]

# Problem definition
problem = {
    'num_vars': 8,
    'names': ["tire_radius", "bike_mass", "rider_power", "air_density", "rolling_resistance", "drag_coefficient", "gravity_earth", "frontal_area"],
    'distributions': [
        {'type': 'Uniform', 'params': [0.3, 0.7]},        # tire_radius: Uniform distribution (meters)
        {'type': 'Uniform', 'params': [5.0, 15.0]},       # bike_mass: Uniform distribution (kg)
        {'type': 'Uniform', 'params': [150.0, 400.0]},    # rider_power: Uniform distribution (Watts)
        {'type': 'Uniform', 'params': [1.18, 1.3]},       # air_density: Uniform distribution (kg/m^3)
        {'type': 'Uniform', 'params': [0.002, 0.005]},    # rolling_resistance: Uniform distribution (dimensionless)
        {'type': 'Uniform', 'params': [0.7, 1.0]},        # drag_coefficient: Uniform distribution (dimensionless)
        {'type': 'Uniform', 'params': [9.78, 9.82]},      # gravity_earth: Uniform distribution (m/s^2)
        {'type': 'Uniform', 'params': [0.3, 0.6]}         # frontal_area: Uniform distribution (m^2)
    ]
}

model = function_of_interest