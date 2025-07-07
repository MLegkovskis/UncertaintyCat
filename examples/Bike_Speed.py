import openturns as ot
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

model = ot.PythonFunction(8, 1, function_of_interest)

# Problem definition
# Define distributions with corrected descriptions
tire_radius = ot.Uniform(0.3, 0.7)
tire_radius.setDescription(["tire_radius"])

bike_mass = ot.Uniform(5.0, 15.0)
bike_mass.setDescription(["bike_mass"])

rider_power = ot.Uniform(150.0, 400.0)
rider_power.setDescription(["rider_power"])

air_density = ot.Uniform(1.18, 1.3)
air_density.setDescription(["air_density"])

rolling_resistance = ot.Uniform(0.002, 0.005)
rolling_resistance.setDescription(["rolling_resistance"])

drag_coefficient = ot.Uniform(0.7, 1.0)
drag_coefficient.setDescription(["drag_coefficient"])

gravity_earth = ot.Uniform(9.78, 9.82)
gravity_earth.setDescription(["gravity_earth"])

frontal_area = ot.Uniform(0.3, 0.6)
frontal_area.setDescription(["frontal_area"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    tire_radius,
    bike_mass,
    rider_power,
    air_density,
    rolling_resistance,
    drag_coefficient,
    gravity_earth,
    frontal_area
])
