import numpy as np

def function_of_interest(X):
    m0, mf, ve, theta, g = X  # Initial mass, fuel mass, exhaust velocity, launch angle, gravity
    # Calculate the final velocity using Tsiolkovsky rocket equation
    delta_v = ve * np.log(m0 / (m0 - mf))
    # Calculate horizontal and vertical components
    vx = delta_v * np.cos(np.radians(theta))
    vy = delta_v * np.sin(np.radians(theta))
    # Calculate maximum altitude (assuming vertical motion under gravity)
    h_max = (vy ** 2) / (2 * g)
    # Calculate horizontal range
    range = (vx * vy) / g
    return [h_max]

# Problem definition
problem = {
    'num_vars': 5,
    'names': ['m0', 'mf', 've', 'theta', 'g'],
    'distributions': [
        {'type': 'Uniform', 'params': [50000, 60000]},    # m0: Initial mass (kg)
        {'type': 'Uniform', 'params': [10000, 20000]},    # mf: Fuel mass (kg)
        {'type': 'Normal', 'params': [3000, 100]},        # ve: Exhaust velocity (m/s)
        {'type': 'Uniform', 'params': [80, 90]},          # theta: Launch angle (degrees)
        {'type': 'Uniform', 'params': [9.8, 9.81]}        # g: Gravity (m/s²)
    ]
}

model = function_of_interest