import numpy as np


# The viscous free fall of a mass under gravity
def function_of_interest(X):
    g = 9.81  # Gravitational acceleration
    z0, v0, m, c = X
    tau = m / c  # Time caracteristic
    vinf = -m * g / c  # Limit speed
    number_of_vertices = 100
    t = np.linspace(0.0, 12.0, number_of_vertices)
    z = z0 + vinf * t + tau * (v0 - vinf) * (1 - np.exp(-t / tau))
    z = np.max(z)
    return [z]


# Problem definition for the deflection model
problem = {
    "num_vars": 4,
    "names": ["z0", "v0", "m", "c"],
    "distributions": [
        {"type": "Uniform", "params": [100.0, 150.0]},  # Z0
        {"type": "Normal", "params": [55.0, 10.0]},  # V0
        {"type": "Normal", "params": [80.0, 8.0]},  # M
        {"type": "Uniform", "params": [0.0, 30.0]},  # C
    ],
}

model = function_of_interest
