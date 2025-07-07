import openturns as ot
import numpy as np

# Viscous Freefall Function
def function_of_interest(X):
    z0, v0, m, c = X
    g = 9.81  # Gravitational acceleration
    tau = m / c  # Time characteristic
    vinf = -m * g / c  # Limit speed
    number_of_vertices = 100
    t = np.linspace(0.0, 12.0, number_of_vertices)
    z = z0 + vinf * t + tau * (v0 - vinf) * (1 - np.exp(-t / tau))
    z = np.max(z)
    return [z]

model = ot.PythonFunction(4, 1, function_of_interest)

# Problem definition for the Viscous Freefall Model
# Define distributions with corrected descriptions
z0 = ot.Uniform(100.0, 150.0)  # z0
z0.setDescription(["z0"])

v0 = ot.Normal(55.0, 10.0)  # Initial velocity
v0.setDescription(["v0"])

m = ot.Normal(80.0, 8.0)  # Mass
m.setDescription(["m"])

c = ot.Uniform(0.0, 30.0)  # Drag coefficient
c.setDescription(["c"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    z0,
    v0,
    m,
    c
])
