import openturns as ot
import numpy as np

# Material Stress Function
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

model = ot.PythonFunction(5, 1, function_of_interest)

# Problem definition for the Material Stress Model
# Define distributions with corrected descriptions
gamma = ot.Uniform(0.15, 0.25)
gamma.setDescription(["gamma"])

phi = ot.Uniform(0.30, 0.45)
phi.setDescription(["phi"])

Rs = ot.Uniform(1e-8, 3e-8)
Rs.setDescription(["Rs"])

G = ot.Uniform(6e10, 8e10)
G.setDescription(["G"])

M = ot.Normal(3.05, 0.15)
M.setDescription(["M"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    gamma,
    phi,
    Rs,
    G,
    M
])
