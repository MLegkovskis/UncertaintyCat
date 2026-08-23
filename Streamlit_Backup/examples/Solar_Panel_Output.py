import openturns as ot
import numpy as np

# Solar Panel Output Function
def function_of_interest(X):
    G, A, eta, T_cell, T_ref, beta = X  # Solar irradiance, area, efficiency, cell temperature, reference temperature, temperature coefficient
    # Adjust efficiency based on temperature
    eta_adjusted = eta * (1 + beta * (T_cell - T_ref))
    # Power output
    P = G * A * eta_adjusted
    return [P]

model = ot.PythonFunction(6, 1, function_of_interest)

# Problem definition for the Solar Panel Output Model
# Define distributions with corrected descriptions
G = ot.Uniform(800, 1000)      # G: Solar irradiance (W/m²)
G.setDescription(["G"])

A = ot.Uniform(1.5, 2.0)        # A: Panel area (m²)
A.setDescription(["A"])

eta = ot.Uniform(0.15, 0.20)    # eta: Efficiency at T_ref (dimensionless)
eta.setDescription(["eta"])

T_cell = ot.Normal(45, 5)       # T_cell: Cell temperature (°C)
T_cell.setDescription(["T_cell"])

T_ref = ot.Uniform(25, 30)      # T_ref: Reference temperature (°C)
T_ref.setDescription(["T_ref"])

beta = ot.Uniform(-0.005, -0.003)  # beta: Temperature coefficient (1/°C)
beta.setDescription(["beta"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    G,
    A,
    eta,
    T_cell,
    T_ref,
    beta
])
