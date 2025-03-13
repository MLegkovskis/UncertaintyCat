import openturns as ot
import numpy as np

# Wind Turbine Power Function
def function_of_interest(X):
    rho_air, A, Cp, v_wind, eta, beta = X  # Air density, rotor area, power coefficient, wind speed, efficiency, blade pitch angle
    # Adjust power coefficient based on blade pitch angle
    Cp_adjusted = Cp * np.cos(np.radians(beta))
    # Power extracted from wind
    P = 0.5 * rho_air * A * Cp_adjusted * v_wind ** 3 * eta
    return [P]

model = ot.PythonFunction(6, 1, function_of_interest)

# Problem definition for the Wind Turbine Power Model
# Define distributions with corrected descriptions
rho_air = ot.Uniform(1.225, 1.25)    # rho_air: Air density (kg/m³)
rho_air.setDescription(["rho_air"])

A = ot.Uniform(1000, 2000)           # A: Rotor swept area (m²)
A.setDescription(["A"])

Cp = ot.Uniform(0.4, 0.5)            # Cp: Power coefficient (dimensionless)
Cp.setDescription(["Cp"])

v_wind = ot.Normal()
v_wind.setParameter(ot.NormalMuSigma()([8, 2, 0]))  # Assuming LogNormalMuSigma is not applicable here
v_wind.setDescription(["v_wind"])

eta = ot.Uniform(0.9, 0.95)          # eta: Efficiency (dimensionless)
eta.setDescription(["eta"])

beta = ot.Uniform(-5, 5)             # beta: Blade pitch angle (degrees)
beta.setDescription(["beta"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    rho_air,
    A,
    Cp,
    v_wind,
    eta,
    beta
])
