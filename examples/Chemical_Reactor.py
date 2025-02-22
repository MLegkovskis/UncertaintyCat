import openturns as ot
import numpy as np

# Chemical Reactor Function
def function_of_interest(X):
    k0, E, R, T, CA0, V = X  # Arrhenius parameters and reactor conditions
    # Calculate the rate constant using Arrhenius equation
    k = k0 * np.exp(-E / (R * T))
    # Assume first-order reaction: A → Products
    # Use the design equation for a continuous stirred-tank reactor (CSTR)
    Q = 1.0  # Flow rate in m³/s
    CA = CA0 / (1 + k * V / Q)
    # Conversion X = (CA0 - CA) / CA0
    conversion = (CA0 - CA) / CA0
    return [conversion]

model = ot.PythonFunction(6, 1, function_of_interest)

# Problem definition for the Chemical Reactor
# Define distributions with corrected descriptions
k0 = ot.LogNormal()
k0.setParameter(ot.LogNormalMuSigma()([1e10, 1e9, 0]))
k0.setDescription(["k0"])

E = ot.Uniform(80000, 120000)
E.setDescription(["E"])

R = ot.Uniform(8.314, 8.514)
R.setDescription(["R"])

T = ot.Normal(350, 5)
T.setDescription(["T"])

CA0 = ot.Uniform(1.0, 2.0)
CA0.setDescription(["CA0"])

V = ot.Uniform(1.0, 5.0)
V.setDescription(["V"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    k0,
    E,
    R,
    T,
    CA0,
    V
])
