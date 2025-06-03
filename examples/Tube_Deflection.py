import openturns as ot
import numpy as np

# Deflection of a Tube Model
def function_of_interest(X):
    F, L, a, De, di, E = X
    I = np.pi * (De**4 - di**4) / 32
    b = L - a
    deflection = -F * a**2 * (L - a) ** 2 / (3 * E * L * I)
    return [deflection]

model = ot.PythonFunction(6, 1, function_of_interest)

# Problem definition for the Tube Deflection Model
# Define distributions with corrected descriptions
F = ot.Normal(1.0, 0.1)  # Force
F.setDescription(["F"])

L = ot.Normal(1.5, 0.01)  # Length
L.setDescription(["L"])

a = ot.Uniform(0.7, 1.2)  # Position of force
a.setDescription(["a"])

De = ot.Uniform(0.75, 0.85)  # External diameter
De.setDescription(["De"])

di = ot.Uniform(0.09, 0.11)  # Internal diameter
di.setDescription(["di"])

E = ot.Normal(200000, 2000)  # Young's modulus
E.setDescription(["E"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    F,
    L,
    a,
    De,
    di,
    E
])
