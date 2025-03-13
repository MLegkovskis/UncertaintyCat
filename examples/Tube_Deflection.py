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
F = ot.Normal()
F.setParameter(ot.LogNormalMuSigma()([1.0, 0.1, 0]))
F.setDescription(["F"])

L = ot.Normal()
L.setParameter(ot.LogNormalMuSigma()([1.5, 0.01, 0]))
L.setDescription(["L"])

a = ot.Uniform(0.7, 1.2)
a.setDescription(["a"])

De = ot.Uniform(0.75, 0.85)
De.setDescription(["De"])

di = ot.Uniform(0.09, 0.11)
di.setDescription(["di"])

E = ot.Normal()
E.setParameter(ot.LogNormalMuSigma()([200000, 2000, 0]))
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
