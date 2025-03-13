import openturns as ot
import numpy as np

# The Chaboche mechanical model of a stress-strain curve
def function_of_interest(X):
    Strain, R, C, Gamma = X
    R *= 1.0e6  # Scale R into MPa
    C *= 1.0e6  # Scale C into MPa
    sigma = R - C * np.expm1(-Gamma * Strain) / Gamma
    sigma /= 1.0e6  # Un-scale sigma from MPa
    return [sigma]

model = ot.PythonFunction(4, 1, function_of_interest)

# Problem definition for the Chaboche Model
# Define distributions with corrected descriptions
Strain = ot.Uniform(0.0, 0.07)
Strain.setDescription(["Strain"])

R = ot.LogNormal()
R.setParameter(ot.LogNormalMuSigma()([750.0, 11.0, 0.0]))
R.setDescription(["R"])

C = ot.Normal(2750.0, 250.0)
C.setDescription(["C"])

Gamma = ot.Normal(10.0, 2.0)
Gamma.setDescription(["Gamma"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    Strain,
    R,
    C,
    Gamma
])
