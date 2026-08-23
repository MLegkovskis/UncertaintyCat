import openturns as ot
import numpy as np

# The logistic model of population growth
def function_of_interest(X):
    y0, a, c = X
    t0 = 0.0  # The initial time (years)
    t = 2000.0  # The final time (years)
    y0 *= 1.0e6  # Convert into millions
    b = np.exp(c)
    y = a * y0 / (b * y0 + (a - b * y0) * np.exp(-a * (t - t0)))
    y /= 1.0e6  # Convert from millions
    return [y]

model = ot.PythonFunction(3, 1, function_of_interest)

# Problem definition for the Logistic Model
# Define distributions with corrected descriptions
Y0 = ot.Normal(3.9, 1.0)         # Y0: Initial population (millions)
Y0.setDescription(["Y0"])

A = ot.Normal(0.03, 0.01)        # A
A.setDescription(["A"])

C = ot.Normal(-22.6, 0.2)        # C
C.setDescription(["C"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    Y0,
    A,
    C
])
