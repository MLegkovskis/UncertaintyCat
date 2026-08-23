import openturns as ot
import numpy as np

# Ishigami Function
def function_of_interest(X):
    x1, x2, x3 = X
    Y = np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)
    return [Y]

model = ot.PythonFunction(3, 1, function_of_interest)

# Problem definition for the Ishigami Function
# Define distributions with corrected descriptions
x1 = ot.Uniform(-np.pi, np.pi)
x1.setDescription(["x1"])

x2 = ot.Uniform(-np.pi, np.pi)
x2.setDescription(["x2"])

x3 = ot.Uniform(-np.pi, np.pi)
x3.setDescription(["x3"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    x1,
    x2,
    x3
])
