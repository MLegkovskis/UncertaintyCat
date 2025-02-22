import openturns as ot
import numpy as np

def function_of_interest(X):
    E, F, L, I = X
    Y = F * L**3 / (3 * E * I)
    return [Y]

model = ot.PythonFunction(4, 1, function_of_interest)

# Define distributions with corrected descriptions
E = ot.Beta(0.9, 3.1, 2.8e7, 4.8e7)
E.setDescription(["E"])

F = ot.LogNormal()
F.setParameter(ot.LogNormalMuSigma()([3.0e4, 9.0e3, 15.0e3]))
F.setDescription(["F"])

L = ot.Uniform(250.0, 260.0)
L.setDescription(["L"])

I = ot.Beta(2.5, 4.0, 310.0, 450.0)
I.setDescription(["I"])

# Define correlation matrix for dependent variables
R = ot.CorrelationMatrix(4)
R[2, 3] = -0.2  # Assuming L and I are correlated with a Spearman correlation of -0.2

# Define copula based on correlation matrix
copula = ot.NormalCopula(R)

# Define joint distribution with copula
problem = ot.JointDistribution([E, F, L, I], copula)
