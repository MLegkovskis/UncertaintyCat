import openturns as ot
import numpy as np

# Truss Model Function with Scaled Units

def function_of_interest(X):
    # Extract variables
    E1, E2, A1, A2, P1, P2, P3, P4, P5, P6 = X

    # Convert units to SI units
    E1 = E1 * 1e9     # GPa to Pa
    E2 = E2 * 1e9     # GPa to Pa
    A1 = A1 * 1e-4    # cm² to m²
    A2 = A2 * 1e-4    # cm² to m²
    P1 = P1 * 1e3     # kN to N
    P2 = P2 * 1e3     # kN to N
    P3 = P3 * 1e3     # kN to N
    P4 = P4 * 1e3     # kN to N
    P5 = P5 * 1e3     # kN to N
    P6 = P6 * 1e3     # kN to N

    # Prevent division by zero (optional but recommended)
    epsilon = 1e-12
    E1 = np.maximum(E1, epsilon)
    E2 = np.maximum(E2, epsilon)
    A1 = np.maximum(A1, epsilon)
    A2 = np.maximum(A2, epsilon)

    # Compute the sums for the deflection formula
    sum1 = 2 * P1 + 6 * P2 + 10 * P3 + 10 * P4 + 6 * P5 + 2 * P6
    sum2 = 36 * P1 + 100 * P2 + 140 * P3 + 140 * P4 + 100 * P5 + 36 * P6

    # Compute the midspan deflection 'y'
    y = - (np.sqrt(2) * sum1) / (A2 * E2) - sum2 / (A1 * E1)

    return [y]

model = ot.PythonFunction(10, 1, function_of_interest)

# Problem definition for the Truss Model
# Define distributions with corrected descriptions
E1 = ot.LogNormal()
E1.setParameter(ot.LogNormalMuSigma()([210, 21, 0]))
E1.setDescription(["E1"])

E2 = ot.LogNormal()
E2.setParameter(ot.LogNormalMuSigma()([210, 21, 0]))
E2.setDescription(["E2"])

A1 = ot.LogNormal()
A1.setParameter(ot.LogNormalMuSigma()([20, 2, 0]))
A1.setDescription(["A1"])

A2 = ot.LogNormal()
A2.setParameter(ot.LogNormalMuSigma()([10, 1, 0]))
A2.setDescription(["A2"])

P1 = ot.Gumbel(5.8477, 0.46622)  # P1 in kN
P1.setDescription(["P1"])

P2 = ot.Gumbel(5.8477, 0.46622)  # P2 in kN
P2.setDescription(["P2"])

P3 = ot.Gumbel(5.8477, 0.46622)  # P3 in kN
P3.setDescription(["P3"])

P4 = ot.Gumbel(5.8477, 0.46622)  # P4 in kN
P4.setDescription(["P4"])

P5 = ot.Gumbel(5.8477, 0.46622)  # P5 in kN
P5.setDescription(["P5"])

P6 = ot.Gumbel(5.8477, 0.46622)  # P6 in kN
P6.setDescription(["P6"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    E1,
    E2,
    A1,
    A2,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6
])
