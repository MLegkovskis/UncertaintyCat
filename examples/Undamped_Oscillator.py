import openturns as ot
import numpy as np

# Undamped (Gayton) Oscillator Performance Function

def function_of_interest(X):
    # Extract input variables
    C_1, C_2, M, R, F_1, T_1 = X

    # Compute natural frequency of the oscillator
    omega_0 = np.sqrt((C_1 + C_2) / M)  # Natural frequency (rad/s)

    # Compute the performance function
    Y = 3 * R - np.abs(2 * F_1 / (M * omega_0**2) * np.sin(omega_0 * T_1 / 2))

    return [Y]

model = ot.PythonFunction(6, 1, function_of_interest)

# Problem definition for the Undamped Oscillator Model
# Define distributions with corrected descriptions
C_1 = ot.Normal()
C_1.setParameter(ot.NormalMuSigma()([1.0, 0.1, 0]))
C_1.setDescription(["C_1"])

C_2 = ot.Normal()
C_2.setParameter(ot.NormalMuSigma()([0.1, 0.01, 0]))
C_2.setDescription(["C_2"])

M = ot.Normal()
M.setParameter(ot.NormalMuSigma()([1.0, 0.05, 0]))
M.setDescription(["M"])

R = ot.Normal()
R.setParameter(ot.NormalMuSigma()([0.5, 0.05, 0]))
R.setDescription(["R"])

F_1 = ot.Normal()
F_1.setParameter(ot.NormalMuSigma()([0.6, 0.1, 0]))
F_1.setDescription(["F_1"])

T_1 = ot.Normal()
T_1.setParameter(ot.NormalMuSigma()([1.0, 0.2, 0]))
T_1.setDescription(["T_1"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    C_1,
    C_2,
    M,
    R,
    F_1,
    T_1
])
