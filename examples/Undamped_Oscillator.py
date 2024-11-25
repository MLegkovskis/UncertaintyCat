import numpy as np

# **Undamped (Gayton) Oscillator Performance Function**
# 
# *Description:*
# This model evaluates the performance of an undamped oscillator subjected to a transient force. It calculates the displacement response and compares it to the yield displacement to determine if the system remains within safe operating limits. A positive value of the performance function `Y` indicates safe operation.

def function_of_interest(X):
    # Extract input variables
    C_1 = X[0]  # First spring constant (N/m)
    C_2 = X[1]  # Second spring constant (N/m)
    M = X[2]    # Oscillator mass (kg)
    R = X[3]    # Yield displacement (m)
    F_1 = X[4]  # Applied force (N)
    T_1 = X[5]  # Duration of applied force (s)

    # Compute natural frequency of the oscillator
    omega_0 = np.sqrt((C_1 + C_2) / M)  # Natural frequency (rad/s)

    # Compute the performance function
    Y = 3 * R - np.abs(2 * F_1 / (M * omega_0**2) * np.sin(omega_0 * T_1 / 2))

    return [Y]

# **Problem Definition for the Undamped Oscillator Model**
problem = {
    'num_vars': 6,
    'names': ['C_1', 'C_2', 'M', 'R', 'F_1', 'T_1'],
    'distributions': [
        {'type': 'Normal', 'params': [1.0, 0.1]},   # C_1: First spring constant
        {'type': 'Normal', 'params': [0.1, 0.01]},  # C_2: Second spring constant
        {'type': 'Normal', 'params': [1.0, 0.05]},  # M: Oscillator mass
        {'type': 'Normal', 'params': [0.5, 0.05]},  # R: Yield displacement
        {'type': 'Normal', 'params': [0.6, 0.1]},   # F_1: Applied force
        {'type': 'Normal', 'params': [1.0, 0.2]}    # T_1: Duration of force
    ]
}

model = function_of_interest
