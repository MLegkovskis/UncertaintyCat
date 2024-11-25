import numpy as np

# **Truss Model Function with Scaled Units**

def function_of_interest(X):
    # Extract variables
    E1 = X[0]  # Young's modulus E1 in GPa
    E2 = X[1]  # Young's modulus E2 in GPa
    A1 = X[2]  # Cross-sectional area A1 in cm²
    A2 = X[3]  # Cross-sectional area A2 in cm²
    P1 = X[4]  # Vertical load P1 in kN
    P2 = X[5]  # Vertical load P2 in kN
    P3 = X[6]  # Vertical load P3 in kN
    P4 = X[7]  # Vertical load P4 in kN
    P5 = X[8]  # Vertical load P5 in kN
    P6 = X[9]  # Vertical load P6 in kN
    
    # **Convert units to SI units**
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

    # **Prevent division by zero (optional but recommended)**
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

# **Problem Definition with Adjusted Units and Distributions**
problem = {
    'num_vars': 10,
    'names': ['E1', 'E2', 'A1', 'A2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
    'distributions': [
        {'type': 'LogNormalMuSigma', 'params': [210, 21, 0]},   # E1 in GPa
        {'type': 'LogNormalMuSigma', 'params': [210, 21, 0]},   # E2 in GPa
        {'type': 'LogNormalMuSigma', 'params': [20, 2, 0]},     # A1 in cm²
        {'type': 'LogNormalMuSigma', 'params': [10, 1, 0]},     # A2 in cm²
        {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P1 in kN
        {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P2 in kN
        {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P3 in kN
        {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P4 in kN
        {'type': 'Gumbel', 'params': [5.8477, 0.46622]},        # P5 in kN
        {'type': 'Gumbel', 'params': [5.8477, 0.46622]}         # P6 in kN
    ]
}

model = function_of_interest
