import numpy as np

# **Borehole Model Function**
# 
# *Description:*
# This function models the water flow rate through a borehole penetrating two aquifers. It calculates the water flow based on eight input variables representing physical properties of the borehole and aquifer system. The model is commonly used in reliability and sensitivity analyses due to its nonlinear behavior and interaction between variables.

def function_of_interest(X):
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X
    # rw  # Radius of borehole (m)
    # r  # Radius of influence (m)
    # Tu   # Transmissivity of upper aquifer (m²/year)
    # Hu  # Potentiometric head of upper aquifer (m)
    # Tl  # Transmissivity of lower aquifer (m²/year)
    # Hl # Potentiometric head of lower aquifer (m)
    # L   # Length of borehole (m)
    # Kw  # Hydraulic conductivity of borehole (m/year)

    # Compute intermediate terms
    log_r_rw = np.log(r / rw)
    numerator = 2 * np.pi * Tu * (Hu - Hl)
    denominator = log_r_rw * (1 + (2 * L * Tu) / (log_r_rw * rw**2 * Kw) + Tu / Tl)
    
    # Calculate water flow rate through the borehole
    Y = numerator / denominator  # Flow rate (m³/year)

    return [Y]

# **Problem Definition for the Borehole Model**
problem = {
    'num_vars': 8,
    'names': ['rw', 'r', 'Tu', 'Hu', 'Tl', 'Hl', 'L', 'Kw'],
    'distributions': [
        {'type': 'Normal', 'params': [0.10, 0.0161812]},        # rw: Normally distributed
        {'type': 'LogNormal', 'params': [7.71, 1.0056, 0]},     # r: Log-normally distributed
        {'type': 'Uniform', 'params': [63070, 115600]},         # Tu: Uniformly distributed
        {'type': 'Uniform', 'params': [990, 1110]},             # Hu: Uniformly distributed
        {'type': 'Uniform', 'params': [63.1, 116]},             # Tl: Uniformly distributed
        {'type': 'Uniform', 'params': [700, 820]},              # Hl: Uniformly distributed
        {'type': 'Uniform', 'params': [1120, 1680]},            # L: Uniformly distributed
        {'type': 'Uniform', 'params': [9855, 12045]}            # Kw: Uniformly distributed
    ]
}

model = function_of_interest
