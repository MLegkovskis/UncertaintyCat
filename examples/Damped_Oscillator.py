import numpy as np

# **Damped Oscillator Performance Function**
# 
# *Description:*
# This model computes the performance of a damped oscillator system by evaluating the force capacity of the secondary spring under dynamic loading. It considers the masses, stiffnesses, damping ratios, and excitation intensity to calculate the performance function `g`. A positive value of `g` indicates that the system meets the performance criteria (safe), while a negative value indicates failure.

def function_of_interest(X):
    p = 3  # Safety factor parameter

    # Extract input variables
    m_p = X[0]    # Primary mass (kg)
    m_s = X[1]    # Secondary mass (kg)
    k_p = X[2]    # Stiffness of the primary spring (N/m)
    k_s = X[3]    # Stiffness of the secondary spring (N/m)
    zeta_p = X[4] # Damping ratio of the primary damper
    zeta_s = X[5] # Damping ratio of the secondary damper
    S_0 = X[6]    # Excitation intensity (m²/s³)
    F_s = X[7]    # Force capacity of the secondary spring (N)

    # Compute natural frequencies
    omega_p = np.sqrt(k_p / m_p)  # Primary natural frequency (rad/s)
    omega_s = np.sqrt(k_s / m_s)  # Secondary natural frequency (rad/s)

    # Compute other parameters
    gamma = m_s / m_p                     # Mass ratio
    omega_a = (omega_p + omega_s) / 2     # Average natural frequency
    zeta_a = (zeta_p + zeta_s) / 2        # Average damping ratio
    theta = (omega_p - omega_s) / omega_a # Tuning parameter

    # Compute mean-square relative displacement
    numerator = np.pi * S_0 / (4 * zeta_s * omega_s**3)
    denominator = (zeta_p * zeta_s * (4 * zeta_a**2 + theta**2) + gamma * zeta_a**2)
    factor = (zeta_a * zeta_s) / denominator
    term = (zeta_p * omega_p**3 + zeta_s * omega_s**3) * omega_p / (4 * zeta_a * omega_a**4)
    mean_square_displacement = numerator * factor * term

    # Evaluate the performance function
    g = F_s - p * k_s * np.sqrt(mean_square_displacement)

    return [g]

# **Problem Definition for the Damped Oscillator Model**
problem = {
    'num_vars': 8,
    'names': ['m_p', 'm_s', 'k_p', 'k_s', 'zeta_p', 'zeta_s', 'S_0', 'F_s'],
    'distributions': [
        # LogNormal distributions with parameters [mu_log, sigma_log, gamma=0]
        {'type': 'LogNormal', 'params': [0.4004899, 0.0997513, 0]},   # m_p: Primary mass
        {'type': 'LogNormal', 'params': [-4.6101453, 0.0997513, 0]},  # m_s: Secondary mass
        {'type': 'LogNormal', 'params': [-0.0196103, 0.1980422, 0]},  # k_p: Primary stiffness
        {'type': 'LogNormal', 'params': [-4.6247805, 0.1980422, 0]},  # k_s: Secondary stiffness
        {'type': 'LogNormal', 'params': [-3.069942, 0.3856625, 0]},   # zeta_p: Primary damping ratio
        {'type': 'LogNormal', 'params': [-4.023594, 0.4723807, 0]},   # zeta_s: Secondary damping ratio
        {'type': 'LogNormal', 'params': [4.600195, 0.0997513, 0]},    # S_0: Excitation intensity
        {'type': 'LogNormal', 'params': [2.703075, 0.0997513, 0]}     # F_s: Force capacity
    ]
}

model = function_of_interest
