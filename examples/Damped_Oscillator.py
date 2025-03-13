import openturns as ot
import numpy as np

# Damped Oscillator Performance Function

def function_of_interest(X):
    p = 3  # Safety factor parameter

    # Extract input variables
    m_p, m_s, k_p, k_s, zeta_p, zeta_s, S_0, F_s = X

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

model = ot.PythonFunction(8, 1, function_of_interest)

# Problem definition for the Damped Oscillator Model
# Define distributions with corrected descriptions
m_p = ot.LogNormal()
m_p.setParameter(ot.LogNormalMuSigma()([0.4004899, 0.0997513, 0]))
m_p.setDescription(["m_p"])

m_s = ot.LogNormal()
m_s.setParameter(ot.LogNormalMuSigma()([-4.6101453, 0.0997513, 0]))
m_s.setDescription(["m_s"])

k_p = ot.LogNormal()
k_p.setParameter(ot.LogNormalMuSigma()([-0.0196103, 0.1980422, 0]))
k_p.setDescription(["k_p"])

k_s = ot.LogNormal()
k_s.setParameter(ot.LogNormalMuSigma()([-4.6247805, 0.1980422, 0]))
k_s.setDescription(["k_s"])

zeta_p = ot.LogNormal()
zeta_p.setParameter(ot.LogNormalMuSigma()([-3.069942, 0.3856625, 0]))
zeta_p.setDescription(["zeta_p"])

zeta_s = ot.LogNormal()
zeta_s.setParameter(ot.LogNormalMuSigma()([-4.023594, 0.4723807, 0]))
zeta_s.setDescription(["zeta_s"])

S_0 = ot.LogNormal()
S_0.setParameter(ot.LogNormalMuSigma()([4.600195, 0.0997513, 0]))
S_0.setDescription(["S_0"])

F_s = ot.LogNormal()
F_s.setParameter(ot.LogNormalMuSigma()([2.703075, 0.0997513, 0]))
F_s.setDescription(["F_s"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    m_p,
    m_s,
    k_p,
    k_s,
    zeta_p,
    zeta_s,
    S_0,
    F_s
])
