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
# Create LogNormal distributions using the original parameters
# Convert from mu_log, sigma_log to the appropriate OpenTURNS parameters

# m_p: Primary mass
m_p = ot.LogNormal()
mu_log = 0.4004899
sigma_log = 0.0997513
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
m_p.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
m_p.setDescription(["m_p"])

# m_s: Secondary mass
m_s = ot.LogNormal()
mu_log = -4.6101453
sigma_log = 0.0997513
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
m_s.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
m_s.setDescription(["m_s"])

# k_p: Primary stiffness
k_p = ot.LogNormal()
mu_log = -0.0196103
sigma_log = 0.1980422
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
k_p.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
k_p.setDescription(["k_p"])

# k_s: Secondary stiffness
k_s = ot.LogNormal()
mu_log = -4.6247805
sigma_log = 0.1980422
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
k_s.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
k_s.setDescription(["k_s"])

# zeta_p: Primary damping ratio
zeta_p = ot.LogNormal()
mu_log = -3.069942
sigma_log = 0.3856625
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
zeta_p.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
zeta_p.setDescription(["zeta_p"])

# zeta_s: Secondary damping ratio
zeta_s = ot.LogNormal()
mu_log = -4.023594
sigma_log = 0.4723807
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
zeta_s.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
zeta_s.setDescription(["zeta_s"])

# S_0: Excitation intensity
S_0 = ot.LogNormal()
mu_log = 4.600195
sigma_log = 0.0997513
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
S_0.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
S_0.setDescription(["S_0"])

# F_s: Force capacity
F_s = ot.LogNormal()
mu_log = 2.703075
sigma_log = 0.0997513
mu = np.exp(mu_log + sigma_log**2/2)  # Convert to mean of LogNormal
sigma = mu * np.sqrt(np.exp(sigma_log**2) - 1)  # Convert to std of LogNormal
F_s.setParameter(ot.LogNormalMuSigma()([mu, sigma, 0]))
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
