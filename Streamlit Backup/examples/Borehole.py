import openturns as ot
import numpy as np

# Borehole Model Function
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

model = ot.PythonFunction(8, 1, function_of_interest)

# Problem definition for the Borehole Model
# Define distributions with corrected descriptions
rw = ot.Normal(0.10, 0.0161812)
rw.setDescription(["rw"])

r = ot.LogNormal(7.71, 1.0056)
r.setDescription(["r"])

Tu = ot.Uniform(63070, 115600)
Tu.setDescription(["Tu"])

Hu = ot.Uniform(990, 1110)
Hu.setDescription(["Hu"])

Tl = ot.Uniform(63.1, 116)
Tl.setDescription(["Tl"])

Hl = ot.Uniform(700, 820)
Hl.setDescription(["Hl"])

L = ot.Uniform(1120, 1680)
L.setDescription(["L"])

Kw = ot.Uniform(9855, 12045)
Kw.setDescription(["Kw"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    rw,
    r,
    Tu,
    Hu,
    Tl,
    Hl,
    L,
    Kw
])