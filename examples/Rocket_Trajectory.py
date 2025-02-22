import openturns as ot
import numpy as np

# Rocket Trajectory Function
def function_of_interest(X):
    m0, mf, ve, theta, g = X  # Initial mass, fuel mass, exhaust velocity, launch angle, gravity
    # Calculate the final velocity using Tsiolkovsky rocket equation
    delta_v = ve * np.log(m0 / (m0 - mf))
    # Calculate horizontal and vertical components
    vx = delta_v * np.cos(np.radians(theta))
    vy = delta_v * np.sin(np.radians(theta))
    # Calculate maximum altitude (assuming vertical motion under gravity)
    h_max = (vy ** 2) / (2 * g)
    # Calculate horizontal range
    range_ = (vx * vy) / g
    return [h_max]

model = ot.PythonFunction(5, 1, function_of_interest)

# Problem definition for the Rocket Trajectory Model
# Define distributions with corrected descriptions
m0 = ot.Uniform(50000, 60000)    # m0: Initial mass (kg)
m0.setDescription(["m0"])

mf = ot.Uniform(10000, 20000)    # mf: Fuel mass (kg)
mf.setDescription(["mf"])

ve = ot.Normal()
ve.setParameter(ot.LogNormalMuSigma()([3000, 100, 0]))  # Assuming LogNormalMuSigma is applicable
ve.setDescription(["ve"])

theta = ot.Uniform(80, 90)        # theta: Launch angle (degrees)
theta.setDescription(["theta"])

g = ot.Uniform(9.8, 9.81)         # g: Gravity (m/s²)
g.setDescription(["g"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    m0,
    mf,
    ve,
    theta,
    g
])
