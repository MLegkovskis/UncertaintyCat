import numpy as np
import openturns as ot
###############################################################################
# 1. Define the function_of_interest (the reciprocal tariff calculation)
###############################################################################
def function_of_interest(X):
    """
    Given [x, m, eps, phi],
    compute the reciprocal tariff that drives the trade balance to zero:
    
      Delta_tau = (x - m) / (eps * phi * m),
    
    and then left-censor at zero (i.e., negative rates become zero).
    """
    x, m, eps, phi = X
    dTau = (x - m) / (eps * phi * m)
    # Left-censor at zero
    if dTau < 0.0:
        dTau = 0.0
    return [dTau]

# Wrap the function into an OpenTURNS PythonFunction
model = ot.PythonFunction(4, 1, function_of_interest)

###############################################################################
# 2. Define the random input variables
###############################################################################

# -- x (exports) --
#   For illustration, assume x can range widely from 1e8 to 1e9 (units: US dollars)
xDist = ot.Uniform(1.0e8, 1.0e9)
xDist.setDescription(["x"])  # descriptive name

# -- m (imports) --
#   Similarly, let m vary from 1e8 to 1e9
mDist = ot.Uniform(1.0e8, 1.0e9)
mDist.setDescription(["m"])

# -- eps (elasticity), around 4 --
#   Use a truncated Normal distribution with mean=4, std=1, truncated to remain > 0
#   (OpenTURNS supports ParametrizedDistribution or TruncatedDistribution)
mean_eps = 4.0
sigma_eps = 1.0
epsNormal = ot.Normal(mean_eps, sigma_eps)
# Truncate below at 0.1 just to avoid negative or near-zero elasticity
epsDist = ot.TruncatedDistribution(epsNormal, 0.1, ot.TruncatedDistribution.LOWER)
epsDist.setDescription(["epsilon"])

# -- phi (pass-through), around 0.25 --
#   Assume uniform from 0.1 to 0.4
phiDist = ot.Uniform(0.1, 0.4)
phiDist.setDescription(["phi"])

###############################################################################
# 3. Build the joint distribution
###############################################################################
# Assume independence among x, m, eps, phi
inputDistributions = [xDist, mDist, epsDist, phiDist]
problem = ot.JointDistribution(inputDistributions)