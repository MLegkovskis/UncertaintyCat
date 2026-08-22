import openturns as ot
import numpy as np

def function_of_interest(X):
    # Inputs:
    # Q: Flow rate
    # Ks: Strickler coefficient
    # Zv: Downstream height
    # Zm: Upstream height
    # B: River width
    # L: River length
    # Zb: River bank altitude
    # Hd: Dyke height
    Q, Ks, Zv, Zm, B, L, Zb, Hd = X
    
    # 1. Calculate the slope alpha
    # Use max(0, ...) to ensure no domain errors if sample makes Zv > Zm
    alpha = (Zm - Zv) / L if Zm >= Zv else 0.0
    
    # 2. Calculate the water depth H using Manning-Strickler
    if Ks > 0 and Q > 0 and alpha > 0:
        H = (Q / (Ks * B * np.sqrt(alpha))) ** 0.6
    else:
        H = 0.0
    
    # 3. Calculate flood altitude Zc and dyke altitude Zd
    Zc = H + Zv
    Zd = Zb + Hd
    
    # 4. Determine Overflow S
    # S > 0 means the water is higher than the dyke (FLOOD)
    S = Zc - Zd
    
    return [S]

model = ot.PythonFunction(8, 1, function_of_interest)

# Problem definition
# ------------------

# Q: Gumbel(scale=558, mode=1013)
# Note: OpenTURNS constructor is Gumbel(scale, mode)
Q = ot.Gumbel(558.0, 1013.0)
Q.setDescription(["Q (Flow Rate)"])

# Ks: Normal(mu=30, sigma=7.5)
Ks = ot.Normal(30.0, 7.5)
Ks.setDescription(["Ks (Strickler)"])

# Zv: Uniform(49, 51)
Zv = ot.Uniform(49.0, 51.0)
Zv.setDescription(["Zv (Downstream)"])

# Zm: Uniform(54, 56)
Zm = ot.Uniform(54.0, 56.0)
Zm.setDescription(["Zm (Upstream)"])

# B: Triangular(min=295, mode=300, max=305)
B = ot.Triangular(295.0, 300.0, 305.0)
B.setDescription(["B (Width)"])

# L: Triangular(min=4990, mode=5000, max=5010)
L = ot.Triangular(4990.0, 5000.0, 5010.0)
L.setDescription(["L (Length)"])

# Zb: Triangular(min=55, mode=55.5, max=56)
Zb = ot.Triangular(55.0, 55.5, 56.0)
Zb.setDescription(["Zb (Bank Alt)"])

# Hd: Uniform(2, 4) - "Low Dyke" scenario
Hd = ot.Uniform(2.0, 4.0)
Hd.setDescription(["Hd (Dyke Height)"])

# Define joint distribution (Independent inputs)
marginals = [Q, Ks, Zv, Zm, B, L, Zb, Hd]
problem = ot.JointDistribution(marginals)