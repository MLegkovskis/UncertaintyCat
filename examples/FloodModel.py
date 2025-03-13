import openturns as ot
import numpy as np

# Flood Model Function
def function_of_interest(X):
    Q, Ks, Zv, Zm, B, L, Zb, Hd = X

    # Calculate the slope alpha
    alpha = (Zm - Zv) / L if Zm >= Zv else 0

    # Calculate the water depth H
    if Ks > 0 and Q > 0 and alpha > 0:
        H = (Q / (Ks * B * np.sqrt(alpha))) ** 0.6
    else:
        H = 0

    # Calculate the flood altitude Zc
    Zc = H + Zv

    # Calculate the altitude of the dyke Zd
    Zd = Zb + Hd

    # Determine if there is a flood
    S = Zc - Zd

    # Calculate the cost of the dyke Cd
    Cd = 8 / 20 if Hd < 8 else Hd / 20

    # Calculate the cost of the flood Cs
    if S < 0:
        Cs = 1 - 0.8 * np.exp(-1000 / (S ** 4))
    else:
        Cs = 1

    # Total cost C
    C = Cd + Cs

    return [C]

model = ot.PythonFunction(8, 1, function_of_interest)

# Problem definition for the Flood Model
# Define distributions with corrected descriptions
Q = ot.Gumbel(1013, 558)       # Q
Q.setDescription(["Q"])

Ks = ot.Normal(30.0, 7.5)      # Ks
Ks.setDescription(["Ks"])

Zv = ot.Uniform(49, 51)        # Zv
Zv.setDescription(["Zv"])

Zm = ot.Uniform(54, 56)        # Zm
Zm.setDescription(["Zm"])

B = ot.Triangular(295, 300, 305)      # B
B.setDescription(["B"])

L = ot.Triangular(4990, 5000, 5010)    # L
L.setDescription(["L"])

Zb = ot.Triangular(55, 55.5, 56)      # Zb
Zb.setDescription(["Zb"])

Hd = ot.Uniform(2, 4)                   # Hd
Hd.setDescription(["Hd"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    Q,
    Ks,
    Zv,
    Zm,
    B,
    L,
    Zb,
    Hd
])
