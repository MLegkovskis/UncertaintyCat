import openturns as ot
import numpy as np

# The critical shear force of a stiffened panel
def function_of_interest(X):
    E, nu, h_c, ell, f_1, f_2, t, a, b_0, p = X
    kxy = 5.35 + 4 * (b_0 / a) ** 2
    D = E * t**3 / (12 * (1 - nu**2))
    A = ell * t
    A_bar = A + t * (p + (f_1 - f_2) / 2)
    h_0 = (A * (h_c + 2 * t) + t**2 * (f_1 - f_2)) / (2 * A_bar)
    h = h_c + t
    factor = 1 + (2 * p * (h - 2 * h_0) - h_c * (f_1 - f_2)) / (4 * h_0 * ell)
    Nxy = kxy * np.pi**2 * D / b_0**2 * factor
    return [Nxy]

model = ot.PythonFunction(10, 1, function_of_interest)

# Problem definition for the Stiffened Panel Model
# Define distributions with corrected descriptions
E = ot.Normal(110.0, 0.05 * 110.0)    # E : Young Modulus
E.setDescription(["E"])

nu = ot.Uniform(0.3675, 0.3825)      # nu : Poisson coefficient
nu.setDescription(["nu"])

h_c = ot.Uniform(0.0285, 0.0315)     # hc : Distance between the mean surface of the hat and the foot of the Stiffener (m)
h_c.setDescription(["h_c"])

ell = ot.Uniform(0.04655, 0.05145)    # ell : Length of the stiffener flank (m)
ell.setDescription(["ell"])

f1 = ot.Uniform(0.0266, 0.0294)      # f1 : Width of the stiffener foot (m)
f1.setDescription(["f_1"])

f2 = ot.Uniform(0.00627, 0.00693)    # f2 : Width of the stiffener hat (m)
f2.setDescription(["f_2"])

t = ot.Uniform(8.019e-05, 8.181e-05)  # t : Thickness of the panel and the stiffener (m)
t.setDescription(["t"])

a = ot.Uniform(0.6039, 0.6161)       # a : Width of the panel (m)
a.setDescription(["a"])

b0 = ot.Uniform(0.04455, 0.04545)    # b0 : Distance between two stiffeners (m)
b0.setDescription(["b_0"])

p = ot.Uniform(0.03762, 0.03838)     # p : Half-width of the stiffener (m)
p.setDescription(["p"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    E,
    nu,
    h_c,
    ell,
    f1,
    f2,
    t,
    a,
    b0,
    p
])
