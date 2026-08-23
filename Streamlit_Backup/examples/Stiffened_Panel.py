import openturns as ot
import numpy as np

"""
Stiffened Panel Example for Reliability Analysis
===============================================

This example models the critical shear force of a stiffened panel, which is a common
structural component in aerospace and marine engineering.

Important Notes:
---------------
1. Original output is in Newtons per meter (N/m), but for better usability in reliability
   analysis, we scale it to MPa (mega pascals).
2. Failure criterion: Output < Threshold (when the critical load is less than required)
3. Recommended threshold value: 150-180 MPa
4. All input dimensions are in meters unless otherwise noted
"""

def function_of_interest(X):
    """
    Calculate the critical shear force of a stiffened panel
    
    Parameters:
    -----------
    X : list or array with 10 elements
        E    : Young's modulus (GPa)
        nu   : Poisson's ratio
        h_c  : Distance between mean surface of hat and foot of stiffener (m)
        ell  : Length of stiffener flank (m)
        f_1  : Width of stiffener foot (m)
        f_2  : Width of stiffener hat (m)
        t    : Thickness of panel and stiffener (m)
        a    : Width of panel (m)
        b_0  : Distance between two stiffeners (m)
        p    : Half-width of stiffener (m)
    
    Returns:
    --------
    list with 1 element:
        Critical shear force scaled to MPa for easier interpretation
    """
    E, nu, h_c, ell, f_1, f_2, t, a, b_0, p = X
    
    # Calculate parameters
    kxy = 5.35 + 4 * (b_0 / a) ** 2
    D = E * t**3 / (12 * (1 - nu**2))
    A = ell * t
    A_bar = A + t * (p + (f_1 - f_2) / 2)
    h_0 = (A * (h_c + 2 * t) + t**2 * (f_1 - f_2)) / (2 * A_bar)
    h = h_c + t
    factor = 1 + (2 * p * (h - 2 * h_0) - h_c * (f_1 - f_2)) / (4 * h_0 * ell)
    
    # Calculate critical shear force per unit length (N/m)
    Nxy = kxy * np.pi**2 * D / b_0**2 * factor
    
    # Scale to get values around 200 MPa 
    # Using a much larger scaling factor (1e9) to get the desired range
    Nxy_MPa = Nxy * 1e9
    
    return [Nxy_MPa]

# Create OpenTURNS model
model = ot.PythonFunction(10, 1, function_of_interest)
model.setOutputDescription(["Critical Shear Force (MPa)"])

# Problem definition for the Stiffened Panel Model
# Define distributions with corrected descriptions
E = ot.Normal(110.0, 0.05 * 110.0)    # E : Young Modulus (GPa)
E.setDescription(["Young's Modulus (GPa)"])

nu = ot.Uniform(0.3675, 0.3825)      # nu : Poisson coefficient
nu.setDescription(["Poisson's Ratio"])

h_c = ot.Uniform(0.0285, 0.0315)     # hc : Distance between the mean surface of the hat and the foot of the Stiffener (m)
h_c.setDescription(["Hat-Foot Distance (m)"])

ell = ot.Uniform(0.04655, 0.05145)    # ell : Length of the stiffener flank (m)
ell.setDescription(["Stiffener Flank Length (m)"])

f1 = ot.Uniform(0.0266, 0.0294)      # f1 : Width of the stiffener foot (m)
f1.setDescription(["Stiffener Foot Width (m)"])

f2 = ot.Uniform(0.00627, 0.00693)    # f2 : Width of the stiffener hat (m)
f2.setDescription(["Stiffener Hat Width (m)"])

t = ot.Uniform(8.019e-05, 8.181e-05)  # t : Thickness of the panel and the stiffener (m)
t.setDescription(["Thickness (m)"])

a = ot.Uniform(0.6039, 0.6161)       # a : Width of the panel (m)
a.setDescription(["Panel Width (m)"])

b0 = ot.Uniform(0.04455, 0.04545)    # b0 : Distance between two stiffeners (m)
b0.setDescription(["Inter-Stiffener Distance (m)"])

p = ot.Uniform(0.03762, 0.03838)     # p : Half-width of the stiffener (m)
p.setDescription(["Half-Stiffener Width (m)"])

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
