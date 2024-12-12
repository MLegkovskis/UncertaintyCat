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


# Problem definition for the stiffened panel
problem = {
    "num_vars": 10,
    "names": ["E", "nu", "h_c", "ell", "f_1", "f_2", "t", "a", "b_0", "p"],
    "distributions": [
        {"type": "Normal", "params": [110.0, 0.05 * 110.0]},  # E : Young Modulus
        {"type": "Uniform", "params": [0.3675, 0.3825]},  # nu : Poisson coefficient
        {
            "type": "Uniform",
            "params": [0.0285, 0.0315],
        },  # hc : Distance between the mean surface of the hat and the foot of the Stiffener (m)
        {
            "type": "Uniform",
            "params": [0.04655, 0.05145],
        },  # ell : Length of the stiffener flank (m)
        {
            "type": "Uniform",
            "params": [0.0266, 0.0294],
        },  # f1 : Width of the stiffener foot (m)
        {
            "type": "Uniform",
            "params": [0.00627, 0.00693],
        },  # f2 : Width of the stiffener hat (m)
        {
            "type": "Uniform",
            "params": [8.019e-05, 8.181e-05],
        },  # t : Thickness of the panel and the stiffener (m)
        {"type": "Uniform", "params": [0.6039, 0.6161]},  # a : Width of the panel (m)
        {
            "type": "Uniform",
            "params": [0.04455, 0.04545],
        },  # b0 : Distance between two stiffeners (m)
        {
            "type": "Uniform",
            "params": [0.03762, 0.03838],
        },  # p : Half-width of the stiffener (m)
    ],
}

model = function_of_interest
