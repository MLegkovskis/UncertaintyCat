import numpy as np

def function_of_interest(X):
    Q, Ks, Zv, Zm, B, L, Zb, Hd = X

    # Calculate the slope alpha
    alpha = (Zm - Zv) / L if Zm >= Zv else 0

    # Calculate the water depth H
    H = (Q / (Ks * B * np.sqrt(alpha)))**0.6 if Ks > 0 and Q > 0 else 0

    # Calculate the flood altitude Zc
    Zc = H + Zv

    # Calculate the altitude of the dyke Zd
    Zd = Zb + Hd

    # Determine if there is a flood
    S = Zc - Zd

    # Calculate the cost of the dyke Cd
    Cd = 8 / 20 if Hd < 8 else Hd / 20

    # Calculate the cost of the flood Cs
    Cs = 1 - 0.8 * np.exp(-1000 / S**4) if S < 0 else 1

    # Total cost C
    C = Cd + Cs

    return [C]

# Problem definition for the flood model
problem = {
    'num_vars': 8,
    'names': ['Q', 'Ks', 'Zv', 'Zm', 'B', 'L', 'Zb', 'Hd'],
    'distributions': [
        {'type': 'Gumbel', 'params': [1013, 558]},       # Q
        {'type': 'Normal', 'params': [30.0, 7.5]},      # Ks
        {'type': 'Uniform', 'params': [49, 51]},        # Zv
        {'type': 'Uniform', 'params': [54, 56]},        # Zm
        {'type': 'Triangular', 'params': [295, 300, 305]},  # B
        {'type': 'Triangular', 'params': [4990, 5000, 5010]},# L
        {'type': 'Triangular', 'params': [55, 55.5, 56]},    # Zb
        {'type': 'Uniform', 'params': [2, 4]}                # Hd
    ]
}

model = function_of_interest
