def function_of_interest(X):
    E, F, L, I = X
    Y = F * L**3 / (3 * E * I)
    return [Y]

# Problem definition for the cantilever beam model
problem = {
    'num_vars': 4,
    'names': ['E', 'F', 'L', 'I'],
    'distributions': [
        {'type': 'Beta', 'params': [0.9, 3.1, 2.8e7, 4.8e7]},  # E
        {'type': 'LogNormalMuSigma', 'params': [3.0e4, 9.0e3, 15.0e3]},  # F
        {'type': 'Uniform', 'params': [250., 260.]},  # L
        {'type': 'Beta', 'params': [2.5, 4, 310., 450.]}  # I
    ]
}

model = function_of_interest