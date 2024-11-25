import numpy as np

def function_of_interest(X):
    x1, x2, x3 = X
    Y = np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)
    return [Y]

# Problem definition for the new model
problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'distributions': [
        {'type': 'Uniform', 'params': [-np.pi, np.pi]},  # x1
        {'type': 'Uniform', 'params': [-np.pi, np.pi]},  # x2
        {'type': 'Uniform', 'params': [-np.pi, np.pi]}   # x3
    ]
}

model = function_of_interest
