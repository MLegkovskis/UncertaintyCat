import openturns as ot
import numpy as np

# Morris Function
def function_of_interest(X):
    # Ensure X is a numpy array
    X = np.asarray(X)

    # Ensure that X has 20 variables
    if X.shape[0] != 20:
        raise ValueError("Expected 20 input variables.")

    U = X  # Input variables

    # Compute transformed variables W
    W = 2 * (U - 0.5)

    # Adjust W at specific indices for nonlinearity
    indices = [2, 4, 6]  # Corresponds to variables 3, 5, 7 in MATLAB (1-based indexing)
    W[indices] = 2 * (1.1 * U[indices] / (U[indices] + 0.1) - 0.5)

    # Initialize the output
    Y = 0.0

    # Linear terms
    for i in range(20):
        if i <= 9:  # First 10 variables
            bi = 20
        else:
            bi = (-1) ** (i + 1)  # Adjust for zero-based indexing
        Y += bi * W[i]

    # Two-way interaction terms
    for i in range(19):
        for j in range(i + 1, 20):
            if i <= 5 and j <= 5:  # First 6 variables
                bij = -15
            else:
                bij = (-1) ** (i + j + 2)
            Y += bij * W[i] * W[j]

    # Three-way interaction terms
    for i in range(18):
        for j in range(i + 1, 19):
            for k in range(j + 1, 20):
                if i <= 4 and j <= 4 and k <= 4:  # First 5 variables
                    bijl = -10
                else:
                    bijl = 0
                Y += bijl * W[i] * W[j] * W[k]

    # Four-way interaction term
    bijls = 5
    Y += bijls * W[0] * W[1] * W[2] * W[3]  # First four variables

    return [Y]

model = ot.PythonFunction(20, 1, function_of_interest)

# Problem definition for the Morris Function
# Define distributions with corrected descriptions
names = ['X' + str(i + 1) for i in range(20)]
distributions = [ot.Uniform(0, 1) for _ in range(20)]
for name, dist in zip(names, distributions):
    dist.setDescription([name])

# Define joint distribution (independent)
problem = ot.JointDistribution(distributions)
