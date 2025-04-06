import openturns as ot
import numpy as np
import pandas as pd

# Define the input distributions with descriptions
E = ot.Beta(0.9, 3.1, 2.8e7, 4.8e7)
E.setDescription(["E"])

F = ot.LogNormal()
F.setParameter(ot.LogNormalMuSigma()([3.0e4, 9.0e3, 15.0e3]))
F.setDescription(["F"])

L = ot.Uniform(250.0, 260.0)
L.setDescription(["L"])

I = ot.Beta(2.5, 4.0, 310.0, 450.0)
I.setDescription(["I"])

# Define a correlation matrix for dependent variables (here L and I are correlated)
R = ot.CorrelationMatrix(4)
R[2, 3] = -0.2  # Spearman correlation between L and I

# Create the copula based on the correlation matrix
copula = ot.NormalCopula(R)

# Define the joint distribution with the copula
joint_distribution = ot.JointDistribution([E, F, L, I], copula)

# Number of samples (adjust this variable as needed)
num_samples = 1000

# Generate sample input rows from the joint distribution
sample_inputs = joint_distribution.getSample(num_samples)

# Convert the OpenTURNS sample to a NumPy array for easier manipulation
inputs_np = np.array(sample_inputs)

# Create a pandas DataFrame with appropriate column names
columns = ["E", "F", "L", "I"]
df = pd.DataFrame(inputs_np, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv("sample_inputs.csv", index=False)

print(f"CSV file 'sample_inputs.csv' created with {num_samples} sample rows (only input columns).")
