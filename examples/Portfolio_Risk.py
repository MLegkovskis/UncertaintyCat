import openturns as ot
import numpy as np

# Portfolio Risk Function
def function_of_interest(X):
    S0, mu, sigma, T = X  # Initial portfolio value, expected return, volatility, time horizon
    N = 10                # Number of assets
    rho = 0.5             # Correlation coefficient between assets (assumed constant)

    np.random.seed(42)    # For reproducibility
    dt = T / 252          # Time step (assuming 252 trading days in a year)

    # Simulate correlated asset returns using Cholesky decomposition
    mean_returns = np.full(N, mu * dt)
    # Construct the covariance matrix with correlations
    cov_matrix = sigma**2 * dt * (rho * np.ones((N, N)) + (1 - rho) * np.eye(N))
    L = np.linalg.cholesky(cov_matrix)

    # Generate random returns
    Z = np.random.normal(size=(N,))
    correlated_returns = mean_returns + L @ Z

    # Calculate portfolio value
    portfolio_value = S0 * np.exp(np.sum(correlated_returns))
    return [portfolio_value]

model = ot.PythonFunction(4, 1, function_of_interest)

# Problem definition for the Portfolio Risk Model
# Define distributions with corrected descriptions
S0 = ot.Uniform(800000, 1200000)  # S0: $800,000 - $1,200,000
S0.setDescription(["S0"])

# Create Normal distributions with mean and standard deviation directly
mu = ot.Normal(0.1, 0.02)  # mu: Expected annual return (10%)
mu.setDescription(["mu"])

sigma = ot.Normal(0.2, 0.05)  # sigma: Volatility (20%)
sigma.setDescription(["sigma"])

T = ot.Uniform(0.5, 1.5)          # T: 0.5 - 1.5 years
T.setDescription(["T"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    S0,
    mu,
    sigma,
    T
])
