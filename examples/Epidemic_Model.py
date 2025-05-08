import openturns as ot
import numpy as np
from scipy.integrate import solve_ivp

# SIR Model Function
def function_of_interest(X):
    S0, I0, R0, beta, gamma, N, t_max = X

    # Define the SIR model differential equations
    def sir_model(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    # Initial conditions vector
    y0 = [S0, I0, R0]

    # Time points
    t_span = [0, t_max]
    t_eval = np.linspace(0, t_max, 100)

    # Solve the differential equations
    sol = solve_ivp(sir_model, t_span, y0, t_eval=t_eval)

    # Total number of infected individuals over time (max I)
    total_infected = max(sol.y[1])

    return [total_infected]

model = ot.PythonFunction(7, 1, function_of_interest)

# Problem definition for the Epidemic Model
# Define distributions with corrected descriptions
S0 = ot.Uniform(990, 1000)       # S0: Initial susceptible population
S0.setDescription(["S0"])

I0 = ot.Uniform(1, 10)           # I0: Initial infected population
I0.setDescription(["I0"])

R0 = ot.Uniform(1, 5)            # R0: Initial recovered population
R0.setDescription(["R0"])

beta = ot.Uniform(0.1, 0.5)      # beta: Transmission rate
beta.setDescription(["beta"])

gamma = ot.Uniform(0.05, 0.2)    # gamma: Recovery rate
gamma.setDescription(["gamma"])

N = ot.Uniform(1000, 1010)       # N: Total population
N.setDescription(["N"])

t_max = ot.Uniform(30, 60)       # t_max: Simulation time (days)
t_max.setDescription(["t_max"])

# Define joint distribution (independent)
problem = ot.JointDistribution([
    S0,
    I0,
    R0,
    beta,
    gamma,
    N,
    t_max
])
