import numpy as np
from scipy.integrate import solve_ivp

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
    # Total number of infected individuals over time
    total_infected = max(sol.y[1])
    return [total_infected]

# Problem definition
problem = {
    'num_vars': 7,
    'names': ['S0', 'I0', 'R0', 'beta', 'gamma', 'N', 't_max'],
    'distributions': [
        {'type': 'Uniform', 'params': [990, 1000]},       # S0: Initial susceptible population
        {'type': 'Uniform', 'params': [1, 10]},           # I0: Initial infected population
        {'type': 'Uniform', 'params': [1, 5]},            # R0: Initial recovered population
        {'type': 'Uniform', 'params': [0.1, 0.5]},        # beta: Transmission rate
        {'type': 'Uniform', 'params': [0.05, 0.2]},       # gamma: Recovery rate
        {'type': 'Uniform', 'params': [1000, 1010]},      # N: Total population
        {'type': 'Uniform', 'params': [30, 60]}           # t_max: Simulation time (days)
    ]
}

model = function_of_interest
