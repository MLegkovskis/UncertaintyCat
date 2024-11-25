# modules/statistical_utils.py

import numpy as np
from scipy.stats import beta, lognorm, uniform, norm, triang, gumbel_r

# Function to convert LogNormalMuSigma parameters to native form
def convert_lognormal_params(mu, sigma, gamma):
    sigma_log = np.sqrt(np.log(1 + (sigma / (mu - gamma))**2))
    mu_log = np.log(mu - gamma) - sigma_log**2 / 2
    return mu_log, sigma_log

# Function to get bounds for distributions
def get_bounds(distribution):
    dist_type = distribution['type']
    params = distribution['params']
    if dist_type == 'Uniform':
        return params
    elif dist_type == 'Normal':
        mean, std = params
        return [mean - 3*std, mean + 3*std]
    elif dist_type == 'LogNormalMuSigma':
        mu, sigma, gamma = params
        mu_log, sigma_log = convert_lognormal_params(mu, sigma, gamma)
        return [lognorm.ppf(0.001, s=sigma_log, scale=np.exp(mu_log)) + gamma,
                lognorm.ppf(0.999, s=sigma_log, scale=np.exp(mu_log)) + gamma]
    elif dist_type == 'LogNormal':
        mu_log, sigma_log, gamma = params
        return [lognorm.ppf(0.001, s=sigma_log, scale=np.exp(mu_log)) + gamma,
                lognorm.ppf(0.999, s=sigma_log, scale=np.exp(mu_log)) + gamma]
    elif dist_type == 'Beta':
        a, b, loc, scale = params
        return [loc, loc + scale]
    elif dist_type == 'Gumbel':
        beta_param, gamma_param = params
        lower_bound = gumbel_r.ppf(0.001, loc=gamma_param, scale=beta_param)
        upper_bound = gumbel_r.ppf(0.999, loc=gamma_param, scale=beta_param)
        return [lower_bound, upper_bound]
    elif dist_type == 'Triangular':
        a, m, b = params
        return [a, b]
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

# Function to extract bounds for SALib
def get_bounds_for_salib(problem):
    bounds = []
    for dist_info in problem['distributions']:
        bounds.append(get_bounds(dist_info))
    problem_for_salib = {
        'num_vars': problem['num_vars'],
        'names': problem['names'],
        'bounds': bounds
    }
    return problem_for_salib

# Function to sample input variables based on problem definition
def sample_inputs(N, problem, seed=42):
    np.random.seed(seed)
    samples = np.zeros((N, problem['num_vars']))
    distributions = problem['distributions']
    for i, dist_info in enumerate(distributions):
        dist_type = dist_info['type']
        params = dist_info['params']
        if dist_type == 'Uniform':
            a, b = params
            samples[:, i] = uniform.rvs(loc=a, scale=b - a, size=N)
        elif dist_type == 'Normal':
            mu, sigma = params
            samples[:, i] = norm.rvs(loc=mu, scale=sigma, size=N)
        elif dist_type == 'Beta':
            a_beta, b_beta, loc, scale = params
            samples[:, i] = beta.rvs(a=a_beta, b=b_beta, loc=loc, scale=scale - loc, size=N)
        elif dist_type == 'LogNormalMuSigma':
            mu, sigma, gamma = params
            mu_log, sigma_log = convert_lognormal_params(mu, sigma, gamma)
            samples[:, i] = lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), size=N) + gamma
        elif dist_type == 'LogNormal':
            mu_log, sigma_log, gamma = params
            samples[:, i] = lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), size=N) + gamma
        elif dist_type == 'Triangular':
            a, m, b = params
            loc = a
            scale = b - a
            c = (m - a) / scale  # Adjust c for scipy's parameterization
            samples[:, i] = triang.rvs(c=c, loc=loc, scale=scale, size=N)
        elif dist_type == 'Gumbel':
            beta_param, gamma_param = params
            samples[:, i] = gumbel_r.rvs(loc=gamma_param, scale=beta_param, size=N)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return samples

def get_constant_value(distribution):
    dist_type = distribution['type']
    params = distribution['params']
    if dist_type == 'Uniform':
        a, b = params
        return (a + b) / 2.0
    elif dist_type == 'Normal':
        mu, sigma = params
        return mu
    elif dist_type == 'LogNormalMuSigma':
        mu, sigma, gamma = params
        return mu
    elif dist_type == 'LogNormal':
        mu_log, sigma_log, gamma = params
        return np.exp(mu_log) + gamma
    elif dist_type == 'Beta':
        alpha, beta_value, a, b = params
        return a + (b - a) * alpha / (alpha + beta_value)
    elif dist_type == 'Gumbel':
        beta_param, gamma_param = params
        return gamma_param
    elif dist_type == 'Triangular':
        a, m, b = params
        return m
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
