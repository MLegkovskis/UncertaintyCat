# modules/statistical_utils.py

import numpy as np
from scipy.stats import beta, lognorm, uniform, norm, triang, gumbel_r
from matplotlib.lines import Line2D


# Function to convert LogNormalMuSigma parameters to native form
def convert_lognormal_params(mu, sigma, gamma):
    sigma_log = np.sqrt(np.log(1 + (sigma / (mu - gamma)) ** 2))
    mu_log = np.log(mu - gamma) - sigma_log**2 / 2
    return mu_log, sigma_log


# Function to get bounds for distributions
def get_bounds(distribution):
    dist_type = distribution["type"]
    params = distribution["params"]
    if dist_type == "Uniform":
        return params
    elif dist_type == "Normal":
        mean, std = params
        return [mean - 3 * std, mean + 3 * std]
    elif dist_type == "LogNormalMuSigma":
        mu, sigma, gamma = params
        mu_log, sigma_log = convert_lognormal_params(mu, sigma, gamma)
        return [
            lognorm.ppf(0.001, s=sigma_log, scale=np.exp(mu_log)) + gamma,
            lognorm.ppf(0.999, s=sigma_log, scale=np.exp(mu_log)) + gamma,
        ]
    elif dist_type == "LogNormal":
        mu_log, sigma_log, gamma = params
        return [
            lognorm.ppf(0.001, s=sigma_log, scale=np.exp(mu_log)) + gamma,
            lognorm.ppf(0.999, s=sigma_log, scale=np.exp(mu_log)) + gamma,
        ]
    elif dist_type == "Beta":
        a, b, loc, scale = params
        return [loc, loc + scale]
    elif dist_type == "Gumbel":
        beta_param, gamma_param = params
        lower_bound = gumbel_r.ppf(0.001, loc=gamma_param, scale=beta_param)
        upper_bound = gumbel_r.ppf(0.999, loc=gamma_param, scale=beta_param)
        return [lower_bound, upper_bound]
    elif dist_type == "Triangular":
        a, m, b = params
        return [a, b]
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


# Function to extract bounds for SALib
def get_bounds_for_salib(problem):
    bounds = []
    for dist_info in problem["distributions"]:
        bounds.append(get_bounds(dist_info))
    problem_for_salib = {
        "num_vars": problem["num_vars"],
        "names": problem["names"],
        "bounds": bounds,
    }
    return problem_for_salib


# Function to sample input variables based on problem definition
def sample_inputs(N, problem, seed=42):
    np.random.seed(seed)
    samples = np.zeros((N, problem["num_vars"]))
    distributions = problem["distributions"]
    for i, dist_info in enumerate(distributions):
        dist_type = dist_info["type"]
        params = dist_info["params"]
        if dist_type == "Uniform":
            a, b = params
            samples[:, i] = uniform.rvs(loc=a, scale=b - a, size=N)
        elif dist_type == "Normal":
            mu, sigma = params
            samples[:, i] = norm.rvs(loc=mu, scale=sigma, size=N)
        elif dist_type == "Beta":
            a_beta, b_beta, loc, scale = params
            samples[:, i] = beta.rvs(
                a=a_beta, b=b_beta, loc=loc, scale=scale - loc, size=N
            )
        elif dist_type == "LogNormalMuSigma":
            mu, sigma, gamma = params
            mu_log, sigma_log = convert_lognormal_params(mu, sigma, gamma)
            samples[:, i] = (
                lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), size=N) + gamma
            )
        elif dist_type == "LogNormal":
            mu_log, sigma_log, gamma = params
            samples[:, i] = (
                lognorm.rvs(s=sigma_log, scale=np.exp(mu_log), size=N) + gamma
            )
        elif dist_type == "Triangular":
            a, m, b = params
            loc = a
            scale = b - a
            c = (m - a) / scale  # Adjust c for scipy's parameterization
            samples[:, i] = triang.rvs(c=c, loc=loc, scale=scale, size=N)
        elif dist_type == "Gumbel":
            beta_param, gamma_param = params
            samples[:, i] = gumbel_r.rvs(loc=gamma_param, scale=beta_param, size=N)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    return samples


def get_constant_value(distribution):
    dist_type = distribution["type"]
    params = distribution["params"]
    if dist_type == "Uniform":
        a, b = params
        return (a + b) / 2.0
    elif dist_type == "Normal":
        mu, sigma = params
        return mu
    elif dist_type == "LogNormalMuSigma":
        mu, sigma, gamma = params
        return mu
    elif dist_type == "LogNormal":
        mu_log, sigma_log, gamma = params
        return np.exp(mu_log) + gamma
    elif dist_type == "Beta":
        alpha, beta_value, a, b = params
        return a + (b - a) * alpha / (alpha + beta_value)
    elif dist_type == "Gumbel":
        beta_param, gamma_param = params
        return gamma_param
    elif dist_type == "Triangular":
        a, m, b = params
        return m
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


def clip_01(input_value):
    """
    Project the value into the [0, 1] interval

    Parameters
    ----------
    input_value : float
        The value

    Returns
    -------
    output_value : float, in [0, 1]
        The clipped value
    """
    output_value = max(0.0, min(1.0, input_value))
    return output_value


def plot_sobol_radial(
    names,
    sobol_indices,
    ax,
    sensitivity_threshold=0.01,
    max_marker_radius=70.0,
    tolerance=0.1,
):
    """
    Plot Sobol indices on a radial plot.

    The Sobol Indices Radial Plot is a polar plot where each input variable
    is placed at equal angular intervals around a circle.
    The elements of the plot are:

    - Variables: Each input variable is positioned at a specific angle on
      the circle, equally spaced from others.

    - Circles:
        - The outer circle (white fill) represents the total-order Sobol'
          index (ST) for each variable.
        - The inner circle (black fill) represents the first-order Sobol' index (S1).
        - The size of the circles is proportional to the magnitude of the respective
          Sobol' indices.

    - Lines:
        - Lines connecting variables represent second-order Sobol' indices (S2).
        - The thickness of the lines corresponds to the magnitude of the
          interaction between the two variables; thicker lines indicate
          stronger interactions.

    Parameters
    ----------
    names : list(str)
        The list of input variable names.
    sobol_indices : dict
        The sobol indices.
        sobol_indices["S1"] is the list of first-order Sobol' indices.
        sobol_indices["ST"] is the list of total order Sobol' indices.
        sobol_indices["S2"] is the matrix S of second order Sobol' indices.
        We expect that S[i, j] is the second order index of variables (i, j)
        for i in {0, ..., dimension - 1} and i in {i + 1, ..., dimension - 1}.
    ax : matplotlib.axis
        The figure
    sensitivity_threshold : float, in [0, 1]
        The threshold on the sensitivity indices.
        Any Sobol' sensitivity index lower than this threshold is not
        represented in the plot.
    max_marker_radius : float, > 0
        The marker size in points.
    tolerance : float, > 0.0
        The (absolute) tolerance on the Sobol' indices.
        If a Sobol' indice is not in [-tolerance, 1 + tolerance]
        then an exception is raise.

    Returns
    -------
    fig : Matplotlib.figure
        The plot

    References
    ----------
    - Jana Fruth, Olivier Roustant, and Sonja Kuhnt.
      Total interaction index: A variance-based sensitivity
      index for second-order interaction screening.
      Journal of Statistical Planning and Inference,
      147:212–223, 2014.
    """
    # Check indices dimensions
    dimension = len(names)
    number_of_entries = len(sobol_indices["S1"])
    if number_of_entries != dimension:
        raise ValueError(
            f"The number of variable names is {dimension} "
            f"but the number of first-order Sobol' indices is {number_of_entries}."
        )
    number_of_entries = len(sobol_indices["ST"])
    if number_of_entries != dimension:
        raise ValueError(
            f"The number of variable names is {dimension} "
            f"but the number of total order Sobol' indices is {number_of_entries}."
        )
    shape = sobol_indices["S2"].shape
    if shape != (dimension, dimension):
        raise ValueError(
            f"The number of variable names is {dimension} "
            f"but the shape of second order Sobol' indices is {shape}."
        )

    # Check indices values (avoid inconsistent use)
    S1 = sobol_indices["S1"]
    ST = sobol_indices["ST"]
    S2 = sobol_indices["S2"]
    for i in range(dimension):
        if S1[i] < -tolerance or S1[i] > 1.0 + tolerance:
            print(
                f"Warning: The first-order Sobol' index of variable #{i} is {S1[i]} "
                f"which is not in [0,1], up to the tolerance {tolerance}."
            )
        S1[i] = clip_01(S1[i])
        if ST[i] < -tolerance or ST[i] > 1.0 + tolerance:
            print(
                f"Warning: The total order Sobol' index of variable #{i} is {ST[i]} "
                f"which is not in [0,1], up to the tolerance {tolerance}."
            )
        ST[i] = clip_01(ST[i])
        for j in range(i + 1, dimension):
            if S2[i, j] < -tolerance or S2[i, j] > 1.0 + tolerance:
                print(
                    f"Warning: The second order Sobol' index of variables ({i}, {j}) is {S2[i, j]} "
                    f"which is not in [0,1], up to the tolerance {tolerance}."
                )
            S2[i, j] = clip_01(S2[i, j])

    # Get indices
    names = np.array(names)

    # Filter out insignificant indices
    significant = ST > sensitivity_threshold
    insignificant = ST <= sensitivity_threshold
    significant_names = names[significant]
    insignificant_names = names[insignificant]
    significant_dimension = len(significant_names)
    significant_angles = np.linspace(
        0, 2 * np.pi, significant_dimension, endpoint=False
    )
    ST = ST[significant]
    S1 = S1[significant]
    #

    # Prepare S2 matrix
    S2_matrix = np.zeros((len(significant_names), len(significant_names)))
    for i in range(len(significant_names)):
        for j in range(i + 1, len(significant_names)):
            idx_i = np.where(names == significant_names[i])[0][0]
            idx_j = np.where(names == significant_names[j])[0][0]
            S2_value = S2[idx_i, idx_j]
            if np.isnan(S2_value) or S2_value < sensitivity_threshold:
                S2_value = 0.0
            S2_matrix[i, j] = S2_value

    # Plotting
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks(significant_angles)
    ax.set_xticklabels(significant_names)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.5)

    # Plot ST and S1 using ax.scatter

    # First plot the ST circles (outer circles)
    # The "s" option of matplotlib.scatter sets the area:
    # the area of each circle is proportional to the ST index.
    for loc, st_val in zip(significant_angles, ST):
        s = st_val * max_marker_radius**2
        ax.scatter(loc, 1, s=s, c="white", edgecolors="black", zorder=2)

    # Then plot the S1 circles (inner circles)
    # The "s" option of matplotlib.scatter sets the area:
    # the area of each circle is proportional to the S1 index.
    for loc, s1_val in zip(significant_angles, S1):
        if s1_val > sensitivity_threshold:
            s = s1_val * max_marker_radius**2
            ax.scatter(loc, 1, s=s, c="black", edgecolors="black", zorder=3)

    # Plot S2 interactions
    # The "lw" option of matplotlib.plot sets the line width:
    # compute the width so that the area of the line is proportional
    # to the S2(i, j) index.
    for i in range(len(significant_names)):
        for j in range(i + 1, len(significant_names)):
            if S2_matrix[i, j] > 0:
                # Compute the distance between the two points (i, j)
                xi = np.cos(significant_angles[i])
                xj = np.cos(significant_angles[j])
                yi = np.sin(significant_angles[i])
                yj = np.sin(significant_angles[j])
                distance_ij = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                lw = S2_matrix[i, j] * max_marker_radius / distance_ij
                ax.plot(
                    [significant_angles[i], significant_angles[j]],
                    [1, 1],
                    c="darkgray",
                    lw=lw,
                    zorder=1,
                )

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="ST",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="S1",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=15,
        ),
        Line2D([0], [0], color="darkgray", lw=3, label="S2"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"Sobol Indices.\nInsignificant: {insignificant_names}")
    return
