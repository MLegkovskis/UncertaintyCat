import openturns as ot

def get_ot_distribution(problem):
    """
    Return the OpenTURNS model

    Parameters
    ----------
    problem : the problem
        The probabilistic problem

    Returns
    -------
    distribution : ot.Distribution
        The probabilistic model
    """
    marginals = []
    for dist_info in problem["distributions"]:
        dist_type = dist_info["type"]
        params = dist_info["params"]
        if dist_type == "Uniform":
            a, b = params
            marginals.append(ot.Uniform(a, b))
        elif dist_type == "Normal":
            mu, sigma = params
            marginals.append(ot.Normal(mu, sigma))
        elif dist_type == "Gumbel":
            beta_param, gamma_param = params
            marginals.append(ot.Gumbel(beta_param, gamma_param))
        elif dist_type == "Triangular":
            a, m, b = params
            marginals.append(ot.Triangular(a, m, b))
        elif dist_type == "Beta":
            alpha, beta_value, a, b = params
            marginals.append(ot.Beta(alpha, beta_value, a, b))
        elif dist_type == "LogNormal":
            mu, sigma, gamma = params
            marginals.append(ot.LogNormal(mu, sigma, gamma))
        elif dist_type == "LogNormalMuSigma":
            mu, sigma, gamma = params
            marginals.append(
                ot.ParametrizedDistribution(ot.LogNormalMuSigma(mu, sigma, gamma))
            )
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

    distribution = ot.ComposedDistribution(marginals)
    distribution.setDescription(problem["names"])
    return distribution


def get_ot_model(model, problem):
    """
    Return the OpenTURNS model

    Parameters
    ----------
    model : the model
        The physical model
    problem : the problem
        The probabilistic problem

    Returns
    -------
    physical_model : ot.Function
        The physical model g
    """
    ot_model = ot.PythonFunction(problem["num_vars"], 1, model)
    return ot_model


def ot_point_to_list(point):
    """
    Return a list corresponding to the OpenTURNS point

    Parameters
    ----------
    point : ot.Point
        The point

    Returns
    -------
    values : list(float)
        The point
    """
    dimension = point.getDimension()
    return [point[i] for i in range(dimension)]
