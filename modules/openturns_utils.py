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

class SuperPoint():
    def __init__(self, point):
        self.point = point
    
    def toPython(self, variableName="", indentation=""):
        dimension = self.point.getDimension()
        code = f"{indentation}"
        if variableName != "":
            code += f"{variableName} = "
        code += "Point(["
        for i in range(dimension):
            code += f"{self.point[i]}"
            if i < dimension - 1:
                code += ", "
        code += "])"
        return code
    

#
class SuperSample():
    def __init__(self, sample):
        self.sample = sample
    
    def toPython(self, variableName="", indentation=""):
        code = f"{indentation}"
        if variableName != "":
            code += f"{variableName} = "
        size = self.sample.getSize()
        dimension = self.sample.getDimension()
        code += "Sample(["
        for i in range(size):
            code += "["
            for j in range(dimension):
                code += f"{self.sample[i, j]}"
                if j < dimension - 1:
                    code += ", "
            code += "]"
            if i < size - 1:
                code += ", "
        code += "])"
        return code
    

#
class SuperDistribution():
    def __init__(self, distribution):
        self.distribution = distribution
    
    def toPython(self, variableName="", indentation=""):
        dimension = self.distribution.getDimension()
        if dimension == 1:
            code = self._toPython1d(variableName, indentation)
        else:
            code = self._toPythonNd(variableName, indentation)
        return code

    def _toPython1d(self, variableName="", indentation=""):
        code = f"{indentation}"
        if variableName != "":
            code += f"{variableName} = "
        className = self.distribution.getClassName()
        if className == "Distribution":
            # Overwrite class name
            className = self.distribution.getImplementation().getClassName()
        parameter = self.distribution.getParameter()
        code += f"{className}("
        # Do not use SuperPoint.toPython() directly, because it involves
        # an unwanted conversion to Point and unwanted square brackets
        paramDimension = parameter.getDimension()
        for i in range(paramDimension):
            code += f"{parameter[i]}"
            if i < paramDimension - 1:
                code += ", "
        code += ")"
        return code

    def _toPythonNd(self, variableName="", indentation=""):
        copula = self.distribution.getCopula()
        copulaName = copula.getImplementation().getClassName()
        if copulaName != "IndependentCopula":
            raise ValueError(f"Non independent copula {copulaName} is not implemented yet")
        className = self.distribution.getClassName()
        if className == "Distribution":
            # Overwrite class name
            className = self.distribution.getImplementation().getClassName()
        if className == "JointDistribution":
            code = f"{indentation}marginals = []\n"
            dimension = self.distribution.getDimension()
            for i in range(dimension):
                marginal = self.distribution.getMarginal(i)
                superMarginal = SuperDistribution(marginal)
                code += f"{indentation}marginals.append({superMarginal.toPython()})\n"
            if variableName != "":
                code += f"{indentation}{variableName} = JointDistribution(marginals)\n"
            else:
                code += f"{indentation}JointDistribution(marginals)\n"
        else:
            raise ValueError(f"Distribution {className} is not implemented yet")
        return code



#
class SuperChaosResult():
    def __init__(self, functionalChaosResult):
        self.functionalChaosResult = functionalChaosResult
    
    def toPython(self):
        coefficients = self.functionalChaosResult.getCoefficients()
        code = ""
        code = "from openturns import *\n"
        code += "def function_of_interest(X):\n"
        code += "    # Define the distribution\n"
        # Serialize the input distribution
        distribution = self.functionalChaosResult.getDistribution()
        superDistribution = SuperDistribution(distribution)
        distributionCode = superDistribution.toPython("distribution", "    ")
        code += f"{distributionCode}"
        # Get the basis
        indices = self.functionalChaosResult.getIndices()
        code += f"    # Set the indices\n"
        code += f"    indices = {indices}\n"
        # Serialize the coefficients
        coefficients = self.functionalChaosResult.getCoefficients()
        coeffCode = SuperSample(coefficients).toPython()
        code += "    # Define the coefficients\n"
        code += f"    coefficients = {coeffCode}\n"
        metamodelCode = """    # Set the basis
    inputDimension = distribution.getDimension()
    polynomials = PolynomialFamilyCollection(inputDimension)
    for i in range(inputDimension):
        marginalPolynomial = StandardDistributionPolynomialFactory(marginals[i])
        polynomials[i] = marginalPolynomial
    enumerate = LinearEnumerateFunction(inputDimension)
    basis = OrthogonalProductPolynomialFactory(polynomials, enumerate)
    # Set the function collection
    function_collection = []
    numberOfIndices = len(indices)
    for i in range(numberOfIndices):
        function_collection.append(basis.build(indices[i]))
    # Set the composed metamodel, set the transformation, create the metamodel
    composedMetaModel = DualLinearCombinationFunction(function_collection, coefficients)
    measure = basis.getMeasure()
    transformation = DistributionTransformation(distribution, measure)
    metaModel = ComposedFunction(composedMetaModel, transformation)
    # Evaluate the metamodel
    Y = metaModel(X)
"""
        code += metamodelCode
        code += "    return [Y[0]]\n"
        return code
