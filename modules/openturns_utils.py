import openturns as ot
import numpy as np

def get_ot_distribution(problem):
    """Get OpenTURNS distribution from problem definition."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")
    return problem

def get_distribution_names(problem):
    """Get names of variables from problem definition."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")
    return problem.getDescription()

def get_dimension(problem):
    """Get dimension of the problem."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")
    return problem.getDimension()

def get_distribution_types(problem):
    """Get types of distributions for each variable."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")
    dimension = problem.getDimension()
    types = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        types.append(marginal.getName())
    return types

def get_distribution_parameters(problem):
    """Get parameters of distributions for each variable."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")
    dimension = problem.getDimension()
    parameters = []
    for i in range(dimension):
        marginal = problem.getMarginal(i)
        parameters.append(list(marginal.getParameter()))
    return parameters

def get_ot_model(model, problem):
    """Get OpenTURNS model from Python function."""
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")
    dimension = problem.getDimension()
    return ot.PythonFunction(dimension, 1, model)

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
        
        # Generate code for marginals first
        code = f"{indentation}marginals = []\n"
        dimension = self.distribution.getDimension()
        for i in range(dimension):
            marginal = self.distribution.getMarginal(i)
            superMarginal = SuperDistribution(marginal)
            code += f"{indentation}marginals.append({superMarginal.toPython()})\n"
        
        # Handle different copula types
        if copulaName == "NormalCopula":
            correlation = copula.getCorrelation()
            code += f"{indentation}R = CorrelationMatrix({dimension})\n"
            for i in range(dimension):
                for j in range(i+1, dimension):
                    if abs(correlation[i,j]) > 1e-10:  # Only add non-zero correlations
                        code += f"{indentation}R[{i}, {j}] = {correlation[i,j]}\n"
            code += f"{indentation}copula = NormalCopula(R)\n"
            if variableName != "":
                code += f"{indentation}{variableName} = JointDistribution(marginals, copula)\n"
            else:
                code += f"{indentation}JointDistribution(marginals, copula)\n"
        elif copulaName == "IndependentCopula":
            if variableName != "":
                code += f"{indentation}{variableName} = JointDistribution(marginals)\n"
            else:
                code += f"{indentation}JointDistribution(marginals)\n"
        else:
            raise ValueError(f"Copula {copulaName} is not implemented yet")
        
        return code

class SuperChaosResult():
    def __init__(self, functionalChaosResult):
        self.functionalChaosResult = functionalChaosResult
    
    def toPython(self):
        coefficients = self.functionalChaosResult.getCoefficients()
        code = "from openturns import *\n\n"
        code += "# Define the distribution\n"
        # Serialize the input distribution
        distribution = self.functionalChaosResult.getDistribution()
        superDistribution = SuperDistribution(distribution)
        distributionCode = superDistribution.toPython("distribution", "")
        code += f"{distributionCode}"
        # Get the basis
        indices = self.functionalChaosResult.getIndices()
        code += f"# Set the indices\n"
        code += f"indices = {indices}\n"
        # Serialize the coefficients
        coefficients = self.functionalChaosResult.getCoefficients()
        coeffCode = SuperSample(coefficients).toPython()
        code += "# Define the coefficients\n"
        code += f"coefficients = {coeffCode}\n"
        code += """# Set the basis
inputDimension = distribution.getDimension()
polynomials = PolynomialFamilyCollection(inputDimension)
for i in range(inputDimension):
    marginal = distribution.getMarginal(i)
    marginalPolynomial = StandardDistributionPolynomialFactory(marginal)
    polynomials[i] = marginalPolynomial
enumerate = LinearEnumerateFunction(inputDimension)
basis = OrthogonalProductPolynomialFactory(polynomials, enumerate)
# Set the function collection
functionFactory = UniVariateFunctionFactory(basis)
functions = []
indices_size = indices.getSize()
for i in range(indices_size):
    functions.append(functionFactory.build(indices[i]))
# Set the model
metamodel = LinearCombinationFunction(functions, coefficients)

def function_of_interest(X):
    # Evaluate the metamodel
    Y = metamodel(X)
    return [Y[0]]
"""
        return code
