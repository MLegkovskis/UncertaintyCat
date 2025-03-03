# utils/model_utils.py

"""
Model utilities for the UncertaintyCat application.
This module contains model validation, code safety, and OpenTURNS utilities.
"""

import ast
import re
import numpy as np
import openturns as ot
import ast
import inspect
import traceback
from groq import Groq
import os

# ================ Code Safety ================

class UnsafeNodeVisitor(ast.NodeVisitor):
    """
    AST node visitor that checks for potentially unsafe code patterns.
    
    This visitor checks for imports of potentially dangerous modules,
    use of dangerous functions like eval and exec, and calls to
    dangerous methods like system and popen.
    """
    def __init__(self):
        # List of disallowed module names
        self.disallowed_modules = {'os', 'sys', 'subprocess', 'shutil'}
        super().__init__()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in self.disallowed_modules:
                raise ValueError(
                    f"Importing module '{alias.name}' is not allowed for security reasons."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in self.disallowed_modules:
            raise ValueError(
                f"Importing from module '{node.module}' is not allowed for security reasons."
            )
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for dangerous built-in functions
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', '__import__']:
                raise ValueError(
                    f"Use of function '{node.func.id}' is not allowed for security reasons."
                )
        # Check for dangerous methods
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in self.disallowed_modules:
                    raise ValueError(
                        f"Use of module '{node.func.value.id}' is not allowed for security reasons."
                    )
            if node.func.attr in ['system', 'popen', 'remove', 'rmdir']:
                raise ValueError(
                    f"Use of method '{node.func.attr}' is not allowed for security reasons."
                )
        self.generic_visit(node)

def check_code_safety(code_str):
    """
    Check if the provided code contains potentially unsafe patterns.
    
    Parameters
    ----------
    code_str : str
        The code string to check
        
    Returns
    -------
    bool
        True if the code is safe, raises ValueError otherwise
        
    Raises
    ------
    ValueError
        If unsafe code is detected
    """
    try:
        tree = ast.parse(code_str)
        UnsafeNodeVisitor().visit(tree)
        return True
    except Exception as e:
        raise ValueError(f"Unsafe code detected: {e}")

# ================ OpenTURNS Utilities ================

def get_distribution_types():
    """
    Get a list of available distribution types in OpenTURNS.
    
    Returns
    -------
    list
        List of distribution type names
    """
    return [
        'Uniform', 'Normal', 'LogNormal', 'LogNormalMuSigma', 'Triangular',
        'Beta', 'Gamma', 'Gumbel', 'Exponential', 'Weibull', 'Rayleigh'
    ]

def get_distribution_names(problem):
    """
    Get the names of input variables from a problem dictionary.
    
    Parameters
    ----------
    problem : dict
        Problem dictionary containing input variable definitions
        
    Returns
    -------
    list
        List of variable names
    """
    if isinstance(problem, dict) and 'names' in problem:
        return problem['names']
    elif isinstance(problem, (ot.Distribution, ot.ComposedDistribution)):
        return [f"X{i+1}" for i in range(problem.getDimension())]
    else:
        raise ValueError("Problem must be a dictionary with 'names' key or an OpenTURNS distribution")

def get_ot_distribution(dist_info):
    """
    Create an OpenTURNS distribution from a distribution dictionary.
    
    Parameters
    ----------
    dist_info : dict
        Dictionary containing distribution type and parameters
        
    Returns
    -------
    ot.Distribution
        OpenTURNS distribution object
    """
    dist_type = dist_info.get('type', '')
    params = dist_info.get('params', [])
    
    if dist_type == 'Uniform':
        return ot.Uniform(params[0], params[1])
    elif dist_type == 'Normal':
        return ot.Normal(params[0], params[1])
    elif dist_type == 'LogNormal':
        return ot.LogNormal(params[0], params[1])
    elif dist_type == 'LogNormalMuSigma':
        return ot.LogNormalMuSigma(params[0], params[1])
    elif dist_type == 'Triangular':
        return ot.Triangular(params[0], params[1], params[2])
    elif dist_type == 'Beta':
        return ot.Beta(params[0], params[1], params[2], params[3])
    elif dist_type == 'Gamma':
        return ot.Gamma(params[0], params[1], params[2])
    elif dist_type == 'Gumbel':
        return ot.Gumbel(params[0], params[1])
    elif dist_type == 'Exponential':
        return ot.Exponential(params[0])
    elif dist_type == 'Weibull':
        return ot.Weibull(params[0], params[1])
    elif dist_type == 'Rayleigh':
        return ot.Rayleigh(params[0])
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def get_ot_model(model):
    """
    Convert a Python function to an OpenTURNS function.
    
    Parameters
    ----------
    model : function
        Python function to convert
        
    Returns
    -------
    ot.Function
        OpenTURNS function object
    """
    # Check if model is already an OpenTURNS function
    if isinstance(model, ot.Function):
        return model
    
    # Create a wrapper function for the model
    def wrapper(X):
        X_array = np.array(X)
        result = model(X_array)
        return [result] if np.isscalar(result) else result
    
    # Create an OpenTURNS function from the wrapper
    return ot.PythonFunction(model.__code__.co_argcount, 1, wrapper)

# ================ Model Validation ================

def get_human_friendly_error_explanation(code, error_message, language_model="gemma2-9b-it"):
    """
    Get a human-friendly explanation of an error message using an LLM.
    
    Parameters
    ----------
    code : str
        The code that caused the error
    error_message : str
        The error message
    language_model : str, optional
        The language model to use, by default "gemma2-9b-it"
        
    Returns
    -------
    str
        Human-friendly explanation of the error
    """
    # For simple errors, provide a direct explanation
    if "name 'np' is not defined" in error_message:
        return "You need to import numpy as np at the beginning of your code."
    
    if "name 'ot' is not defined" in error_message:
        return "You need to import openturns as ot at the beginning of your code."
    
    if "object has no attribute" in error_message:
        attr = re.search(r"'([^']*)'", error_message)
        if attr:
            return f"The object doesn't have the attribute '{attr.group(1)}'. Check the spelling or if you're using the correct object."
    
    # For more complex errors, use the language model
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        
        prompt = f"""
        I have the following Python code:
        
        ```python
        {code}
        ```
        
        And I'm getting this error:
        
        ```
        {error_message}
        ```
        
        Please provide a very concise explanation of what's wrong and how to fix it. Focus only on the specific issue causing the error. Keep your response under 100 words.
        """
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=language_model
        )
        
        explanation = chat_completion.choices[0].message.content
        
        # Clean up the explanation
        explanation = explanation.replace("```python", "").replace("```", "").strip()
        
        return explanation
    
    except Exception as e:
        # Fallback to a generic explanation if the API call fails
        return f"Error in your code: {error_message}"

def extract_model_and_problem(code):
    """
    Extract model function and problem dictionary from code.
    
    Parameters
    ----------
    code : str
        Code string containing model and problem definitions
        
    Returns
    -------
    tuple
        (model, problem) tuple
    """
    try:
        # Create a namespace to execute the code
        namespace = {}
        
        # Execute the code in the namespace
        exec(code, namespace)
        
        # Look for model and problem in the namespace
        model = namespace.get('model')
        if model is None:
            # Look for a function named 'function_of_interest'
            model = namespace.get('function_of_interest')
        
        problem = namespace.get('problem')
        
        # Validate model and problem
        if model is None:
            raise ValueError("No 'model' or 'function_of_interest' function found in the code")
        
        if problem is None:
            raise ValueError("No 'problem' dictionary found in the code")
        
        # Validate that problem is an OpenTURNS distribution
        if not isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
            raise ValueError("Problem must be an OpenTURNS distribution")
        
        return model, problem
    
    except Exception as e:
        # Get the traceback for more detailed error information
        tb = traceback.format_exc()
        raise ValueError(f"Error extracting model and problem: {str(e)}\n{tb}")

def validate_problem_structure(problem):
    """
    Validate that the problem is an OpenTURNS distribution.
    
    Parameters
    ----------
    problem : ot.Distribution
        Problem definition as an OpenTURNS distribution
        
    Returns
    -------
    bool
        True if valid, raises ValueError otherwise
    """
    # Check if it's an OpenTURNS distribution
    if isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        return True
    
    raise ValueError("Problem must be an OpenTURNS distribution")

def test_model(model, problem, code, language_model):
    """Tests the model with a small sample to catch potential errors early on."""
    try:
        # For OpenTURNS models and distributions
        if isinstance(model, ot.Function) and isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
            test_sample = problem.getSample(1)
            model(test_sample)
            return
        
        # For legacy Python function and dictionary-based problems
        from modules.monte_carlo import monte_carlo_simulation
        monte_carlo_simulation(model, problem, N=2)
    except Exception as e:
        error_explanation = get_human_friendly_error_explanation(code, str(e), language_model)
        raise ValueError(f"Model test failed: {error_explanation}")

# ================ Statistical Utilities ================

def sample_inputs(N, problem, seed=42):
    """
    Sample input variables based on problem definition.
    
    Parameters
    ----------
    N : int
        Number of samples to generate
    problem : ot.Distribution or ot.JointDistribution
        The OpenTURNS distribution that defines the problem
    seed : int, optional
        Random seed for reproducibility, by default 42
        
    Returns
    -------
    numpy.ndarray
        Array of sampled input values
    
    Raises
    ------
    ValueError
        If the problem is not an OpenTURNS distribution
    """
    if not isinstance(problem, (ot.Distribution, ot.JointDistribution)):
        raise ValueError("Only OpenTURNS distributions are supported")
    ot.RandomGenerator.SetSeed(seed)
    return np.array(problem.getSample(N))
