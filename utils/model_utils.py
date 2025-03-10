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

# ================ OpenTURNS Utilities ================

def get_distribution_names(problem):
    """
    Get the names of input variables from a problem distribution.
    
    Parameters
    ----------
    problem : ot.Distribution or ot.ComposedDistribution
        OpenTURNS distribution object
        
    Returns
    -------
    list
        List of variable names
    """
    if isinstance(problem, (ot.Distribution, ot.ComposedDistribution)):
        names = []
        for i in range(problem.getDimension()):
            marginal = problem.getMarginal(i)
            name = marginal.getDescription()[0]
            names.append(name if name != "" else f"X{i+1}")
        return names
    else:
        raise ValueError("Problem must be an OpenTURNS distribution")

def get_ot_model(model):
    """
    Convert a Python function to an OpenTURNS function.
    
    Parameters
    ----------
    model : function or ot.Function
        Python function to convert or OpenTURNS function
        
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

def extract_model_and_problem(code):
    """
    Extract model function and problem distribution from code.
    
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
        raise ValueError(f"Model test failed: {str(e)}")

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
