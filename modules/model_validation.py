from groq import Groq
import os
import streamlit as st
import numpy as np
import openturns as ot

# Validation logic
expected_params = {
    'Uniform': 2,
    'Normal': 2,
    'LogNormalMuSigma': 3,
    'LogNormal': 3,
    'Beta': 4,
    'Gumbel': 2,
    'Triangular': 3
}

def validate_problem_structure(problem):
    """Validates the structure of the problem definition."""
    # Handle OpenTURNS Distribution
    if isinstance(problem, (ot.Distribution, ot.JointDistribution, ot.ComposedDistribution)):
        # For OpenTURNS distributions, we just need to verify it's properly configured
        try:
            dimension = problem.getDimension()
            # Test that we can get a sample
            test_sample = problem.getSample(1)
            # Test that we can get descriptions
            descriptions = problem.getDescription()
            if len(descriptions) != dimension:
                raise ValueError("Distribution description length does not match dimension")
            return
        except Exception as e:
            raise ValueError(f"Invalid OpenTURNS distribution: {str(e)}")

    # Legacy validation for dictionary-based problems
    required_keys = ['num_vars', 'names', 'distributions']
    for key in required_keys:
        if key not in problem:
            raise ValueError(f"Missing required key '{key}' in problem definition.")

    if not isinstance(problem['num_vars'], int):
        raise ValueError("problem['num_vars'] must be an integer.")

    if not isinstance(problem['names'], list):
        raise ValueError("problem['names'] must be a list.")

    if len(problem['names']) != problem['num_vars']:
        raise ValueError("Length of problem['names'] must match problem['num_vars'].")

    if not isinstance(problem['distributions'], list):
        raise ValueError("problem['distributions'] must be a list.")

    if len(problem['distributions']) != problem['num_vars']:
        raise ValueError("Length of problem['distributions'] must match problem['num_vars'].")

    for i, dist in enumerate(problem['distributions']):
        if not isinstance(dist, dict):
            raise ValueError(f"Distribution {i} must be a dictionary.")
        
        if 'type' not in dist:
            raise ValueError(f"Distribution {i} missing 'type' key.")
            
        if 'params' not in dist:
            raise ValueError(f"Distribution {i} missing 'params' key.")
            
        if not isinstance(dist['params'], list):
            raise ValueError(f"Parameters for distribution {i} must be a list.")
            
        dist_type = dist['type']
        if dist_type not in expected_params:
            raise ValueError(f"Unknown distribution type '{dist_type}' in distribution {i}.")
            
        expected_param_count = expected_params[dist_type]
        if len(dist['params']) != expected_param_count:
            raise ValueError(
                f"Distribution {i} ({dist_type}) expects {expected_param_count} "
                f"parameters but got {len(dist['params'])}."
            )

def get_human_friendly_error_explanation(code_snippet, error_message, model_name):
    """Generates a human-friendly explanation of the error using the Groq API."""
    examples = """
    Examples of helpful explanations and corrections:

    - "It seems you've specified an unsupported or improperly configured distribution. For example, if you're using a `Beta` distribution, you must provide four parameters `[alpha, beta, lower_bound, upper_bound]`. Using something like `{{'type': 'Beta', 'params': [2.5, 4]}}` is invalid because it's missing the lower and upper bounds. A correct Beta distribution example would be:
    ```python
    {{'type': 'Beta', 'params': [2.5, 4, 310., 450.]}}
    ```

    - If you tried to use a `Triangular` distribution but provided only two parameters (e.g., `{{'type': 'Triangular', 'params': [55, 55.5]}}`), you need three parameters `[a, mode, b]`. For example:
    ```python
    {{'type': 'Triangular', 'params': [55, 55.5, 56]}}
    ```
    
    - "It seems you've specified an unsupported distribution (e.g., 'BetaX'). Supported distributions are: Uniform, Normal, LogNormalMuSigma, LogNormal, Beta, Gumbel, and Triangular. Please use one of these. For instance, if you intended to use 'Beta', change 'BetaX' to 'Beta'.

    UncertaintyCat supports the following distributions and requires them to be correctly specified:

    1. **Uniform**
    ```python
    {{'type': 'Uniform', 'params': [lower_bound, upper_bound]}}
    ```

    2. **Normal**
    ```python
    {{'type': 'Normal', 'params': [mean, standard_deviation]}}
    ```

    3. **LogNormalMuSigma**
    ```python
    {{'type': 'LogNormalMuSigma', 'params': [mu, sigma, gamma]}}
    ```
    - `mu`: Mean of the LogNormal variable.
    - `sigma`: Standard deviation.
    - `gamma`: Location parameter (default is `0` if not specified).

    4. **LogNormal**
    ```python
    {{'type': 'LogNormal', 'params': [mu_log, sigma_log, gamma]}}
    ```

    5. **Beta**
    ```python
    {{'type': 'Beta', 'params': [alpha, beta, lower_bound, upper_bound]}}
    ```

    6. **Gumbel**
    ```python
    {{'type': 'Gumbel', 'params': [beta_param, gamma_param]}}
    ```

    7. **Triangular**
    ```python
    {{'type': 'Triangular', 'params': [a, mode, b]}}
    ```

    - Make sure that the number of input variables you unpack in your model function matches `problem['num_vars']` and `problem['names']`.
    - Ensure your model function returns a list or NumPy array.

    Below are three valid model definitions from the `examples` folder that you can reference as starting points:

    **Valid Model Example 1 (from Beam.py)**:
    ```python
    def function_of_interest(X):
        E, F, L, I = X
        Y = F * L**3 / (3 * E * I)
        return [Y]

    problem = {
        'num_vars': 4,
        'names': ['E', 'F', 'L', 'I'],
        'distributions': [
            {'type': 'Beta', 'params': [0.9, 3.1, 2.8e7, 4.8e7]},  # E
            {'type': 'LogNormalMuSigma', 'params': [3.0e4, 9.0e3, 15.0e3]}, # F
            {'type': 'Uniform', 'params': [250., 260.]},             # L
            {'type': 'Beta', 'params': [2.5, 4, 310., 450.]}      # I
        ]
    }

    model = function_of_interest
    ```

    **Valid Model Example 2 (from Ishigami.py)**:
    ```python
    import numpy as np

    def function_of_interest(X):
        x1, x2, x3 = X
        Y = np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)
        return [Y]

    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'distributions': [
            {'type': 'Uniform', 'params': [-np.pi, np.pi]}, # x1
            {'type': 'Uniform', 'params': [-np.pi, np.pi]}, # x2
            {'type': 'Uniform', 'params': [-np.pi, np.pi]}  # x3
        ]
    }

    model = function_of_interest
    ```

    **Valid Model Example 3 (from Truss_Model.py)**:
    ```python
    import numpy as np

    def function_of_interest(X):
        E1, E2, A1, A2, P1, P2, P3, P4, P5, P6 = X
        # Convert units and compute Y as shown in the Truss_Model.py
        # Ensure all parameters and distributions match the expected [type, params] format.
        # ...
        return [y]

    problem = {
        'num_vars': 10,
        'names': ['E1', 'E2', 'A1', 'A2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
        'distributions': [
            {'type': 'LogNormalMuSigma', 'params': [210, 21, 0]},  # E1
            {'type': 'LogNormalMuSigma', 'params': [210, 21, 0]},  # E2
            {'type': 'LogNormalMuSigma', 'params': [20, 2, 0]},   # A1
            {'type': 'LogNormalMuSigma', 'params': [10, 1, 0]},   # A2
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},      # P1
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},      # P2
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},      # P3
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},      # P4
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]},      # P5
            {'type': 'Gumbel', 'params': [5.8477, 0.46622]}       # P6
        ]
    }

    model = function_of_interest
    ```

    These examples show the correct structure, supported distributions, proper parameter counts, and ensure that the number of variables and their names match `num_vars` and `names`.

    Also consult other models like `Borehole_Model.py`, `Material_Stress.py`, and `Portfolio_Risk.py` in the `examples` folder for different distributions and inputs. Ensuring all parameters are valid and consistent will fix the error and allow your simulation to run successfully.
    """

    prompt = f"""
    The user has provided a Python code snippet (a model) that should produce a numerical output from a NumPy array `X`, but it failed execution. Using the instructions and examples below:

    1. Explain what went wrong.
    2. Suggest how to fix it.
    3. Provide at least two valid model definitions as direct examples.
    4. Refer to the `examples` folder for guidance.
    5. No follow-up phrases like 'Let me know...'.

    Error Message:
    ```
    {error_message}
    ```

    User Code:
    ```
    {code_snippet}
    ```

    {examples}
    """
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name
        )
        response_text = chat_completion.choices[0].message.content
        return response_text
    except Exception as e:
        return f"Failed to generate human-friendly explanation: {e}"
    
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
        monte_carlo_simulation(2, model, problem)
    except Exception as e:
        error_explanation = get_human_friendly_error_explanation(code, str(e), language_model)
        raise ValueError(f"Model test failed: {error_explanation}")