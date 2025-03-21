# utils/core_utils.py

"""
Core utilities for the UncertaintyCat application.
This module contains API utilities and model options.
"""

import os
import streamlit as st
import json
import requests
import traceback
import re
from groq import Groq

# ================ API Utilities ================

def call_groq_api(prompt, model_name="gemma2-9b-it"):
    """
    Call the Groq API with a provided prompt.
    
    Parameters
    ----------
    prompt : str
        The prompt to send to the API
    model_name : str, optional
        The model to use, by default "gemma2-9b-it"
        
    Returns
    -------
    str
        The response text from the API
    """
    client = Groq(
        api_key=os.getenv('GROQ_API_KEY'),
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name
    )
    response_text = chat_completion.choices[0].message.content
    response_text = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)

    return response_text

# ================ Security Utilities ================

def check_code_safety(code_str):
    """
    Checks if the provided code contains potentially unsafe imports or statements.
    
    Args:
        code_str (str): The code string to check
        
    Returns:
        tuple: (is_safe, message) where is_safe is a boolean and message describes any issues
    """
    # List of potentially dangerous modules that should be blocked
    dangerous_modules = [
        'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 
        'urllib', 'ftplib', 'telnetlib', 'smtplib', 'pickle', 
        'multiprocessing', 'tempfile', 'pathlib', 'glob'
    ]
    
    # Check for import statements of dangerous modules
    for module in dangerous_modules:
        # Check different import patterns
        patterns = [
            rf'import\s+{module}\b',                    # import os
            rf'from\s+{module}\s+import',               # from os import path
            rf'import\s+.*,\s*{module}\b',              # import sys, os
            rf'import\s+{module}\s+as',                 # import os as operating_system
        ]
        
        for pattern in patterns:
            if re.search(pattern, code_str, re.IGNORECASE):
                return False, f"Potentially unsafe import detected: '{module}'. This module is restricted for security reasons."
    
    # Check for exec or eval functions
    if re.search(r'exec\s*\(', code_str) or re.search(r'eval\s*\(', code_str):
        return False, "The use of exec() or eval() is not allowed for security reasons."
    
    # Check for __import__ function
    if re.search(r'__import__\s*\(', code_str):
        return False, "The use of __import__() is not allowed for security reasons."
    
    # Check for open() function which could be used to access files
    if re.search(r'open\s*\(', code_str):
        return False, "The use of open() is not allowed for security reasons."
    
    return True, "Code passed security checks."

# ================ Model Options ================

def get_model_options():
    """
    Dynamically get a list of available model files from the examples directory.
    
    Returns
    -------
    list
        List of Python files in the examples directory
    """
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    model_files = []
    
    if os.path.exists(examples_dir) and os.path.isdir(examples_dir):
        for file in os.listdir(examples_dir):
            if file.endswith('.py'):
                model_files.append(file)
    
    # Sort alphabetically for consistent display
    return sorted(model_files)
