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
