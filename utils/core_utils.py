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

# ================ Output Suppression Utility ================
import sys
import contextlib

@contextlib.contextmanager
def suppress_output():
    """
    Context manager to suppress stdout and stderr (useful for noisy library warnings).
    Usage:
        with suppress_output():
            # code that prints unwanted warnings
            ...
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# ================ API Utilities ================

def call_groq_api(prompt, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Call the Groq API with a provided prompt.
    
    Parameters
    ----------
    prompt : str
        The prompt to send to the API
    model_name : str, optional
        The model to use, by default "meta-llama/llama-4-scout-17b-16e-instruct"
        
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

# ================ Chat Interface Utilities ================

def create_chat_interface(session_key, context_generator, input_placeholder="Ask a question...", 
                         disclaimer_text=None, language_model="meta-llama/llama-4-scout-17b-16e-instruct",
                         side_by_side=False):
    """
    Creates a standardized chat interface for UncertaintyCat modules.
    
    Parameters
    ----------
    session_key : str
        Unique key for storing chat messages in session state
    context_generator : callable
        Function that generates the context for the AI assistant
    input_placeholder : str, optional
        Placeholder text for the chat input field
    disclaimer_text : str, optional
        Text to display as a disclaimer above the chat interface
    language_model : str, optional
        The language model to use for generating responses
    side_by_side : bool, optional
        If True, the function will not create any UI elements and will return the necessary
        components for the caller to arrange in a side-by-side layout
        
    Returns
    -------
    dict or None
        If side_by_side is True, returns a dictionary with chat components.
        Otherwise, creates UI elements directly and returns None.
    """
    # Initialize session state for chat messages if not already done
    if f"{session_key}_chat_messages" not in st.session_state:
        st.session_state[f"{session_key}_chat_messages"] = []
    
    if side_by_side:
        # Return components for side-by-side layout
        components = {
            "title": "Ask Questions About This Analysis",
            "disclaimer": disclaimer_text,
            "session_key": session_key,
            "messages": st.session_state[f"{session_key}_chat_messages"],
            "placeholder": input_placeholder,
            "context_generator": context_generator,
            "language_model": language_model
        }
        return components
    else:
        # Traditional vertical layout
        # Add chat interface header
        st.write("### Ask Questions About This Analysis")
        
        # Display disclaimer if provided
        if disclaimer_text:
            st.info(disclaimer_text)
        
        # Create containers for chat interface
        chat_container = st.container()
        input_container = st.container()
        
        # Get user input (placing this at the bottom of the page)
        with input_container:
            prompt = st.chat_input(input_placeholder)
        
        # Display chat messages in the scrollable container
        with chat_container:
            message_area = st.container()
            with message_area:
                # Display existing messages
                for message in st.session_state[f"{session_key}_chat_messages"]:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
        
        # Process user input
        if prompt:
            # Add user message to chat history
            st.session_state[f"{session_key}_chat_messages"].append({"role": "user", "content": prompt})
            
            # Generate context for the assistant
            context = context_generator(prompt)
            
            # Include previous conversation history
            chat_history = ""
            if len(st.session_state[f"{session_key}_chat_messages"]) > 1:
                chat_history = "Previous conversation:\n"
                for i, msg in enumerate(st.session_state[f"{session_key}_chat_messages"][:-1]):
                    role = "User" if msg["role"] == "user" else "Assistant"
                    chat_history += f"{role}: {msg['content']}\n\n"
            
            # Create the final prompt
            chat_prompt = f"""
            {context}
            
            {chat_history}
            
            Current user question: {prompt}
            
            Please provide a helpful, accurate response to this question.
            """
            
            # Call API with chat history
            with st.spinner("Thinking..."):
                try:
                    response_text = call_groq_api(chat_prompt, model_name=language_model)
                except Exception as e:
                    st.error(f"Error calling API: {str(e)}")
                    response_text = "I'm sorry, I encountered an error while processing your question. Please try again."
            
            # Add assistant response to chat history
            st.session_state[f"{session_key}_chat_messages"].append({"role": "assistant", "content": response_text})
            
            # Rerun to display the new message immediately
            st.rerun()
        
        return None

def process_chat_input(components):
    """
    Process chat input for side-by-side layout.
    
    Parameters
    ----------
    components : dict
        Dictionary containing chat components
        
    Returns
    -------
    bool
        True if a rerun is needed, False otherwise
    """
    session_key = components["session_key"]
    prompt = components.get("current_prompt")
    
    if prompt:
        # Add user message to chat history
        st.session_state[f"{session_key}_chat_messages"].append({"role": "user", "content": prompt})
        
        # Generate context for the assistant
        context = components["context_generator"](prompt)
        
        # Include previous conversation history
        chat_history = ""
        if len(st.session_state[f"{session_key}_chat_messages"]) > 1:
            chat_history = "Previous conversation:\n"
            for i, msg in enumerate(st.session_state[f"{session_key}_chat_messages"][:-1]):
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_history += f"{role}: {msg['content']}\n\n"
        
        # Create the final prompt
        chat_prompt = f"""
        {context}
        
        {chat_history}
        
        Current user question: {prompt}
        
        Please provide a helpful, accurate response to this question.
        """
        
        # Call API with chat history
        with st.spinner("Thinking..."):
            try:
                response_text = call_groq_api(chat_prompt, model_name=components["language_model"])
            except Exception as e:
                st.error(f"Error calling API: {str(e)}")
                response_text = "I'm sorry, I encountered an error while processing your question. Please try again."
        
        # Add assistant response to chat history
        st.session_state[f"{session_key}_chat_messages"].append({"role": "assistant", "content": response_text})
        
        return True
    
    return False

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
