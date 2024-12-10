from groq import Groq
import os
import streamlit as st

def generate_prompt(code_snippet):
    return f"""
- Translate the following NumPy-based Python function into a markdown document that explains the following model defined in python:
{code_snippet}
- You must never re-print the model definition as python code.
- You must keep your response to an abolute minimum - keep it extremely concise! If you are supplied with particularly large model with many inputs, feel free to obfuscate some of the maths when translating the model into latex/Markdown.
- Present the mathematical equations using LaTex.
- Tabulate (into ONE table!) the associated input uncertainties and their characteristics as detailed in the 'problem' dictionary.

Provide the output in pure Markdown without additional explanations!
"""

def get_markdown_from_code(code_snippet, model_name='gemma2-9b-it'):
    try:
        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        prompt = generate_prompt(code_snippet)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model='gemma2-9b-it'  # or model=model_name if you want to pass it as an argument
        )
        response_text = chat_completion.choices[0].message.content
        return response_text
    except Exception as e:
        st.error(f"Error generating markdown interpretation: {e}")
        return None