import streamlit as st
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION
from modules.forbidden_patterns import forbidden_patterns

def model_understanding(model, problem, model_code_str, is_pce_used=False, original_model_code_str=None, metamodel_str=None, language_model='groq'):
    # Prepare the input parameters and their uncertainties
    input_parameters = []
    for name, dist_info in zip(problem['names'], problem['distributions']):
        input_parameters.append(f"- **{name}**: {dist_info['type']} distribution with parameters {dist_info['params']}")

    inputs_description = '\n'.join(input_parameters)

    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code_str.strip().split('\n')])

    max_attempts = 5

    def generate_prompt(additional_instructions=""):
        # The prompt now includes a requirement to explain the model's purpose and importance in simple terms.
        return f"""
{RETURN_INSTRUCTION}

Translate the following Python model code into a very concise Markdown explanation, focusing on:
- The model's purpose and physical meaning, described simply so that a non-expert can understand what the model calculates and why it matters.
- How uncertainty in the inputs affects the output and why understanding this uncertainty is important.
- Clear mathematical relationships from the model, presented as LaTeX equations (using $...$ or $$...$$).
- Deduce and specify the units of all variables (even if uncertain, make a best guess).
- Define each input variable and show its uncertainty distribution in a Markdown table.
- If conditions are needed in the equations, use a piecewise LaTeX definition with inequalities (\\geq, \\leq) and no logical words (like 'if', 'and', 'or') inside the math.

Do not reprint the Python code. 
For large polynomials or complex expressions, include only the first three terms and the last two terms, separated by \\dots.

Ensure:
- No code-like >= or <=; use \\geq and \\leq.
- All LaTeX is syntactically correct, balanced, and directly renderable in Markdown.
- Avoid complex LaTeX environments like align. Only use $...$ or $$...$$.
- Provide a single Markdown table summarizing input uncertainties. Variables in math mode, units outside math.
- Output should have:
  1) A brief, layman-friendly introduction describing the model's purpose and why uncertainty matters.
  2) The LaTeX equations showing the core mathematical relationships.
  3) The Markdown table summarizing the uncertain inputs.

Any deviation (such as logical words inside equations, or reprinting code) invalidates the response.

Additional Input Distributions:
{inputs_description}

Original Python code (for reference, do not reprint):
```python
{model_code_formatted}
```

{additional_instructions}
""".strip()

    response_key = 'model_understanding_response_markdown'

    if response_key not in st.session_state:
        prompt = generate_prompt()
        best_response_text = ""
        attempts = 0

        while attempts < max_attempts:
            print(f"\n=== Attempt {attempts + 1} ===")
            print("Using prompt:")
            print(prompt)

            response_text = call_groq_api(prompt, model_name=language_model)
            print("Raw Response from API:")
            print(response_text)

            # Check for forbidden patterns
            found_problems = False
            problem_feedback = ""
            for fp in forbidden_patterns:
                if fp in response_text:
                    found_problems = True
                    problem_feedback += f"\n- The phrase '{fp.strip()}' was found. Remove or replace such words from equations. Use piecewise definitions and inequalities as instructed."

            if found_problems:
                print("Issues found. Updating the prompt with additional instructions.")
                # Update the prompt with the additional instructions to correct the issues
                prompt = generate_prompt(additional_instructions=problem_feedback)
                best_response_text = response_text
            else:
                print("No forbidden patterns detected. Returning this response.")
                st.session_state[response_key] = response_text
                break

            attempts += 1

        if response_key not in st.session_state:
            # If we have no valid response after max_attempts, return best effort
            st.session_state[response_key] = best_response_text

    response_markdown = st.session_state[response_key]
    st.markdown(response_markdown)