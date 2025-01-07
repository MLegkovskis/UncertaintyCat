import os
from groq import Groq
from modules.forbidden_patterns import forbidden_patterns

def generate_refined_prompt(code_snippet, additional_instructions=""):
    """
    Generate a prompt for the model, with optional additional instructions injected.
    """
    base_prompt = f"""
Translate the following Python model code into a very concise Markdown explanation of the underlying mathematical relationships and input uncertainties.

{code_snippet}

**Requirements:**

- Do not show or mention the Python code. Provide only a minimal, very concise explanation and the requested outputs.
- Present all mathematical relationships as proper LaTeX equations using only inline ($...$) or display ($$...$$) math blocks.
- If conditions are needed, use piecewise definitions with standard math inequalities, for example:
  $$f(x)=\\begin{{cases}}
  a & x \\geq b \\\\
  c & x < b
  \\end{{cases}}$$
- Use only LaTeX inequalities (\\geq, \\leq) instead of code-like >= or <=.
- All LaTeX must be syntactically correct: balanced braces, no extraneous characters, and directly renderable in standard Markdown.
- Avoid complex LaTeX environments like align. Only use $...$ or $$...$$.
- Provide exactly one Markdown table summarizing the input uncertainties, using 'problem'. Present variables in math mode, e.g. $Q$, and units (if any) as plain text outside math.
- Output only Markdown:
  1) One brief sentence introducing the model’s purpose,
  2) The LaTeX equations,
  3) The Markdown table of input symbols, their names (your best guess) and associated uncertainties.
  
No additional commentary or code excerpts. The output must be ready to render as-is in Markdown.

Any deviation from these rules (e.g., using logical words in equations, showing code, extra braces) invalidates the response. Ensure correctness, simplicity, and complete compliance.

{additional_instructions}
"""
    return base_prompt

def get_markdown_from_code(code_snippet, model_name='llama-3.3-70b-versatile', max_attempts=5):    
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    
    # Initial prompt
    prompt = generate_refined_prompt(code_snippet)
    attempts = 0
    best_response_text = ""

    while attempts < max_attempts:
        print(f"\n=== Attempt {attempts + 1} ===")
        print("Using prompt:")
        print(prompt)

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name
            )
            response_text = chat_completion.choices[0].message.content
            print("Raw Response from API:")
            print(response_text)

            # Check for forbidden patterns
            found_problems = False
            problem_feedback = ""
            for fw in forbidden_patterns:
                if fw in response_text:
                    found_problems = True
                    # Add feedback to the prompt
                    problem_feedback += f"\n- Please remove or replace '{fw.strip()}' in equations. Use piecewise notation and inequalities as instructed."

            # You can also add more checks here for balanced braces or other LaTeX syntax issues
            # If you find other issues, append instructions to problem_feedback accordingly.

            if found_problems:
                print("Issues found. Updating the prompt with additional instructions.")
                # Update the prompt by adding new instructions to avoid previously encountered errors
                prompt = generate_refined_prompt(code_snippet, additional_instructions=problem_feedback)
                best_response_text = response_text
            else:
                print("No forbidden patterns detected. Returning this response.")
                return response_text

            attempts += 1
        except Exception as e:
            print(f"Error during generation on attempt {attempts + 1}: {e}")
            attempts += 1

    print("Maximum attempts reached. Returning best effort Markdown.")
    return best_response_text
