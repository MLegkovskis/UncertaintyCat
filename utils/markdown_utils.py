# utils/markdown_utils.py

import os
from groq import Groq

# ================ Common Prompts ================

RETURN_INSTRUCTION = """
You are an expert consultant specializing in Uncertainty Quantification (UQ) and Sensitivity Analysis (SA), producing technically-sound, extremely concise briefs for technical audiences.

Guidelines for an Extremely Concise and Insightful Technical Brief:

- Output Format:
  - Your entire response must be in 100% pure valid Markdown format - this is critical for our report generation.
  - Do not include a "CONCLUSIONS" section, as this will be part of a larger document.
  - Do not include a title as the title of the brief is hard-coded and will be supplied externally.
  - Do not use the term function_of_interest to describe the model you're working with; always try to give it a meaningful name.

- Mathematical Content:
  - Include mathematical expressions and equations where appropriate.
  - Use LaTeX syntax for mathematics, enclosed within single dollar signs ($) for inline math or double dollar signs ($$) for display math.

- Style and Clarity:
  - Be extremely succinct; focus solely on key insights and main results.
  - No waffling around; keep the briefs to the point, technical, mathematical, and concise.
  - Organize all content using bullet points, tables etc.
  - Avoid lengthy paragraphs; keep sentences short and to the point.

- Tables and Data:
  - Present numerical data in tables using Markdown syntax.
  - Reference tables in the text, explaining their significance concisely.

Your goal is to produce a succinct, bullet-pointed, and rigorous analysis that is easy to read and immediately highlights the main results, aiding in critical decision-making.

Here are the details:
"""

# ================ Forbidden Patterns ================

forbidden_patterns = [
    # Logical/conditional words in the equations:
    " otherwise",
    " or ",
    # Code-like inequalities:
    ">=",
    "<=",

    # Textual logical connectors in LaTeX:
    "\\text{and}",

    # Direct Python references:
    "import ",
    "def ",
    "np.",
    "fsolve",
    "python code",
    "the code above",

    # Complex LaTeX environments (align or others):
    "\\begin{align",
    "\\end{align",
    "\\begin{align*}",
    "\\end{align*}",

    # Raw HTML tags:
    "<div>",
    "</div>",
    "<table>",
    "</table>",
    "<html>",
    "</html>",
    "<body>",
    "</body>",

    # Markdown code blocks:
    "```",
    "`"
]

# ================ Markdown Generation Functions ================

def generate_refined_prompt(code_snippet, additional_instructions=""):
    """
    Generate a prompt for the model, with optional additional instructions injected.
    
    Parameters
    ----------
    code_snippet : str
        The code snippet to analyze
    additional_instructions : str, optional
        Additional instructions to include in the prompt, by default ""
        
    Returns
    -------
    str
        The generated prompt
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
  1) One brief sentence introducing the model's purpose,
  2) The LaTeX equations,
  3) The Markdown table of input symbols, their names (your best guess) and associated uncertainties.
  
No additional commentary or code excerpts. The output must be ready to render as-is in Markdown.

Any deviation from these rules (e.g., using logical words in equations, showing code, extra braces) invalidates the response. Ensure correctness, simplicity, and complete compliance.

{additional_instructions}
"""
    return base_prompt

def get_markdown_from_code(code_snippet, model_name='gemma2-9b-it', max_attempts=3):
    """
    Get a markdown explanation of a code snippet using the Groq API.
    
    Parameters
    ----------
    code_snippet : str
        The code snippet to analyze
    model_name : str, optional
        The model to use, by default 'gemma2-9b-it'
    max_attempts : int, optional
        Maximum number of attempts to get a valid response, by default 3
        
    Returns
    -------
    str
        The markdown explanation
    """
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
