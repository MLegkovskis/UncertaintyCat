import streamlit as st
from modules.api_utils import call_groq_api
from modules.common_prompt import RETURN_INSTRUCTION


def model_understanding(model, problem, model_code_str, is_pce_used=False, original_model_code_str=None, metamodel_str=None, language_model='groq'):
    # Prepare the input parameters and their uncertainties
    input_parameters = []
    for name, dist_info in zip(problem['names'], problem['distributions']):
        input_parameters.append(f"- **{name}**: {dist_info['type']} distribution with parameters {dist_info['params']}")

    inputs_description = '\n'.join(input_parameters)

    model_code = model_code_str

    # Format the model code for inclusion in the prompt
    model_code_formatted = '\n'.join(['    ' + line for line in model_code.strip().split('\n')])

    prompt = f"""
{RETURN_INSTRUCTION}

Given the following user-defined model defined in Python code:

```python
{model_code_formatted}
```

**Important:** For large mathematical expressions,such as lengthy polynomials that will always contain 'Y = (', when converting the code into LaTeX equations, include only the first three terms and the last two terms of the expression. Use an ellipsis (\\dots) to represent the omitted middle terms. Ensure that the equations remain clear and readable.

and the following uncertain input distributions:

{inputs_description}

Please:
- **First of all --> Do not ever simply reprint the supplied model as Python code**. Instead:
  - You must always try to convert the numpy-based original model into clear mathematical equations using LaTeX compatible with Markdown!
  - Provide a description of the model and its physical meaning, including the output variable.
  - Deduce and specify the units of all variables, including the model output.
  - Define all the input variables, even if you are not 100% sure about their meanings.
  - Explicitly state the associated input uncertainties and their characteristics.
"""

    response_key = 'model_understanding_response_markdown'

    if response_key not in st.session_state:
        response_markdown = call_groq_api(prompt, model_name=language_model)
        st.session_state[response_key] = response_markdown
    else:
        response_markdown = st.session_state[response_key]

    st.markdown(response_markdown)