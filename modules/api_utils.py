# modules/api_utils.py

import os
from groq import Groq
import openai

def call_groq_api(prompt, model_name="llama3-70b-8192"):
    if model_name == "mistralai/mathstral-7b-v0.1" or model_name == "meta/llama-3.1-405b-instruct":
        # Use OpenAI client with NVIDIA API endpoint
        openai.api_key = os.getenv('NVIDIA_API_KEY')
        openai.api_base = "https://integrate.api.nvidia.com/v1"

        chat_completion = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        response_text = chat_completion.choices[0].message.content

    elif model_name == "gpt-4o":

        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4-turbo",
        )

        response_text = chat_completion.choices[0].message.content.replace("```markdown", "").replace("```", "")

    else:
        # Use Groq client
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
            model=model_name  # Use the selected model
        )
        response_text = chat_completion.choices[0].message.content

    return response_text
