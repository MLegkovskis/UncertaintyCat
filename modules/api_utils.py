# modules/api_utils.py

import os
from groq import Groq
import re

def call_groq_api(prompt, model_name="gemma2-9b-it"):
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
