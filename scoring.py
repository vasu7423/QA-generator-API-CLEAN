# scoring.py

import openai
import os
from dotenv import load_dotenv

# Load the OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_transparency_score(user_input):
    prompt = f"""
    Evaluate the following product idea for completeness and clarity on a scale of 0 to 1.
    Give only the score (e.g., 0.6, 0.9, etc.)

    Product Idea:
    "{user_input}"
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an evaluator AI that only outputs a numeric score."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=10
    )

    score = response['choices'][0]['message']['content'].strip()
    return float(score)
