"""
BEFORE: this is how the Q&A assistant started life as a notebook cell —
a single hardcoded call, no structure, no error handling, no way to run it
as a service.
"""
import os

question = "What is training/serving skew?"
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("(no API key set) Mock answer: This is a mock answer.")
else:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    print(response.choices[0].message.content)
