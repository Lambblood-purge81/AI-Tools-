import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Basic GPT response
def generate_gpt_response(prompt, model="gpt-4"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streaming GPT response with memory/history
def generate_gpt_response_with_history(messages: list, model="gpt-4"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        collected_chunks = []
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                collected_chunks.append(chunk.choices[0].delta.content)
                yield chunk.choices[0].delta.content
        full_response = ''.join(collected_chunks)
    except Exception as e:
        yield f"❌ Error: {str(e)}"