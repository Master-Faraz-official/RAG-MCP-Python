import os
import requests
from dotenv import load_dotenv

load_dotenv()

LM_STUDIO_API = os.getenv("LM_STUDIO_URL", "http://localhost:11435/v1/completions")

def messages_to_prompt(messages):
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
        else:
            prompt += f"{role.capitalize()}: {content}\n"
    return prompt

def query_lm(
    messages: list,
    model: str = "llama-3.2-1b-instruct",
    max_tokens: int = 300,
    temperature: float = 0.7
):
    prompt = messages_to_prompt(messages)
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    print("DEBUG: Payload sent to LM Studio:", data)  # For debugging

    try:
        response = requests.post(LM_STUDIO_API, json=data)
        response.raise_for_status()
        result = response.json()
        print("DEBUG: Full response:", result)  # Debug print
        # Adjust this line based on actual response structure
        return result.get("choices", [{}])[0].get("text", "").strip()
    except Exception as e:
        return f"Error: {e}"

def query_lm_from_text(prompt, model="llama-3.2-1b-instruct", max_tokens=300, temperature=0.7):
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(LM_STUDIO_API, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "").strip()
    except Exception as e:
        return f"Error: {e}"

