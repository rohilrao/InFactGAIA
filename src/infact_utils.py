import openai
from openai import OpenAI

def call_llm(provider_name: str, api_key: str, model: str, prompt: str) -> str:
    if provider_name.lower() == "openai":
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif provider_name.lower() == "anthropic":
        raise NotImplementedError("Anthropic support not yet implemented.")

    else:
        raise ValueError(f"Unsupported provider: {provider_name}")
