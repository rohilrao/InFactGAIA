import openai

def call_llm(provider_name: str, api_key: str, model: str, prompt: str) -> str:
    if provider_name.lower() == "openai":
        openai.api_key = api_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']

    elif provider_name.lower() == "anthropic":
        raise NotImplementedError("Anthropic support not yet implemented.")

    else:
        raise ValueError(f"Unsupported provider: {provider_name}")