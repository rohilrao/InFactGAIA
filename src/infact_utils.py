import openai
from openai import OpenAI
import anthropic

def call_llm(provider_name: str, api_key: str, model: str, prompt: str) -> str:
    """
    Make an API call to the specified LLM provider
    
    Args:
        provider_name: "openai" or "anthropic"
        api_key: API key for the provider
        model: Model name
        prompt: Text prompt to send
        
    Returns:
        The text response from the model
    """
    if provider_name.lower() == "openai":
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif provider_name.lower() == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    else:
        raise ValueError(f"Unsupported provider: {provider_name}")