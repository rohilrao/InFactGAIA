from typing import List, Dict, Any, Union
import time
import logging
import openai
from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    Implementation of LLMProvider for OpenAI's GPT models.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (default: gpt-4-turbo)
        """
        super().__init__(api_key, model)
        self.client = openai.OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    def format_message(self, text: str) -> List[Dict]:
        """
        Format a text message for OpenAI's API.
        
        Args:
            text: Plain text message
            
        Returns:
            List[Dict]: Formatted message for OpenAI
        """
        return [{"role": "user", "content": text}]
    
    def send_message(self, content: Union[str, List[Dict]], max_tokens: int = 8192, temperature: float = 0.1) -> str:
        """
        Send a message to GPT and get a response.
        
        Args:
            content: Message content (string or structured content)
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature for sampling
            
        Returns:
            str: The text response from GPT
        """
        # Format the content for OpenAI's API
        if isinstance(content, str):
            messages = [{"role": "user", "content": content}]
        else:
            # Handle structured content for OpenAI
            # This is a simplification - complex content types like images would need special handling
            messages = []
            for item in content:
                if item.get("type") == "text":
                    messages.append({"role": "user", "content": item.get("text", "")})
                # For other content types, we'd need custom handling
        
        # Send the message
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract the response text
        return completion.choices[0].message.content
    
    def send_with_retry(self, content: Union[str, List[Dict]], max_tokens: int = 8192, 
                         temperature: float = 0.1, max_attempts: int = 3) -> str:
        """
        Send a message with retry logic for handling rate limits.
        
        Args:
            content: Message content
            max_tokens: Maximum tokens in response
            temperature: Temperature for sampling
            max_attempts: Maximum number of retry attempts
            
        Returns:
            str: The text response from GPT
        """
        attempt = 1
        backoff_time = 2  # Initial backoff time in seconds
        
        while attempt <= max_attempts:
            try:
                return self.send_message(content, max_tokens, temperature)
            
            except Exception as e:
                error_message = str(e)
                self.logger.warning(f"API request failed (attempt {attempt}/{max_attempts}): {error_message}")
                
                # Check if it's a rate limit error
                if "429" in error_message or "rate limit" in error_message.lower():
                    if attempt < max_attempts:
                        self.logger.info(f"Rate limit hit. Backing off for {backoff_time} seconds.")
                        time.sleep(backoff_time)
                        backoff_time *= 2  # Exponential backoff
                        attempt += 1
                    else:
                        self.logger.error("Max retry attempts reached after rate limiting.")
                        raise
                else:
                    # If it's not a rate limit error, re-raise
                    self.logger.error(f"Non-rate-limit error: {error_message}")
                    raise