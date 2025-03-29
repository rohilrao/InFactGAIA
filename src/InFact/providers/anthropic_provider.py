from typing import List, Dict, Any, Union
import time
import logging
import anthropic
from .llm_provider import LLMProvider


class AnthropicProvider(LLMProvider):
    """
    Implementation of LLMProvider for Anthropic's Claude models.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use (default: claude-3-opus-20240229)
        """
        super().__init__(api_key, model)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.logger = logging.getLogger(__name__)
    
    def format_message(self, text: str) -> List[Dict]:
        """
        Format a text message for Anthropic's API.
        
        Args:
            text: Plain text message
            
        Returns:
            List[Dict]: Formatted message for Anthropic
        """
        return [{"type": "text", "text": text}]
    
    def send_message(self, content: Union[str, List[Dict]], max_tokens: int = 8192, temperature: float = 0.1) -> str:
        """
        Send a message to Claude and get a response.
        
        Args:
            content: Message content (string or structured content)
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature for sampling
            
        Returns:
            str: The text response from Claude
        """
        # Format the content if it's a string
        if isinstance(content, str):
            formatted_content = self.format_message(content)
        else:
            formatted_content = content
        
        # Send the message
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": formatted_content}]
        )
        
        # Extract the response text
        return self._extract_message_text(message)
    
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
            str: The text response from Claude
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
                if "429" in error_message or "rate_limit_error" in error_message:
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
    
    def _extract_message_text(self, message) -> str:
        """
        Extract text content from an Anthropic message response.
        
        Args:
            message: Anthropic message response object
            
        Returns:
            str: Extracted text content
        """
        if message.content and len(message.content) > 0:
            content_block = message.content[0]
            if hasattr(content_block, 'text'):
                return content_block.text
        return ""