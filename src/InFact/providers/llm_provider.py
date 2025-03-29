from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    This defines the interface that all providers must implement.
    """
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the provider
            model: Model name/identifier
        """
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def send_message(self, content: Union[str, List[Dict]], max_tokens: int = 8192, temperature: float = 0.1) -> str:
        """
        Send a message to the LLM and get a response.
        
        Args:
            content: Message content (string or structured content)
            max_tokens: Maximum number of tokens in the response
            temperature: Temperature for sampling
            
        Returns:
            str: The text response from the LLM
        """
        pass
    
    @abstractmethod
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
            str: The text response from the LLM
        """
        pass
    
    @abstractmethod
    def format_message(self, text: str) -> Union[str, List[Dict]]:
        """
        Format a simple text message according to the provider's expected format.
        
        Args:
            text: Plain text message
            
        Returns:
            The formatted message ready to be sent to the provider
        """
        pass