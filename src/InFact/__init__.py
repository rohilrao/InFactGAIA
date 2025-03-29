"""
InFact - A framework for Bayesian updating of beliefs based on evidence.
"""

# Import the main class from the root level
from .infact_node import InFactNode

# Import provider classes from their subpackage
from .providers.llm_provider import LLMProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider

__version__ = "0.1.0"
__all__ = ['InFactNode', 'LLMProvider', 'AnthropicProvider', 'OpenAIProvider']