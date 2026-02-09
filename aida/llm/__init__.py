"""
AIDA LLM Module

Provides a unified interface for interacting with various LLM backends
using the Adapter Design Pattern.

Supported providers:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude)
- Ollama (Local models)

Example:
    >>> from aida.llm import LLMClientFactory
    >>> client = LLMClientFactory.create("openai", model="gpt-4o-mini")
    >>> response = client.complete("Hello, world!")
"""

from .base import LLMClient, LLMResponse
from .factory import LLMClientFactory
# Import adapters after factory so @register decorators can run
from .adapters import (
    OpenAIAdapter,
    AnthropicAdapter,
    DeepSeekAdapter,
    OllamaAdapter,
)

__all__ = [
    # Base classes
    'LLMClient',
    'LLMResponse',
    # Adapters
    'OpenAIAdapter',
    'AnthropicAdapter',
    'DeepSeekAdapter',
    'OllamaAdapter',
    # Factory
    'LLMClientFactory',
]
