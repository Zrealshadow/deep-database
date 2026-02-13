"""
Base classes for LLM adapters.

Defines the abstract interface that all LLM adapters must implement.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class LLMResponse:
    """Standardized response from LLM adapters"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None

    @property
    def text(self) -> str:
        """Alias for content"""
        return self.content


@dataclass
class Message:
    """Chat message format"""
    role: str  # "system", "user", "assistant"
    content: str


class LLMClient(ABC):
    """
    Abstract base class for LLM adapters.

    All LLM adapters must implement this interface to ensure
    consistent behavior across different providers.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model identifier (provider-specific)
            api_key: API key for authentication
            **kwargs: Additional provider-specific options
        """
        self.model = model
        if api_key is None and self.env_key:
            api_key = os.environ.get(self.env_key)

        assert api_key is not None, "API key must be provided"
        
        self.api_key = api_key
        self.options = kwargs

    @property
    def env_key(self) -> Optional[str]:
        """Return the environment variable name for the API key, or None."""
        return None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'anthropic')"""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider"""
        ...

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            response_format: Response format spec, e.g. {"type": "json_object"}
            **kwargs: Additional provider-specific options

        Returns:
            LLMResponse with the generated content
        """
        ...

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a chat completion for the given messages.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            response_format: Response format spec, e.g. {"type": "json_object"}
            **kwargs: Additional provider-specific options

        Returns:
            LLMResponse with the generated content
        """
        ...

    def get_model(self) -> str:
        """Get the current model, falling back to default"""
        return self.model or self.default_model
