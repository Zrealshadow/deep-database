"""
LLM Client Factory

Provides a unified factory for creating LLM clients.
"""

from typing import Optional, Dict, Type
from .base import LLMClient


class LLMClientFactory:
    """
    Factory for creating LLM clients.

    Adapters self-register using the @LLMClientFactory.register decorator.

    Example:
        >>> client = LLMClientFactory.create("openai", model="gpt-4o-mini")
        >>> response = client.complete("Hello!")

        # Custom adapter
        >>> @LLMClientFactory.register("custom")
        ... class CustomAdapter(LLMClient): ...
    """

    _adapters: Dict[str, Type[LLMClient]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register an LLM adapter.

        Usage:
            @LLMClientFactory.register("openai")
            class OpenAIAdapter(LLMClient): ...
        """
        def decorator(adapter_class: Type[LLMClient]):
            cls._adapters[name.lower()] = adapter_class
            return adapter_class
        return decorator

    @classmethod
    def create(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> LLMClient:
        """
        Create an LLM client for the specified provider.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic', 'ollama')
            model: Model identifier (uses provider default if not specified)
            api_key: API key (uses environment variable if not specified)
            **kwargs: Additional provider-specific options

        Returns:
            LLMClient instance

        Raises:
            ValueError: If provider is not registered
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._adapters:
            available = ", ".join(cls._adapters.keys())
            raise ValueError(
                f"Unknown provider: '{provider}'. Available: {available}"
            )

        adapter_class = cls._adapters[provider_lower]
        return adapter_class(model=model, api_key=api_key, **kwargs)

    @classmethod
    def available_providers(cls) -> list:
        """Return list of available provider names"""
        return list(cls._adapters.keys())
