"""
Concrete LLM Adapters

Implements the LLMClient interface for various LLM providers.
"""

from typing import Optional, List, Dict, Any
from .base import LLMClient, LLMResponse, Message
from .factory import LLMClientFactory


@LLMClientFactory.register("openai")
class OpenAIAdapter(LLMClient):
    """
    Adapter for OpenAI API (GPT models).

    Requires: pip install openai

    Example:
        >>> adapter = OpenAIAdapter(model="gpt-4o-mini")
        >>> response = adapter.complete("Hello!")
    """

    @property
    def env_key(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"

    def _get_client(self):
        """Lazy load OpenAI client"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install with: pip install openai"
            )
        return OpenAI(api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.chat(messages, temperature, max_tokens, response_format, **kwargs)

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        client = self._get_client()

        # Convert to OpenAI format
        oai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        request_kwargs = {
            "model": self.get_model(),
            "messages": oai_messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        if response_format:
            request_kwargs["response_format"] = response_format

        response = client.chat.completions.create(**request_kwargs)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            provider=self.provider_name,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else None,
            raw_response=response
        )


@LLMClientFactory.register("anthropic")
class AnthropicAdapter(LLMClient):
    """
    Adapter for Anthropic API (Claude models).

    Requires: pip install anthropic

    Example:
        >>> adapter = AnthropicAdapter(model="claude-3-haiku-20240307")
        >>> response = adapter.complete("Hello!")
    """

    @property
    def env_key(self) -> str:
        return "ANTHROPIC_API_KEY"

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return "claude-3-haiku-20240307"

    def _get_client(self):
        """Lazy load Anthropic client"""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not found. Install with: pip install anthropic"
            )
        return anthropic.Anthropic(api_key=self.api_key)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        messages = [Message(role="user", content=prompt)]

        # Anthropic handles system prompt separately
        if system_prompt:
            kwargs["system"] = system_prompt

        return self.chat(messages, temperature, max_tokens, response_format, **kwargs)

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        client = self._get_client()

        # Extract system prompt if present
        system = kwargs.pop("system", "")
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        response = client.messages.create(
            model=self.get_model(),
            max_tokens=max_tokens or 1024,
            system=system,
            messages=anthropic_messages,
            temperature=temperature,
            **kwargs
        )

        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            provider=self.provider_name,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            } if response.usage else None,
            raw_response=response
        )


@LLMClientFactory.register("deepseek")
class DeepSeekAdapter(OpenAIAdapter):
    """
    Adapter for DeepSeek API using OpenAI-compatible interface.

    Requires: pip install openai

    Example:
        >>> adapter = DeepSeekAdapter(model="deepseek-chat")
        >>> response = adapter.complete("Hello!")
    """

    @property
    def env_key(self) -> str:
        return "DEEPSEEK_API_KEY"

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def default_model(self) -> str:
        return "deepseek-chat"

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not found. Install with: pip install openai"
            )
        return OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")


@LLMClientFactory.register("ollama")
class OllamaAdapter(LLMClient):
    """
    Adapter for Ollama local LLM server.

    Requires: Ollama running locally (https://ollama.ai)

    Example:
        >>> adapter = OllamaAdapter(model="llama2")
        >>> response = adapter.complete("Hello!")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,  # Not used for Ollama
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def default_model(self) -> str:
        return "llama2"

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return self.chat(messages, temperature, max_tokens, response_format, **kwargs)

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package not found. Install with: pip install requests"
            )

        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        request_data = {
            "model": self.get_model(),
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=request_data
        )
        response.raise_for_status()
        result = response.json()

        return LLMResponse(
            content=result["message"]["content"],
            model=result.get("model", self.get_model()),
            provider=self.provider_name,
            usage={
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
            } if "eval_count" in result else None,
            raw_response=result
        )
