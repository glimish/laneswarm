"""Multi-provider LLM abstraction layer.

Supports Anthropic (Claude), OpenAI, Google (Gemini), and local models (Ollama).
Each provider implements the LLMProvider protocol. Auth supports API keys,
subscription billing, and cloud platform integration (Vertex, Bedrock, Azure).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ToolParameter:
    """A parameter for a tool."""

    name: str
    type: str
    description: str
    required: bool = True


@dataclass
class Tool:
    """A tool definition for LLM function calling."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    """Token usage from an LLM response."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    stop_reason: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Each provider must implement the complete() method.
    Providers handle their own auth (API key, subscription, cloud platform).
    """

    @property
    def name(self) -> str:
        """Provider name (e.g., 'anthropic', 'openai', 'google', 'ollama')."""
        ...

    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a completion request to the LLM."""
        ...

    def list_models(self) -> list[str]:
        """List available model IDs for this provider."""
        ...


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    name: str
    auth: str = "api_key"  # "api_key" | "subscription" | "vertex" | "bedrock" | "azure"
    api_key_env: str = ""
    api_key: str = ""
    base_url: str = ""
    # Cloud platform fields
    vertex_project: str = ""
    vertex_region: str = ""
    bedrock_region: str = ""
    azure_endpoint: str = ""
    azure_api_key_env: str = ""
    # Subscription fields
    subscription_token_env: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        """Resolve API key from direct value or environment variable."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def resolve_subscription_token(self) -> str | None:
        """Resolve subscription token from environment variable."""
        if self.subscription_token_env:
            return os.environ.get(self.subscription_token_env)
        return None


class ProviderRegistry:
    """Registry for LLM providers.

    Model IDs use 'provider/model-name' format (e.g., 'anthropic/claude-opus-4-6').
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider

    def get(self, name: str) -> LLMProvider:
        """Get a provider by name."""
        if name not in self._providers:
            available = ", ".join(sorted(self._providers.keys()))
            raise KeyError(f"Provider '{name}' not registered. Available: {available}")
        return self._providers[name]

    def parse_model_id(self, model_id: str) -> tuple[str, str]:
        """Parse 'provider/model-name' into (provider_name, model_name)."""
        if "/" not in model_id:
            raise ValueError(
                f"Model ID '{model_id}' must use 'provider/model-name' format "
                f"(e.g., 'anthropic/claude-opus-4-6')"
            )
        provider_name, model_name = model_id.split("/", 1)
        return provider_name, model_name

    async def complete(
        self,
        model_id: str,
        messages: list[Message],
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send a completion request using the provider/model from the model_id."""
        provider_name, model_name = self.parse_model_id(model_id)
        provider = self.get(provider_name)
        return await provider.complete(
            messages=messages,
            model=model_name,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temperature,
            system=system,
        )

    @property
    def providers(self) -> dict[str, LLMProvider]:
        return dict(self._providers)
