"""Ollama (local models) LLM provider.

Supports running local models like Llama, Mistral, CodeLlama, etc.
No API key needed — connects to a local Ollama server.
"""

from __future__ import annotations

import logging

from . import LLMResponse, Message, ProviderConfig, TokenUsage, Tool, ToolCall

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Ollama local model provider."""

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._base_url = config.base_url or "http://localhost:11434"

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send completion request to Ollama.

        Uses the OpenAI-compatible API that Ollama exposes.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required for Ollama provider: pip install openai"
            )

        client = openai.OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="ollama",  # Ollama doesn't need a real key
        )

        # Build messages
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "system" and system:
                continue
            api_messages.append({"role": msg.role, "content": msg.content})

        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return LLMResponse(
            content=content,
            tool_calls=[],
            usage=usage,
            model=model,
            stop_reason=choice.finish_reason or "",
        )

    def list_models(self) -> list[str]:
        """List models available on the local Ollama instance."""
        try:
            import httpx

            response = httpx.get(f"{self._base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            logger.debug("Could not list Ollama models", exc_info=True)
        return []
