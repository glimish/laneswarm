"""OpenAI LLM provider.

Supports:
- API key auth (standard)
- Azure OpenAI
"""

from __future__ import annotations

import logging

from . import LLMProvider, LLMResponse, Message, ProviderConfig, TokenUsage, Tool, ToolCall

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI GPT provider."""

    MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o1-mini",
        "o3-mini",
    ]

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = self._create_client()

    @property
    def name(self) -> str:
        return "openai"

    def _create_client(self):
        """Create OpenAI client based on auth mode."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        if self._config.auth == "azure":
            from openai import AzureOpenAI

            azure_key = None
            if self._config.azure_api_key_env:
                import os

                azure_key = os.environ.get(self._config.azure_api_key_env)
            return AzureOpenAI(
                azure_endpoint=self._config.azure_endpoint,
                api_key=azure_key,
                api_version="2024-10-21",
            )
        else:
            api_key = self._config.resolve_api_key()
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            # If no explicit key, OpenAI SDK reads OPENAI_API_KEY from env automatically
            if self._config.base_url:
                kwargs["base_url"] = self._config.base_url
            return openai.OpenAI(**kwargs)

    def _build_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert Tool objects to OpenAI function calling format."""
        result = []
        for tool in tools:
            properties = {}
            required = []
            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    required.append(param.name)

            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return result

    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send completion request to OpenAI."""
        import json as json_module

        # Convert messages to OpenAI format
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role == "system" and system:
                continue  # Already added
            api_messages.append({"role": msg.role, "content": msg.content})

        kwargs: dict = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        response = self._client.chat.completions.create(**kwargs)

        # Parse response
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = []

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json_module.loads(tc.function.arguments),
                    )
                )

        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            stop_reason=choice.finish_reason or "",
        )

    def list_models(self) -> list[str]:
        return list(self.MODELS)
