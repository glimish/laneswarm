"""Anthropic (Claude) LLM provider.

Supports:
- API key auth (standard)
- Vertex AI (GCP)
- Bedrock (AWS)
"""

from __future__ import annotations

import logging

from . import LLMProvider, LLMResponse, Message, ProviderConfig, TokenUsage, Tool, ToolCall

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Anthropic Claude provider."""

    MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = self._create_client()

    @property
    def name(self) -> str:
        return "anthropic"

    def _create_client(self):
        """Create Anthropic client based on auth mode."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        if self._config.auth == "vertex":
            from anthropic import AnthropicVertex

            return AnthropicVertex(
                project_id=self._config.vertex_project,
                region=self._config.vertex_region or "us-east5",
            )
        elif self._config.auth == "bedrock":
            from anthropic import AnthropicBedrock

            return AnthropicBedrock(
                aws_region=self._config.bedrock_region or "us-east-1",
            )
        else:
            # API key auth — resolve from config or let the SDK read ANTHROPIC_API_KEY from env
            api_key = self._config.resolve_api_key()
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            # If no explicit key, Anthropic SDK reads ANTHROPIC_API_KEY from env automatically
            return anthropic.Anthropic(**kwargs)

    def _build_tools(self, tools: list[Tool]) -> list[dict]:
        """Convert Tool objects to Anthropic tool format."""
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
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
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
        """Send completion request to Claude."""
        # Convert messages to Anthropic format
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                # System messages handled separately in Anthropic API
                continue
            api_messages.append({"role": msg.role, "content": msg.content})

        # Extract system from messages if not provided
        if system is None:
            system_msgs = [m.content for m in messages if m.role == "system"]
            if system_msgs:
                system = "\n\n".join(system_msgs)

        kwargs: dict = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._build_tools(tools)

        response = self._client.messages.create(**kwargs)

        # Parse response
        content_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            model=model,
            stop_reason=response.stop_reason or "",
        )

    def list_models(self) -> list[str]:
        return list(self.MODELS)
