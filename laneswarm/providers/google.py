"""Google (Gemini) LLM provider.

Uses the unified Google Gen AI SDK (google-genai), which supports both
the Gemini Developer API (API key) and Vertex AI (GCP).

Replaces the deprecated google-generativeai package.
"""

from __future__ import annotations

import logging

from . import LLMResponse, Message, ProviderConfig, TokenUsage, Tool, ToolCall

logger = logging.getLogger(__name__)


class GoogleProvider:
    """Google Gemini provider using the google-genai SDK."""

    MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    def __init__(self, config: ProviderConfig) -> None:
        self._config = config
        self._client = self._create_client()

    @property
    def name(self) -> str:
        return "google"

    def _create_client(self):
        """Create Google Gen AI client."""
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package required: pip install google-genai"
            )

        api_key = self._config.resolve_api_key()
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or configure api_key_env in .laneswarm.toml"
            )
        return genai.Client(api_key=api_key)

    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send completion request to Gemini."""
        from google.genai import types

        # Build system instruction
        system_instruction = system
        if not system_instruction:
            system_msgs = [m.content for m in messages if m.role == "system"]
            if system_msgs:
                system_instruction = "\n\n".join(system_msgs)

        # Convert messages to Gemini content format
        contents = []
        for msg in messages:
            if msg.role == "system":
                continue
            role = "model" if msg.role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg.content)],
                )
            )

        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        # Parse response
        content = response.text or ""
        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = TokenUsage(
                input_tokens=getattr(um, "prompt_token_count", 0),
                output_tokens=getattr(um, "candidates_token_count", 0),
            )

        return LLMResponse(
            content=content,
            tool_calls=[],
            usage=usage,
            model=model,
            stop_reason="stop",
        )

    def list_models(self) -> list[str]:
        return list(self.MODELS)
