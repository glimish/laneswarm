"""Google (Gemini) LLM provider.

Supports:
- API key auth (standard)
- Vertex AI (GCP)
"""

from __future__ import annotations

import logging

from . import LLMResponse, Message, ProviderConfig, TokenUsage, Tool, ToolCall

logger = logging.getLogger(__name__)


class GoogleProvider:
    """Google Gemini provider."""

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
        """Create Google Generative AI client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required: pip install google-generativeai"
            )

        api_key = self._config.resolve_api_key()
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or configure api_key_env in .laneswarm.toml"
            )
        genai.configure(api_key=api_key)
        return genai

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
        genai = self._client

        # Build system instruction
        system_instruction = system
        if not system_instruction:
            system_msgs = [m.content for m in messages if m.role == "system"]
            if system_msgs:
                system_instruction = "\n\n".join(system_msgs)

        gen_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        # Convert messages to Gemini format
        history = []
        last_content = ""
        for msg in messages:
            if msg.role == "system":
                continue
            role = "model" if msg.role == "assistant" else "user"
            if msg == messages[-1] and msg.role == "user":
                last_content = msg.content
            else:
                history.append({"role": role, "parts": [msg.content]})

        chat = gen_model.start_chat(history=history)
        response = chat.send_message(last_content or "Continue.")

        # Parse response
        content = response.text or ""
        usage = TokenUsage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = TokenUsage(
                input_tokens=getattr(response.usage_metadata, "prompt_token_count", 0),
                output_tokens=getattr(response.usage_metadata, "candidates_token_count", 0),
            )

        return LLMResponse(
            content=content,
            tool_calls=[],  # Gemini tool calling handled differently; simplified for v1
            usage=usage,
            model=model,
            stop_reason="stop",
        )

    def list_models(self) -> list[str]:
        return list(self.MODELS)
