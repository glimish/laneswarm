"""Claude Agent SDK provider.

Uses the Claude CLI via the Agent SDK, enabling subscription-based auth
(Claude Pro/Max) without requiring an API key. The CLI handles authentication
via `claude login` (OAuth).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys

from . import LLMProvider, LLMResponse, Message, ProviderConfig, TokenUsage, Tool, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows: prevent CTRL_C_EVENT from killing Claude CLI child processes.
#
# On Windows, all processes sharing a console receive CTRL_C_EVENT.  When
# multiple Claude CLI subprocesses run in parallel the stray signal kills
# them with exit code 0xC000013A (STATUS_CONTROL_C_EXIT).
#
# We monkey-patch ``anyio.open_process`` to add CREATE_NEW_PROCESS_GROUP
# so each Claude CLI process is in its own group and immune to the parent
# console's CTRL_C broadcasts.
# ---------------------------------------------------------------------------
_PATCHED = False


def _patch_subprocess_for_windows() -> None:
    """One-time patch: inject CREATE_NEW_PROCESS_GROUP on Windows.

    Patches ``anyio.open_process`` (the top-level function that the Claude
    SDK imports) so every child process is spawned in its own process group,
    preventing CTRL_C_EVENT from killing Claude CLI instances.
    """
    global _PATCHED
    if _PATCHED or sys.platform != "win32":
        return
    _PATCHED = True

    import anyio._core._subprocesses as _sp_mod  # type: ignore[import-untyped]
    import anyio  # noqa: F811

    _original = _sp_mod.open_process

    async def _patched(*args, **kwargs):  # type: ignore[no-untyped-def]
        flags = kwargs.get("creationflags", 0)
        flags |= subprocess.CREATE_NEW_PROCESS_GROUP
        kwargs["creationflags"] = flags
        return await _original(*args, **kwargs)

    # Patch both the module-level function and the public import
    _sp_mod.open_process = _patched
    anyio.open_process = _patched

# Sentinel to check if the SDK is available
_SDK_AVAILABLE: bool | None = None


def is_claude_sdk_available() -> bool:
    """Check if the Claude Agent SDK and CLI are available."""
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is not None:
        return _SDK_AVAILABLE

    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        _SDK_AVAILABLE = False
        return False

    # Check that the claude CLI binary exists
    cli = shutil.which("claude")
    if cli is None:
        _SDK_AVAILABLE = False
        return False

    _SDK_AVAILABLE = True
    return True


class ClaudeSDKProvider:
    """Provider that uses the Claude Agent SDK (spawns claude CLI).

    This provider does NOT require an API key. Authentication is handled
    by the Claude CLI via `claude login`, supporting Claude Pro/Max
    subscription billing.
    """

    MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._config = config
        if not is_claude_sdk_available():
            raise RuntimeError(
                "Claude Agent SDK not available. Install with: pip install claude-agent-sdk\n"
                "Also ensure the Claude CLI is installed and authenticated: claude login"
            )
        self._cli_path = shutil.which("claude")
        _patch_subprocess_for_windows()

    @property
    def name(self) -> str:
        return "claude"

    async def complete(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int = 4096,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Send completion request via Claude Agent SDK."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        # Build the prompt from user messages only (system goes in options)
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                continue  # System handled via ClaudeAgentOptions.system_prompt
            prompt_parts.append(msg.content)

        prompt = "\n\n".join(prompt_parts)

        # Prepend directive to prevent tool-seeking behavior — the Claude CLI's
        # default mode is to use file-editing tools, but we want text output only.
        text_output_directive = (
            "IMPORTANT: You have NO tools available. Output ALL content directly "
            "as text in your response. Never say 'let me create' or 'I will set up' "
            "— just output your response directly. If asked to produce JSON, "
            "output the JSON directly. If asked to produce code files, use "
            "markdown code blocks.\n\n"
        )
        effective_system = text_output_directive + (system or "You are a helpful coding assistant.")

        options = ClaudeAgentOptions(
            model=model,
            max_turns=1,
            permission_mode="bypassPermissions",
            cli_path=self._cli_path,
            # Disable all tools so Claude outputs text only (no file editing)
            allowed_tools=[],
            # Override the default Claude Code system prompt
            system_prompt=effective_system,
        )

        # Collect all messages — do NOT break early to avoid async generator cleanup issues
        content_parts: list[str] = []
        usage = TokenUsage()
        result_content = ""

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        content_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                if message.result:
                    result_content = message.result
                # Extract usage if available
                if message.usage:
                    # The Claude CLI uses prompt caching, so input tokens are
                    # split across input_tokens, cache_creation_input_tokens,
                    # and cache_read_input_tokens.  Sum all three for the true
                    # input token count.
                    raw_in = message.usage.get("input_tokens", 0)
                    cache_create = message.usage.get("cache_creation_input_tokens", 0)
                    cache_read = message.usage.get("cache_read_input_tokens", 0)
                    usage = TokenUsage(
                        input_tokens=raw_in + cache_create + cache_read,
                        output_tokens=message.usage.get("output_tokens", 0),
                    )

        assistant_content = "\n".join(content_parts)

        logger.debug(
            "Claude SDK response: assistant_content=%d chars, result_content=%d chars",
            len(assistant_content), len(result_content),
        )

        # Use the LONGER of result vs. assistant content.
        # ResultMessage.result is sometimes a summary that loses code blocks,
        # while AssistantMessage content has the full response with formatting.
        if result_content and assistant_content:
            final_content = (
                assistant_content
                if len(assistant_content) >= len(result_content)
                else result_content
            )
            logger.debug(
                "Using %s (%d chars)",
                "assistant_content" if final_content is assistant_content else "result_content",
                len(final_content),
            )
        else:
            final_content = result_content or assistant_content

        if not final_content:
            logger.warning("Claude SDK returned empty response (no content or result)")

        return LLMResponse(
            content=final_content,
            tool_calls=[],
            usage=usage,
            model=model,
            stop_reason="end_turn",
        )

    def list_models(self) -> list[str]:
        return list(self.MODELS)
