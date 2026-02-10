"""Base agent: wraps LLM provider + Flanes AgentSession.

All Laneswarm agents (coder, reviewer, integrator) inherit from BaseAgent.
Handles LLM calls, token recording, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from flanes.agent_sdk import AgentSession
from flanes.repo import Repository

from ..events import EventBus, EventType
from ..providers import LLMResponse, Message, ProviderRegistry, Tool

logger = logging.getLogger(__name__)


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel all pending tasks on *loop* (mirrors asyncio.run internals)."""
    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return
    for task in to_cancel:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))


class BaseAgent:
    """Base class for all Laneswarm agents.

    Wraps an LLM provider and a Flanes AgentSession. Subclasses override
    run() to implement their specific logic.
    """

    def __init__(
        self,
        repo_path: Path,
        agent_id: str,
        agent_type: str,
        model_id: str,
        registry: ProviderRegistry,
        event_bus: EventBus | None = None,
        lane: str | None = None,
        workspace: str | None = None,
    ) -> None:
        self.repo_path = repo_path
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_id = model_id  # "provider/model-name" format
        self.registry = registry
        self.event_bus = event_bus or EventBus()
        self._lane = lane
        self._workspace = workspace

        # Per-instance logger so output shows which agent is acting
        # e.g. "laneswarm.agent.coder-003" instead of generic "laneswarm.agents.coder"
        self.log = logging.getLogger(f"laneswarm.agent.{agent_id}")

        # Parse provider/model
        self._provider_name, self._model_name = registry.parse_model_id(model_id)

    def create_session(
        self,
        lane: str | None = None,
        workspace: str | None = None,
    ) -> AgentSession:
        """Create a Flanes AgentSession for this agent."""
        return AgentSession(
            repo_path=self.repo_path,
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            model=self.model_id,
            lane=lane or self._lane,
            workspace=workspace or self._workspace,
        )

    def create_repo(self) -> Repository:
        """Create a Repository instance for this agent's thread."""
        return Repository.find(self.repo_path)

    async def call_llm(
        self,
        messages: list[Message],
        max_tokens: int = 8192,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Call the LLM via the provider registry.

        Automatically records token usage.
        """
        start = time.time()
        response = await self.registry.complete(
            model_id=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            temperature=temperature,
            system=system,
        )
        elapsed_ms = (time.time() - start) * 1000

        self.log.debug(
            "LLM call: model=%s tokens_in=%d tokens_out=%d time=%.0fms",
            self.model_id,
            response.usage.input_tokens,
            response.usage.output_tokens,
            elapsed_ms,
        )

        return response

    def call_llm_sync(
        self,
        messages: list[Message],
        max_tokens: int = 8192,
        tools: list[Tool] | None = None,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> LLMResponse:
        """Synchronous wrapper for call_llm.

        Used when running in ThreadPoolExecutor threads.
        Creates a fresh event loop per call so multiple worker threads
        never share a single loop.  Manually manages shutdown to suppress
        the ``anyio`` cancel-scope RuntimeError that the Claude Agent SDK
        triggers during async-generator cleanup on Windows.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.call_llm(messages, max_tokens, tools, temperature, system)
            )
        finally:
            try:
                # Cancel remaining tasks
                _cancel_all_tasks(loop)
                loop.run_until_complete(loop.shutdown_asyncgens())
            except RuntimeError:
                pass  # anyio cancel-scope teardown across tasks
            finally:
                loop.close()

    def emit(self, event_type: EventType, task_id: str = "", **data) -> None:
        """Publish an event."""
        self.event_bus.emit(
            event_type=event_type,
            task_id=task_id,
            agent_id=self.agent_id,
            **data,
        )
