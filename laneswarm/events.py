"""Event bus for real-time feedback.

Callback-based: orchestrator publishes SwarmEvent objects,
subscribers (CLI display, web dashboard) receive them.

Thread-safe: a lock protects the subscriber list and history so
worker threads can safely publish events concurrently.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    """Types of events published by the orchestrator."""

    # Task lifecycle
    TASK_QUEUED = "task_queued"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRYING = "task_retrying"

    # Agent lifecycle
    AGENT_SPAWNED = "agent_spawned"
    AGENT_WORKING = "agent_working"
    AGENT_FINISHED = "agent_finished"

    # Review
    REVIEW_STARTED = "review_started"
    REVIEW_ACCEPTED = "review_accepted"
    REVIEW_REJECTED = "review_rejected"

    # Integration
    PROMOTE_STARTED = "promote_started"
    PROMOTE_COMPLETED = "promote_completed"
    PROMOTE_CONFLICT = "promote_conflict"

    # Orchestrator
    PLAN_CREATED = "plan_created"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    PROGRESS_UPDATE = "progress_update"

    # Verification
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"

    # Integration validation (post-completion)
    INTEGRATION_VALIDATION_STARTED = "integration_validation_started"
    INTEGRATION_VALIDATION_PASSED = "integration_validation_passed"
    INTEGRATION_VALIDATION_FAILED = "integration_validation_failed"

    # Smoke testing (post-completion runtime validation)
    SMOKE_TEST_STARTED = "smoke_test_started"
    SMOKE_TEST_PASSED = "smoke_test_passed"
    SMOKE_TEST_FAILED = "smoke_test_failed"

    # Cost
    COST_UPDATE = "cost_update"
    BUDGET_WARNING = "budget_warning"


@dataclass
class SwarmEvent:
    """An event from the Laneswarm orchestrator."""

    event_type: EventType
    task_id: str = ""
    agent_id: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        parts = [f"[{self.event_type.value}]"]
        if self.task_id:
            parts.append(f"task={self.task_id}")
        if self.agent_id:
            parts.append(f"agent={self.agent_id}")
        if self.data:
            for key, value in self.data.items():
                parts.append(f"{key}={value}")
        return " ".join(parts)


EventCallback = Callable[[SwarmEvent], None]


class EventBus:
    """Thread-safe callback-based event bus.

    Subscribers register callbacks that are invoked synchronously
    when events are published.  A lock protects the subscriber list
    and event history so multiple worker threads can publish safely.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[EventCallback] = []
        self._history: list[SwarmEvent] = []
        self._max_history: int = 1000

    def subscribe(self, callback: EventCallback) -> None:
        """Subscribe to all events."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """Unsubscribe from events."""
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def publish(self, event: SwarmEvent) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            # Snapshot the subscriber list under the lock so callbacks
            # run outside the lock (avoids deadlocks if a callback publishes).
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                pass  # Don't let subscriber errors break the orchestrator

    def emit(
        self,
        event_type: EventType,
        task_id: str = "",
        agent_id: str = "",
        **data: Any,
    ) -> None:
        """Convenience method to create and publish an event."""
        self.publish(SwarmEvent(
            event_type=event_type,
            task_id=task_id,
            agent_id=agent_id,
            data=data,
        ))

    @property
    def history(self) -> list[SwarmEvent]:
        """Get event history (returns a copy)."""
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        """Clear event history."""
        with self._lock:
            self._history.clear()
