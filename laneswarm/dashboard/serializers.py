"""JSON serializers for dashboard data.

Convert internal objects (SwarmEvent, Task, TaskGraph) into
JSON-safe dicts for sending over WebSocket.
"""

from __future__ import annotations

from typing import Any

from ..events import EventBus, SwarmEvent
from ..task_graph import Task, TaskGraph


def serialize_event(event: SwarmEvent) -> dict[str, Any]:
    """Serialize a SwarmEvent to a JSON-safe dict."""
    return {
        "event_type": event.event_type.value,
        "task_id": event.task_id,
        "agent_id": event.agent_id,
        "data": _safe_data(event.data),
        "timestamp": event.timestamp,
    }


def serialize_task(task: Task) -> dict[str, Any]:
    """Serialize a Task for the task grid (compact view)."""
    return {
        "task_id": task.task_id,
        "title": task.title,
        "status": task.status.value,
        "current_phase": task.current_phase,
        "estimated_complexity": task.estimated_complexity,
        "dependencies": task.dependencies,
        "retries": task.retries,
        "max_retries": task.max_retries,
        "tokens_used": task.tokens_used,
        "wall_time_ms": task.wall_time_ms,
        "error_message": task.error_message,
        "files_written": task.files_written,
    }


def serialize_task_detail(task: Task) -> dict[str, Any]:
    """Serialize a Task with full detail (for task detail panel)."""
    base = serialize_task(task)
    base.update({
        "description": task.description,
        "lane_name": task.lane_name,
        "files_to_create": task.files_to_create,
        "files_to_read": task.files_to_read,
        "agent_steps": task.agent_steps,
        "verification_result": task.verification_result,
        "review_summary": task.review_summary,
        "last_review_feedback": task.last_review_feedback,
        "transition_id": task.transition_id,
    })
    return base


def serialize_snapshot(
    task_graph: TaskGraph | None,
    event_bus: EventBus,
    max_events: int = 100,
) -> dict[str, Any]:
    """Serialize a full state snapshot for initial WS connection.

    Includes all tasks, progress counts, recent events, and cost totals.
    """
    tasks = []
    progress = {"total": 0, "pending": 0, "in_progress": 0, "completed": 0, "failed": 0, "blocked": 0}
    total_tokens = 0
    total_wall_ms = 0.0

    if task_graph is not None:
        tasks = [serialize_task(t) for t in task_graph.tasks]
        progress = task_graph.progress()
        for t in task_graph.tasks:
            total_tokens += t.tokens_used
            total_wall_ms += t.wall_time_ms

    # Recent events (most recent first)
    history = event_bus.history
    recent_events = [
        serialize_event(e) for e in reversed(history[-max_events:])
    ]

    return {
        "tasks": tasks,
        "progress": progress,
        "events": recent_events,
        "costs": {
            "total_tokens": total_tokens,
            "total_wall_ms": total_wall_ms,
        },
    }


def _safe_data(data: dict) -> dict:
    """Make event data JSON-safe by converting non-serializable values."""
    safe = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = [str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for v in value]
        elif isinstance(value, dict):
            safe[key] = _safe_data(value)
        else:
            safe[key] = str(value)
    return safe
