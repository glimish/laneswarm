"""Tests for dashboard serializers."""

from laneswarm.events import EventBus, EventType, SwarmEvent
from laneswarm.task_graph import Task, TaskGraph, TaskStatus
from laneswarm.dashboard.serializers import (
    serialize_event,
    serialize_snapshot,
    serialize_task,
    serialize_task_detail,
)


def _make_task(task_id="t1", status=TaskStatus.PENDING, **kwargs):
    defaults = {
        "task_id": task_id,
        "title": f"Task {task_id}",
        "description": f"Description for {task_id}",
        "status": status,
    }
    defaults.update(kwargs)
    return Task(**defaults)


def test_serialize_event():
    event = SwarmEvent(
        event_type=EventType.TASK_STARTED,
        task_id="t1",
        agent_id="coder-t1",
        data={"model": "claude-sonnet"},
    )
    result = serialize_event(event)
    assert result["event_type"] == "task_started"
    assert result["task_id"] == "t1"
    assert result["agent_id"] == "coder-t1"
    assert result["data"]["model"] == "claude-sonnet"
    assert isinstance(result["timestamp"], float)


def test_serialize_event_with_non_serializable_data():
    """Non-JSON-safe values in data should be stringified."""
    event = SwarmEvent(
        event_type=EventType.AGENT_WORKING,
        data={"obj": object(), "nested": {"key": 42}},
    )
    result = serialize_event(event)
    assert isinstance(result["data"]["obj"], str)
    assert result["data"]["nested"]["key"] == 42


def test_serialize_task():
    task = _make_task(
        status=TaskStatus.IN_PROGRESS,
        current_phase="coding",
        tokens_used=5000,
        wall_time_ms=12000.0,
        files_written=["src/main.py", "src/utils.py"],
    )
    result = serialize_task(task)
    assert result["task_id"] == "t1"
    assert result["status"] == "in_progress"
    assert result["current_phase"] == "coding"
    assert result["tokens_used"] == 5000
    assert result["files_written"] == ["src/main.py", "src/utils.py"]
    # Compact view should NOT include description or agent_steps
    assert "description" not in result
    assert "agent_steps" not in result


def test_serialize_task_detail():
    task = _make_task(
        status=TaskStatus.COMPLETED,
        current_phase="completed",
        agent_steps=[
            {"phase": "coding", "iteration": 0, "timestamp": 1000.0, "summary": "Started coding"},
            {"phase": "reviewing", "iteration": 0, "timestamp": 1001.0, "summary": "Review accepted"},
        ],
        files_written=["main.py"],
        review_summary="Code looks good.",
        verification_result={"passed": True, "summary": "All checks passed"},
    )
    result = serialize_task_detail(task)
    # Should include everything from compact view plus detail fields
    assert result["task_id"] == "t1"
    assert result["description"] == "Description for t1"
    assert len(result["agent_steps"]) == 2
    assert result["review_summary"] == "Code looks good."
    assert result["verification_result"]["passed"] is True


def test_serialize_snapshot_empty():
    event_bus = EventBus()
    result = serialize_snapshot(None, event_bus)
    assert result["tasks"] == []
    assert result["progress"]["total"] == 0
    assert result["costs"]["total_tokens"] == 0


def test_serialize_snapshot_with_data():
    event_bus = EventBus()
    event_bus.emit(EventType.RUN_STARTED, total_tasks=3)
    event_bus.emit(EventType.TASK_STARTED, task_id="t1")

    tasks = [
        _make_task("t1", TaskStatus.COMPLETED, tokens_used=1000, wall_time_ms=5000),
        _make_task("t2", TaskStatus.IN_PROGRESS, tokens_used=500, wall_time_ms=2000),
        _make_task("t3", TaskStatus.PENDING),
    ]
    graph = TaskGraph(tasks)

    result = serialize_snapshot(graph, event_bus)
    assert len(result["tasks"]) == 3
    assert result["progress"]["total"] == 3
    assert result["progress"]["completed"] == 1
    assert result["costs"]["total_tokens"] == 1500
    assert result["costs"]["total_wall_ms"] == 7000
    # Events should be most-recent first
    assert len(result["events"]) == 2
    assert result["events"][0]["event_type"] == "task_started"


def test_serialize_snapshot_limits_events():
    event_bus = EventBus()
    for i in range(150):
        event_bus.emit(EventType.PROGRESS_UPDATE, completed=i)

    result = serialize_snapshot(None, event_bus, max_events=50)
    assert len(result["events"]) == 50
