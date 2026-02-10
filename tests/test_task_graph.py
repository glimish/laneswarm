"""Tests for TaskGraph and Task."""

import json

import pytest

from laneswarm.task_graph import Task, TaskGraph, TaskStatus


def test_create_task():
    task = Task(
        task_id="001",
        title="Setup",
        description="Initialize project",
    )
    assert task.task_id == "001"
    assert task.status == TaskStatus.PENDING
    assert task.lane_name == "task-001"
    assert task.retries == 0


def test_task_serialization():
    task = Task(
        task_id="002",
        title="Models",
        description="Create DB models",
        dependencies=["001"],
        estimated_complexity="high",
    )

    d = task.to_dict()
    restored = Task.from_dict(d)

    assert restored.task_id == task.task_id
    assert restored.title == task.title
    assert restored.dependencies == ["001"]
    assert restored.estimated_complexity == "high"
    assert restored.lane_name == "task-002"


def test_task_graph_add_and_get(sample_task_graph):
    assert len(sample_task_graph) == 4
    assert "001" in sample_task_graph
    assert sample_task_graph.get_task("001").title == "Project setup"


def test_task_graph_duplicate_id():
    graph = TaskGraph()
    graph.add_task(Task(task_id="001", title="A", description="..."))
    with pytest.raises(ValueError, match="already exists"):
        graph.add_task(Task(task_id="001", title="B", description="..."))


def test_task_graph_validation_missing_dep():
    graph = TaskGraph([
        Task(task_id="001", title="A", description="...", dependencies=["999"]),
    ])
    errors = graph.validate()
    assert len(errors) == 1
    assert "999" in errors[0]


def test_task_graph_validation_cycle():
    graph = TaskGraph([
        Task(task_id="001", title="A", description="...", dependencies=["002"]),
        Task(task_id="002", title="B", description="...", dependencies=["001"]),
    ])
    errors = graph.validate()
    assert len(errors) == 1
    assert "cycle" in errors[0].lower()


def test_task_graph_validation_valid(sample_task_graph):
    errors = sample_task_graph.validate()
    assert errors == []


def test_get_ready_tasks(sample_task_graph):
    ready = sample_task_graph.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].task_id == "001"


def test_ready_after_completion(sample_task_graph):
    sample_task_graph.mark_completed("001")
    ready = sample_task_graph.get_ready_tasks()
    ids = {t.task_id for t in ready}
    assert ids == {"002", "003"}


def test_ready_after_all_deps(sample_task_graph):
    sample_task_graph.mark_completed("001")
    sample_task_graph.mark_completed("002")
    sample_task_graph.mark_completed("003")
    ready = sample_task_graph.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0].task_id == "004"


def test_mark_failed_with_retry(sample_task_graph):
    sample_task_graph.mark_failed("001", "test error")
    task = sample_task_graph.get_task("001")
    assert task.status == TaskStatus.PENDING  # Reset for retry
    assert task.retries == 1
    assert task.error_message == "test error"


def test_mark_failed_max_retries(sample_task_graph):
    for _ in range(3):
        sample_task_graph.mark_failed("001", "repeated error")
    task = sample_task_graph.get_task("001")
    assert task.status == TaskStatus.FAILED
    assert task.retries == 3


def test_all_complete():
    graph = TaskGraph([
        Task(task_id="001", title="A", description="..."),
    ])
    assert not graph.all_complete()
    graph.mark_completed("001")
    assert graph.all_complete()


def test_progress(sample_task_graph):
    progress = sample_task_graph.progress()
    assert progress["pending"] == 4
    assert progress["total"] == 4

    sample_task_graph.mark_completed("001")
    sample_task_graph.mark_in_progress("002")
    progress = sample_task_graph.progress()
    assert progress["completed"] == 1
    assert progress["in_progress"] == 1
    assert progress["pending"] == 2


def test_topological_order(sample_task_graph):
    order = sample_task_graph.topological_order()
    assert order.index("001") < order.index("002")
    assert order.index("001") < order.index("003")
    assert order.index("002") < order.index("004")
    assert order.index("003") < order.index("004")


def test_json_roundtrip(sample_task_graph):
    json_str = sample_task_graph.to_json()
    restored = TaskGraph.from_json(json_str)
    assert len(restored) == len(sample_task_graph)
    assert restored.get_task("001").title == "Project setup"
    assert restored.get_task("004").dependencies == ["002", "003"]
