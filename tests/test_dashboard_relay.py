"""Tests for dashboard relay."""

import asyncio
import json

import pytest

from laneswarm.events import EventBus, EventType
from laneswarm.task_graph import Task, TaskGraph, TaskStatus
from laneswarm.dashboard.relay import DashboardRelay


def _make_graph():
    tasks = [
        Task(task_id="t1", title="Task 1", description="Desc 1", status=TaskStatus.PENDING),
        Task(task_id="t2", title="Task 2", description="Desc 2", dependencies=["t1"]),
    ]
    return TaskGraph(tasks)


class FakeWebSocket:
    """Minimal fake WebSocket for testing."""

    def __init__(self):
        self.sent = []
        self._messages = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            msg = await asyncio.wait_for(self._messages.get(), timeout=0.1)
            return msg
        except asyncio.TimeoutError:
            raise StopAsyncIteration

    async def push_message(self, msg):
        await self._messages.put(json.dumps(msg))


def test_relay_sends_snapshot_on_connect():
    """New client should receive a full snapshot."""
    event_bus = EventBus()
    graph = _make_graph()
    relay = DashboardRelay(event_bus, graph)

    async def run():
        loop = asyncio.get_running_loop()
        relay.start_listening(loop)

        ws = FakeWebSocket()
        # Run handle_client which sends snapshot then waits for messages
        await relay.handle_client(ws)

        assert len(ws.sent) >= 1
        snapshot = ws.sent[0]
        assert snapshot["type"] == "snapshot"
        assert len(snapshot["data"]["tasks"]) == 2

        relay.stop_listening()

    asyncio.run(run())


def test_relay_handles_get_task_detail():
    """Client requesting task detail should get a response."""
    event_bus = EventBus()
    graph = _make_graph()
    relay = DashboardRelay(event_bus, graph)

    async def run():
        loop = asyncio.get_running_loop()
        relay.start_listening(loop)

        ws = FakeWebSocket()
        # Queue a task detail request
        await ws.push_message({"type": "get_task_detail", "task_id": "t1"})

        await relay.handle_client(ws)

        # Should have snapshot + task_detail
        assert len(ws.sent) >= 2
        detail = ws.sent[1]
        assert detail["type"] == "task_detail"
        assert detail["data"]["task_id"] == "t1"
        assert detail["data"]["description"] == "Desc 1"

        relay.stop_listening()

    asyncio.run(run())


def test_relay_event_forwarding():
    """Events published to EventBus should be queued for broadcast."""
    event_bus = EventBus()
    relay = DashboardRelay(event_bus)

    async def run():
        loop = asyncio.get_running_loop()
        relay.start_listening(loop)

        # Publish an event
        event_bus.emit(EventType.TASK_STARTED, task_id="t1", model="claude")

        # Give the event loop a moment to process
        await asyncio.sleep(0.05)

        # The event should be in the queue
        assert not relay._queue.empty()
        msg = await relay._queue.get()
        assert msg["type"] == "event"
        assert msg["data"]["event_type"] == "task_started"
        assert msg["data"]["task_id"] == "t1"

        relay.stop_listening()

    asyncio.run(run())


def test_relay_set_task_graph():
    """set_task_graph should update the reference."""
    event_bus = EventBus()
    relay = DashboardRelay(event_bus)
    assert relay.task_graph is None

    graph = _make_graph()
    relay.set_task_graph(graph)
    assert relay.task_graph is graph
