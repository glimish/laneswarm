"""Tests for the event bus."""

from laneswarm.events import EventBus, EventType, SwarmEvent


def test_publish_subscribe():
    bus = EventBus()
    received = []
    bus.subscribe(lambda e: received.append(e))

    bus.emit(EventType.TASK_STARTED, task_id="001")

    assert len(received) == 1
    assert received[0].event_type == EventType.TASK_STARTED
    assert received[0].task_id == "001"


def test_multiple_subscribers():
    bus = EventBus()
    a, b = [], []
    bus.subscribe(lambda e: a.append(e))
    bus.subscribe(lambda e: b.append(e))

    bus.emit(EventType.RUN_STARTED)

    assert len(a) == 1
    assert len(b) == 1


def test_unsubscribe():
    bus = EventBus()
    received = []
    callback = lambda e: received.append(e)
    bus.subscribe(callback)
    bus.unsubscribe(callback)

    bus.emit(EventType.RUN_STARTED)

    assert len(received) == 0


def test_history():
    bus = EventBus()
    bus.emit(EventType.TASK_STARTED, task_id="001")
    bus.emit(EventType.TASK_COMPLETED, task_id="001")

    assert len(bus.history) == 2
    assert bus.history[0].event_type == EventType.TASK_STARTED
    assert bus.history[1].event_type == EventType.TASK_COMPLETED


def test_subscriber_error_doesnt_break():
    bus = EventBus()

    def bad_callback(e):
        raise RuntimeError("oops")

    received = []
    bus.subscribe(bad_callback)
    bus.subscribe(lambda e: received.append(e))

    bus.emit(EventType.RUN_STARTED)

    # Second subscriber should still receive the event
    assert len(received) == 1


def test_event_data():
    bus = EventBus()
    received = []
    bus.subscribe(lambda e: received.append(e))

    bus.emit(EventType.COST_UPDATE, task_id="001", tokens=5000, cost=0.05)

    assert received[0].data["tokens"] == 5000
    assert received[0].data["cost"] == 0.05


def test_event_str():
    event = SwarmEvent(
        event_type=EventType.TASK_COMPLETED,
        task_id="001",
        agent_id="coder-1",
        data={"tokens": 1234},
    )
    s = str(event)
    assert "task_completed" in s
    assert "001" in s
    assert "coder-1" in s
