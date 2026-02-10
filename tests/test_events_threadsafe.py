"""Thread-safety tests for EventBus."""

import threading
import time

from laneswarm.events import EventBus, EventType


def test_concurrent_publish():
    """Multiple threads publishing events simultaneously should not lose events."""
    bus = EventBus()
    received = []
    lock = threading.Lock()

    def on_event(event):
        with lock:
            received.append(event)

    bus.subscribe(on_event)

    events_per_thread = 100
    num_threads = 10

    def publish_events(thread_id):
        for i in range(events_per_thread):
            bus.emit(
                EventType.PROGRESS_UPDATE,
                task_id=f"t-{thread_id}-{i}",
                thread=thread_id,
                index=i,
            )

    threads = [
        threading.Thread(target=publish_events, args=(tid,))
        for tid in range(num_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected_count = events_per_thread * num_threads
    assert len(received) == expected_count
    assert len(bus.history) == expected_count


def test_concurrent_subscribe_unsubscribe():
    """Subscribing and unsubscribing while events are being published."""
    bus = EventBus()
    counts = {"a": 0, "b": 0}

    def callback_a(event):
        counts["a"] += 1

    def callback_b(event):
        counts["b"] += 1

    bus.subscribe(callback_a)

    # Thread 1: publishes events
    # Thread 2: subscribes callback_b midway, then unsubscribes callback_a
    def publisher():
        for _ in range(200):
            bus.emit(EventType.PROGRESS_UPDATE)
            time.sleep(0.001)

    def modifier():
        time.sleep(0.05)  # Let some events through first
        bus.subscribe(callback_b)
        time.sleep(0.05)
        bus.unsubscribe(callback_a)

    t1 = threading.Thread(target=publisher)
    t2 = threading.Thread(target=modifier)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # callback_a should have received some events (not all)
    # callback_b should have received some events (not all)
    assert counts["a"] > 0
    assert counts["b"] > 0
    assert counts["a"] + counts["b"] > 100  # Both got a decent share


def test_history_cap_under_concurrency():
    """History should be capped at _max_history even under concurrent writes."""
    bus = EventBus()
    bus._max_history = 100

    def publish_many(n):
        for _ in range(n):
            bus.emit(EventType.PROGRESS_UPDATE)

    threads = [
        threading.Thread(target=publish_many, args=(200,))
        for _ in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # History should never exceed max + some slack (due to batching)
    assert len(bus.history) <= bus._max_history


def test_subscriber_error_does_not_block():
    """A failing subscriber should not prevent other subscribers from receiving events."""
    bus = EventBus()
    results = []

    def bad_callback(event):
        raise ValueError("I'm broken!")

    def good_callback(event):
        results.append(event.event_type)

    bus.subscribe(bad_callback)
    bus.subscribe(good_callback)

    bus.emit(EventType.TASK_COMPLETED, task_id="t1")

    assert len(results) == 1
    assert results[0] == EventType.TASK_COMPLETED
