"""WebSocket relay: bridges the sync EventBus to async WebSocket clients.

The DashboardRelay subscribes to the EventBus and forwards every event
to all connected WebSocket clients as JSON.  New clients receive a full
state snapshot on connect, then a real-time event stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..events import EventBus, SwarmEvent
from ..task_graph import TaskGraph
from .serializers import serialize_event, serialize_snapshot, serialize_task_detail

logger = logging.getLogger(__name__)


class DashboardRelay:
    """Bridges the synchronous EventBus to async WebSocket clients.

    Usage:
        relay = DashboardRelay(event_bus, task_graph)
        # In an async context:
        await relay.start()
        # Or from a sync thread:
        relay.start_background()
    """

    def __init__(
        self,
        event_bus: EventBus,
        task_graph: TaskGraph | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.task_graph = task_graph
        self._clients: set = set()
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_task_graph(self, task_graph: TaskGraph) -> None:
        """Update the task graph reference (called when a run starts)."""
        self.task_graph = task_graph

    def _on_event(self, event: SwarmEvent) -> None:
        """EventBus callback — runs in worker threads.

        Puts the serialized event into the async queue using
        run_coroutine_threadsafe so the async broadcast picks it up.
        """
        msg = {"type": "event", "data": serialize_event(event)}
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._queue.put(msg), self._loop)

    def start_listening(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register as an EventBus subscriber and store the event loop."""
        self._loop = loop
        self.event_bus.subscribe(self._on_event)
        logger.debug("DashboardRelay subscribed to EventBus")

    def stop_listening(self) -> None:
        """Unsubscribe from EventBus."""
        self.event_bus.unsubscribe(self._on_event)
        self._loop = None

    async def handle_client(self, websocket) -> None:
        """Handle a single WebSocket client connection.

        Sends a snapshot on connect, then streams events.
        Also handles incoming requests (e.g. get_task_detail).
        """
        self._clients.add(websocket)
        client_id = id(websocket)
        logger.info("Dashboard client connected (%d total)", len(self._clients))

        try:
            # Send initial snapshot
            snapshot = serialize_snapshot(self.task_graph, self.event_bus)
            await websocket.send(json.dumps({"type": "snapshot", "data": snapshot}))

            # Handle incoming messages from this client
            async for raw_msg in websocket:
                try:
                    msg = json.loads(raw_msg)
                    await self._handle_client_message(websocket, msg)
                except json.JSONDecodeError:
                    logger.debug("Client %d sent invalid JSON", client_id)
        except Exception as e:
            logger.debug("Client %d disconnected: %s", client_id, e)
        finally:
            self._clients.discard(websocket)
            logger.info("Dashboard client disconnected (%d remaining)", len(self._clients))

    async def _handle_client_message(self, websocket, msg: dict) -> None:
        """Process a message from a WebSocket client."""
        msg_type = msg.get("type", "")

        if msg_type == "get_task_detail":
            task_id = msg.get("task_id", "")
            if self.task_graph and task_id in self.task_graph:
                task = self.task_graph.get_task(task_id)
                detail = serialize_task_detail(task)
                await websocket.send(json.dumps({"type": "task_detail", "data": detail}))

        elif msg_type == "get_snapshot":
            snapshot = serialize_snapshot(self.task_graph, self.event_bus)
            await websocket.send(json.dumps({"type": "snapshot", "data": snapshot}))

    async def broadcast_loop(self) -> None:
        """Continuously read from the queue and broadcast to all clients."""
        while True:
            msg = await self._queue.get()
            if not self._clients:
                continue
            raw = json.dumps(msg)
            # Broadcast to all connected clients, removing dead ones
            disconnected = set()
            for ws in self._clients:
                try:
                    await ws.send(raw)
                except Exception:
                    disconnected.add(ws)
            self._clients -= disconnected
