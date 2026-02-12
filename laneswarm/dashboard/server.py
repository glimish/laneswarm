"""Dashboard server: single-port HTTP + WebSocket.

Serves the static frontend (HTML/CSS/JS) over HTTP and provides
a WebSocket endpoint at /ws for real-time event streaming.

Uses the `websockets` library which supports both protocols on
a single port via the process_request hook.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any

from ..events import EventBus
from ..task_graph import TaskGraph
from .relay import DashboardRelay

logger = logging.getLogger(__name__)

# Directory containing the static web assets
WEB_DIR = Path(__file__).parent / "web"

# Content types for static files
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _serve_static(path: str) -> tuple[HTTPStatus, list[tuple[str, str]], bytes] | None:
    """Serve a static file from the web directory.

    Returns (status, headers, body) or None if the path should be
    handled as a WebSocket upgrade.
    """
    # Strip query string
    path = path.split("?")[0]

    # Default to index.html
    if path == "/" or path == "":
        path = "/index.html"

    # Security: prevent path traversal
    clean = Path(path.lstrip("/"))
    if ".." in clean.parts:
        return (HTTPStatus.FORBIDDEN, [], b"Forbidden")

    file_path = WEB_DIR / clean
    if not file_path.is_file():
        return (HTTPStatus.NOT_FOUND, [], b"Not Found")

    # Determine content type
    suffix = file_path.suffix.lower()
    content_type = (
        CONTENT_TYPES.get(suffix)
        or mimetypes.guess_type(str(file_path))[0]
        or "application/octet-stream"
    )

    body = file_path.read_bytes()
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-cache"),
    ]
    return (HTTPStatus.OK, headers, body)


class DashboardServer:
    """Combined HTTP + WebSocket server for the Laneswarm dashboard.

    Usage:
        server = DashboardServer(event_bus, task_graph)
        # Blocking:
        server.serve(host="0.0.0.0", port=8420)
        # Non-blocking (background thread):
        server.serve_background(host="0.0.0.0", port=8420)
    """

    def __init__(
        self,
        event_bus: EventBus,
        task_graph: TaskGraph | None = None,
    ) -> None:
        self.relay = DashboardRelay(event_bus, task_graph)
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_task_graph(self, task_graph: TaskGraph) -> None:
        """Update the task graph (called when a run starts)."""
        self.relay.set_task_graph(task_graph)

    def serve(self, host: str = "0.0.0.0", port: int = 8420) -> None:
        """Start the server (blocking)."""
        asyncio.run(self._run(host, port))

    def serve_background(
        self,
        host: str = "0.0.0.0",
        port: int = 8420,
    ) -> None:
        """Start the server in a background daemon thread.

        Blocks until the server is ready to accept connections (up to 5s).
        """
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_in_thread,
            args=(host, port),
            daemon=True,
            name="laneswarm-dashboard",
        )
        self._thread.start()
        # Wait for server to bind port before returning
        self._ready.wait(timeout=5)

    def _run_in_thread(self, host: str, port: int) -> None:
        """Entry point for the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run(host, port))
        except Exception as e:
            logger.error("Dashboard server error: %s", e)
        finally:
            loop.close()

    async def _run(self, host: str, port: int) -> None:
        """Async server main loop."""
        try:
            import websockets
            from websockets.asyncio.server import serve
        except ImportError:
            logger.error(
                "websockets library not installed. Install it with: pip install websockets>=12.0"
            )
            return

        loop = asyncio.get_running_loop()
        self._loop = loop
        self.relay.start_listening(loop)

        # Start the broadcast loop as a background task
        broadcast_task = asyncio.create_task(self.relay.broadcast_loop())

        async def handler(websocket) -> None:
            """Handle incoming WebSocket connections."""
            await self.relay.handle_client(websocket)

        async def process_request(connection, request):
            """Route HTTP requests to static files, let WS upgrades through."""
            # Check if this is a WebSocket upgrade request
            if request.path == "/ws":
                return None  # Let websockets handle the upgrade

            # Check for upgrade header (WS connections to non-/ws paths)
            for header_name, header_value in request.headers.raw_items():
                if header_name.lower() == "upgrade" and header_value.lower() == "websocket":
                    return None

            # Serve static files
            result = _serve_static(request.path)
            if result is not None:
                status, headers, body = result
                return websockets.http11.Response(
                    status.value,
                    status.phrase,
                    websockets.datastructures.Headers(headers),
                    body,
                )
            return None

        try:
            async with serve(
                handler,
                host,
                port,
                process_request=process_request,
            ) as server:
                self._server = server
                logger.info("Dashboard server running at http://%s:%d", host, port)
                # Signal readiness so serve_background() can return
                if hasattr(self, "_ready"):
                    self._ready.set()
                # Run forever
                await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            broadcast_task.cancel()
            self.relay.stop_listening()

    def stop(self) -> None:
        """Stop the server."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
