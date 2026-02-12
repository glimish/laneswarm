"""Tests for the SmokerAgent: app detection, server lifecycle, HTTP checks."""

import subprocess
import textwrap
from pathlib import Path

import pytest

from laneswarm.agents.smoker import (
    AppInfo,
    _build_ws_text_frame,
    _detect_app,
    _fetch_openapi_routes,
    _http_get,
    _http_request,
    _run_http_checks,
    _setup_env,
    _wait_for_port,
    _ws_functional_check,
    _ws_handshake,
)

# ---------------------------------------------------------------------------
# App detection tests
# ---------------------------------------------------------------------------


def test_detect_aiohttp(tmp_path: Path):
    """Detect aiohttp framework from imports."""
    (tmp_path / "server.py").write_text(
        textwrap.dedent("""\
        from aiohttp import web

        async def hello(request):
            return web.Response(text="Hello")

        app = web.Application()
        app.router.add_get('/', hello)
        app.router.add_get('/ws', hello)

        if __name__ == '__main__':
            web.run_app(app, port=8080)
    """)
    )
    (tmp_path / "requirements.txt").write_text("aiohttp\n")

    info = _detect_app(tmp_path)
    assert info.framework == "aiohttp"
    assert info.entry_point == "server.py"
    assert info.port == 8080
    assert "/" in info.routes
    assert any("ws" in ep for ep in (info.ws_endpoints + info.routes))
    assert info.requirements_file == "requirements.txt"
    assert info.is_python is True


def test_detect_flask(tmp_path: Path):
    """Detect Flask framework from imports."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""\
        from flask import Flask

        app = Flask(__name__)

        @app.route('/')
        def index():
            return "Hello"

        @app.route('/api/users')
        def users():
            return "[]"

        if __name__ == '__main__':
            app.run(port=5000)
    """)
    )

    info = _detect_app(tmp_path)
    assert info.framework == "flask"
    assert info.entry_point == "app.py"
    assert info.port == 5000
    assert "/" in info.routes
    assert "/api/users" in info.routes


def test_detect_fastapi(tmp_path: Path):
    """Detect FastAPI framework from imports."""
    (tmp_path / "main.py").write_text(
        textwrap.dedent("""\
        from fastapi import FastAPI

        app = FastAPI()

        @app.get('/api/health')
        def health():
            return {"status": "ok"}
    """)
    )

    info = _detect_app(tmp_path)
    assert info.framework == "fastapi"
    assert info.entry_point == "main.py"
    assert "/api/health" in info.routes


def test_detect_express(tmp_path: Path):
    """Detect Express (Node.js) framework."""
    (tmp_path / "server.js").write_text(
        textwrap.dedent("""\
        const express = require('express');
        const app = express();

        app.get('/api/data', (req, res) => {
            res.json({data: []});
        });

        app.listen(3000);
    """)
    )

    info = _detect_app(tmp_path)
    assert info.framework == "express"
    assert info.entry_point == "server.js"
    assert info.port == 3000
    assert info.is_python is False


def test_detect_static_site(tmp_path: Path):
    """Detect static HTML site (no server framework)."""
    (tmp_path / "index.html").write_text("<html><body>Hello</body></html>")
    (tmp_path / "style.css").write_text("body { color: red; }")

    info = _detect_app(tmp_path)
    assert info.framework == "static"
    assert info.entry_point is None


def test_detect_unknown(tmp_path: Path):
    """Unknown project with no recognizable framework."""
    (tmp_path / "README.md").write_text("# My project")

    info = _detect_app(tmp_path)
    assert info.framework == "unknown"
    assert info.entry_point is None


def test_detect_entry_point_priority(tmp_path: Path):
    """Entry point search follows priority order: run.py > main.py > app.py."""
    (tmp_path / "app.py").write_text("from flask import Flask\n")
    (tmp_path / "run.py").write_text("from app import app\napp.run()\n")

    info = _detect_app(tmp_path)
    assert info.entry_point == "run.py"  # run.py takes priority


def test_detect_dunder_main(tmp_path: Path):
    """Detect __main__.py as entry point when no top-level script exists."""
    pkg = tmp_path / "myapp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "__main__.py").write_text("from flask import Flask\nprint('hello')\n")

    info = _detect_app(tmp_path)
    assert info.entry_point is not None
    assert "__main__.py" in info.entry_point


def test_detect_skips_venv(tmp_path: Path):
    """Should skip .venv/venv directories during scanning."""
    venv_dir = tmp_path / "venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "flask_internal.py").write_text("from flask import Flask\n")

    # Only a README at top level — should be unknown
    (tmp_path / "README.md").write_text("hello")

    info = _detect_app(tmp_path)
    assert info.framework == "unknown"


def test_detect_requirements_txt(tmp_path: Path):
    """Detect requirements.txt."""
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\n")
    (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n")

    info = _detect_app(tmp_path)
    assert info.requirements_file == "requirements.txt"


def test_detect_pyproject_toml(tmp_path: Path):
    """Fall back to pyproject.toml when no requirements.txt."""
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'myapp'\n")

    info = _detect_app(tmp_path)
    assert info.requirements_file == "pyproject.toml"


# ---------------------------------------------------------------------------
# Route / WS / port detection
# ---------------------------------------------------------------------------


def test_detect_websocket_endpoints(tmp_path: Path):
    """Detect WebSocket endpoints from source patterns."""
    (tmp_path / "server.py").write_text(
        textwrap.dedent("""\
        from aiohttp import web

        app = web.Application()
        app.router.add_get('/ws', websocket_handler)
        app.router.add_get('/chat/ws', chat_ws_handler)

        if __name__ == '__main__':
            web.run_app(app, port=8080)
    """)
    )

    info = _detect_app(tmp_path)
    assert "/ws" in info.ws_endpoints or "/ws" in info.routes
    assert len(info.ws_endpoints) >= 1


def test_detect_static_dirs(tmp_path: Path):
    """Detect static file directory configuration."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""\
        from aiohttp import web
        app = web.Application()
        app.router.add_static('/static', 'public')
    """)
    )

    info = _detect_app(tmp_path)
    assert "/static" in info.static_dirs


# ---------------------------------------------------------------------------
# HTTP check helpers
# ---------------------------------------------------------------------------


def test_http_get_unreachable():
    """HTTP GET to unreachable port returns failure."""
    result = _http_get("http://127.0.0.1:19999/", "test")
    assert result["passed"] is False
    assert "failed" in result["detail"].lower() or "refused" in result["detail"].lower()


def test_wait_for_port_not_listening():
    """_wait_for_port returns False for ports nobody is listening on."""
    # Use a high port that's very unlikely to be in use
    assert _wait_for_port(59123, timeout=1.0) is False


# ---------------------------------------------------------------------------
# WebSocket handshake helper
# ---------------------------------------------------------------------------


def test_ws_handshake_unreachable():
    """WS handshake to unreachable port returns failure."""
    result = _ws_handshake(19999, "/ws")
    assert result["passed"] is False
    assert "failed" in result["detail"].lower() or "refused" in result["detail"].lower()


# ---------------------------------------------------------------------------
# AppInfo dataclass
# ---------------------------------------------------------------------------


def test_app_info_defaults():
    """AppInfo has sensible defaults."""
    info = AppInfo()
    assert info.framework == "unknown"
    assert info.entry_point is None
    assert info.port == 8080
    assert info.routes == []
    assert info.ws_endpoints == []
    assert info.is_python is True


def test_app_info_custom():
    """AppInfo can be customized."""
    info = AppInfo(
        framework="flask",
        entry_point="app.py",
        port=5000,
        routes=["/", "/api"],
        ws_endpoints=["/ws"],
    )
    assert info.framework == "flask"
    assert info.port == 5000
    assert len(info.routes) == 2


# ---------------------------------------------------------------------------
# Regression tests for first live run bugs
# ---------------------------------------------------------------------------


def test_detect_websockets(tmp_path: Path):
    """Detect websockets library (standalone WS server)."""
    (tmp_path / "server.py").write_text(
        textwrap.dedent("""\
        import websockets
        from websockets.http11 import Response

        async def handler(websocket):
            pass

        async def start_server():
            async with websockets.serve(handler, '0.0.0.0', 8765):
                pass
    """)
    )

    info = _detect_app(tmp_path)
    assert info.framework == "websockets"
    assert info.entry_point == "server.py"
    assert info.is_python is True


def test_detect_websockets_from_import(tmp_path: Path):
    """Detect websockets from 'from websockets' import style."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""\
        from websockets.server import serve

        async def handler(ws):
            pass
    """)
    )

    info = _detect_app(tmp_path)
    assert info.framework == "websockets"


def test_detect_port_with_type_annotation(tmp_path: Path):
    """Port detection handles Python type-annotated class attrs: PORT: int = 8765."""
    (tmp_path / "config.py").write_text(
        textwrap.dedent("""\
        class Config:
            HOST: str = '0.0.0.0'
            PORT: int = 8765
    """)
    )
    (tmp_path / "server.py").write_text(
        textwrap.dedent("""\
        import websockets
        from config import Config
    """)
    )

    info = _detect_app(tmp_path)
    assert info.port == 8765


def test_detect_port_simple_assignment(tmp_path: Path):
    """Port detection still works for simple assignments: port = 3000."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""\
        from flask import Flask
        app = Flask(__name__)
        port = 3000
        app.run(port=port)
    """)
    )

    info = _detect_app(tmp_path)
    assert info.port == 3000


def test_unknown_framework_with_entry_point_does_not_skip(tmp_path: Path):
    """Even with unknown framework, if entry point exists, detection returns it.

    The skip condition in _run_smoke_test should only skip when entry_point is None.
    """
    (tmp_path / "run.py").write_text(
        textwrap.dedent("""\
        # Some custom server framework
        import my_custom_server
        my_custom_server.start(port=9000)
    """)
    )

    info = _detect_app(tmp_path)
    assert info.framework == "unknown"
    assert info.entry_point == "run.py"  # entry point found despite unknown framework
    assert info.port == 9000


# ---------------------------------------------------------------------------
# Regression tests for second live run: false positive prevention
# ---------------------------------------------------------------------------


def test_routes_ignore_dict_get(tmp_path: Path):
    """dict.get('key') should NOT be detected as an HTTP route.

    Previously, `data.get('username')` matched the route pattern
    `\\.(?:get|post|put|delete)\\(` and returned 'username' as a route.
    """
    (tmp_path / "server.py").write_text(
        textwrap.dedent("""\
        import websockets

        data = {}
        username = data.get('username', '')
        room = data.get('room', '')
        msg_type = msg.get('type')
        content = data.get('content', '')
        before_id = data.get('before_id', None)
        handler = MESSAGE_HANDLERS.get(msg_type)
    """)
    )

    info = _detect_app(tmp_path)
    # None of these dict.get() calls should be detected as routes
    assert "username" not in info.routes
    assert "room" not in info.routes
    assert "type" not in info.routes
    assert "content" not in info.routes
    assert "before_id" not in info.routes
    # Only routes starting with / should be in the list
    for route in info.routes:
        assert route.startswith("/"), f"Route '{route}' doesn't start with /"


def test_routes_require_slash_prefix(tmp_path: Path):
    """Only paths starting with / should be detected as routes."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""\
        from flask import Flask
        app = Flask(__name__)

        @app.route('/')
        def index():
            pass

        @app.route('/api/users')
        def users():
            pass
    """)
    )

    info = _detect_app(tmp_path)
    assert "/" in info.routes
    assert "/api/users" in info.routes
    # All routes must start with /
    for route in info.routes:
        assert route.startswith("/")


def test_ws_endpoints_ignore_type_annotations(tmp_path: Path):
    """WebSocketServerProtocol type annotations should NOT be WS endpoints.

    Previously, `websockets.WebSocketServerProtocol, dict` was matched
    by the greedy `WebSocket.*?['"]` pattern.
    """
    (tmp_path / "manager.py").write_text(
        textwrap.dedent("""\
        import websockets

        class Manager:
            active: dict[websockets.WebSocketServerProtocol, dict] = {}
            rooms: dict[str, set[websockets.WebSocketServerProtocol]] = {}

            async def send(self, ws: websockets.WebSocketServerProtocol):
                if ws.open:
                    pass
    """)
    )

    info = _detect_app(tmp_path)
    # No WS endpoints should be detected from type annotations
    assert len(info.ws_endpoints) == 0


def test_ws_endpoints_detect_new_websocket_js(tmp_path: Path):
    """JS `new WebSocket('ws://host/ws')` should detect /ws endpoint."""
    (tmp_path / "app.js").write_text(
        textwrap.dedent("""\
        const protocol = 'ws:';
        const host = window.location.host;
        ws = new WebSocket(protocol + '//' + host + '/ws');

        ws.onmessage = function(event) {
            console.log(event.data);
        };
    """)
    )
    # Need a Python file so it's not detected as static-only
    (tmp_path / "server.py").write_text("import websockets\n")

    info = _detect_app(tmp_path)
    assert "/ws" in info.ws_endpoints


def test_ws_endpoints_detect_path_check(tmp_path: Path):
    """Python `path == '/ws'` should detect /ws as a WS endpoint."""
    (tmp_path / "server.py").write_text(
        textwrap.dedent("""\
        import websockets

        async def process_request(path, headers):
            if path == '/ws':
                return None
            return serve_static(path)
    """)
    )

    info = _detect_app(tmp_path)
    assert "/ws" in (info.ws_endpoints + info.routes)


def test_routes_detect_path_equality(tmp_path: Path):
    """websockets-style `path == '/route'` checks should be detected."""
    (tmp_path / "server.py").write_text(
        textwrap.dedent("""\
        import websockets

        async def process_request(path, headers):
            if path == '/ws':
                return None
            if path == '/health':
                return (200, {}, b'ok')
    """)
    )

    info = _detect_app(tmp_path)
    assert "/ws" in info.routes
    assert "/health" in info.routes


# ---------------------------------------------------------------------------
# OpenAPI route fetching
# ---------------------------------------------------------------------------


def test_fetch_openapi_routes_success(monkeypatch):
    """_fetch_openapi_routes parses a valid OpenAPI spec."""
    import io
    import json
    import urllib.request

    spec = {
        "openapi": "3.0.0",
        "paths": {
            "/api/seed": {"post": {"summary": "Seed"}},
            "/api/export/csv": {"get": {"summary": "Export CSV"}},
            "/api/health": {"get": {"summary": "Health"}, "head": {"summary": "Health"}},
        },
    }
    body = json.dumps(spec).encode("utf-8")

    def fake_urlopen(req, timeout=None):
        resp = io.BytesIO(body)
        resp.status = 200
        resp.read = resp.read
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = _fetch_openapi_routes(8080)

    assert result is not None
    assert "/api/seed" in result
    assert result["/api/seed"] == ["POST"]
    assert "/api/export/csv" in result
    assert result["/api/export/csv"] == ["GET"]
    assert sorted(result["/api/health"]) == ["GET", "HEAD"]


def test_fetch_openapi_routes_failure(monkeypatch):
    """_fetch_openapi_routes returns None on HTTP error."""
    import urllib.error
    import urllib.request

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            404,
            "Not Found",
            {},
            None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = _fetch_openapi_routes(8080)
    assert result is None


def test_fetch_openapi_routes_bad_json(monkeypatch):
    """_fetch_openapi_routes returns None on invalid JSON."""
    import io
    import urllib.request

    def fake_urlopen(req, timeout=None):
        resp = io.BytesIO(b"not json")
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = _fetch_openapi_routes(8080)
    assert result is None


# ---------------------------------------------------------------------------
# HTTP method-aware checks
# ---------------------------------------------------------------------------


def test_http_request_unreachable():
    """_http_request to unreachable port returns failure."""
    result = _http_request("http://127.0.0.1:19999/api/seed", "POST", "test")
    assert result["passed"] is False
    assert "POST" in result["detail"]


def test_run_http_checks_uses_post_for_post_only_route():
    """POST-only routes should be tested with POST, not GET."""
    app_info = AppInfo(
        framework="fastapi",
        port=19999,  # unreachable — we just check the method dispatch
        routes=["/", "/api/seed"],
        route_methods={
            "/": ["GET"],
            "/api/seed": ["POST"],
        },
    )
    checks = _run_http_checks(19999, app_info)

    # Find the check for /api/seed
    seed_check = [c for c in checks if c["name"] == "http_route:/api/seed"]
    assert len(seed_check) == 1
    # It should have tried POST (visible in the detail even though it fails)
    assert "POST" in seed_check[0]["detail"]


def test_run_http_checks_falls_back_to_get_without_method_info():
    """Routes without method info fall back to GET."""
    app_info = AppInfo(
        framework="fastapi",
        port=19999,
        routes=["/", "/api/data"],
        route_methods={},  # no method info
    )
    checks = _run_http_checks(19999, app_info)

    data_check = [c for c in checks if c["name"] == "http_route:/api/data"]
    assert len(data_check) == 1
    # Should NOT contain "POST" — it fell back to GET
    assert "POST" not in data_check[0]["detail"]


def test_app_info_route_methods_default():
    """AppInfo.route_methods defaults to empty dict."""
    info = AppInfo()
    assert info.route_methods == {}


# ---------------------------------------------------------------------------
# Regression tests: smoker round-2 bug fixes
# ---------------------------------------------------------------------------


def test_setup_env_unexpected_exception_cleans_up(tmp_path, monkeypatch):
    """_setup_env raises on unexpected exceptions and cleans up venv."""
    (tmp_path / "requirements.txt").write_text("some-pkg\n")
    app_info = AppInfo(requirements_file="requirements.txt")

    call_count = {"n": 0}
    original_run = subprocess.run

    def patched_run(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: venv creation — let it succeed
            return original_run(*args, **kwargs)
        # Second call: pip install — raise unexpected error
        raise OSError("disk full")

    monkeypatch.setattr(subprocess, "run", patched_run)

    with pytest.raises(RuntimeError, match="pip install failed.*disk full"):
        _setup_env(tmp_path, app_info)


def test_http_get_accepts_2xx_status(monkeypatch):
    """_http_get treats all 2xx status codes as success."""
    import urllib.request

    class FakeResp:
        status = 201

        def read(self, n=None):
            return b"created"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    result = _http_get("http://127.0.0.1:9999/resource", "test_201")
    assert result["passed"] is True
    assert "201" in result["detail"]


def test_http_get_accepts_204_no_content(monkeypatch):
    """_http_get treats 204 No Content as success."""
    import urllib.request

    class FakeResp:
        status = 204

        def read(self, n=None):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    result = _http_get("http://127.0.0.1:9999/empty", "test_204")
    assert result["passed"] is True
    assert "204" in result["detail"]


def test_detect_fastapi_mount_static(tmp_path: Path):
    """Detect static mount path from app.mount('/static', StaticFiles(...))."""
    (tmp_path / "main.py").write_text(
        textwrap.dedent("""\
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles

        app = FastAPI()
        static_dir = "public"
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    """)
    )

    info = _detect_app(tmp_path)
    assert "/static" in info.static_dirs


def test_detect_routes_with_api_router_prefix(tmp_path: Path):
    """Routes from APIRouter files get prefix prepended."""
    (tmp_path / "main.py").write_text(
        textwrap.dedent("""\
        from fastapi import FastAPI
        app = FastAPI()
    """)
    )
    routers = tmp_path / "routers"
    routers.mkdir()
    (routers / "__init__.py").write_text("")
    (routers / "tags.py").write_text(
        textwrap.dedent("""\
        from fastapi import APIRouter
        router = APIRouter(prefix="/api/tags", tags=["tags"])

        @router.get("/summary")
        def summary():
            pass

        @router.get("/{tag_id}")
        def get_tag(tag_id: int):
            pass
    """)
    )

    info = _detect_app(tmp_path)
    assert "/api/tags/summary" in info.routes


def test_detect_routes_no_double_prefix(tmp_path: Path):
    """Routes already containing prefix are not double-prefixed."""
    (tmp_path / "main.py").write_text(
        textwrap.dedent("""\
        from fastapi import FastAPI
        app = FastAPI()

        @app.get("/api/health")
        def health():
            pass
    """)
    )

    info = _detect_app(tmp_path)
    assert "/api/health" in info.routes
    # Should NOT have /api/health/api/health
    assert not any(r.count("/api/health") > 1 for r in info.routes)


# ---------------------------------------------------------------------------
# Static dir normalization tests
# ---------------------------------------------------------------------------


def test_static_dir_normalization_leading_slash(tmp_path: Path):
    """Static dirs detected without leading / get normalized."""
    (tmp_path / "app.py").write_text(
        textwrap.dedent("""\
        from flask import Flask
        app = Flask(__name__, static_folder='public')
    """)
    )

    info = _detect_app(tmp_path)
    # Should be normalized to /public, not bare "public"
    assert "/public" in info.static_dirs
    assert "public" not in info.static_dirs


def test_static_url_construction_with_slash(tmp_path: Path):
    """Static URL is well-formed when prefix has leading /."""
    info = AppInfo(
        framework="fastapi",
        static_dirs=["/static"],
        routes=["/"],
        port=9999,
    )
    # _run_http_checks will try to connect — just verify
    # it doesn't produce a malformed URL
    checks = _run_http_checks(9999, info)
    # The check will fail (no server) but the URL should be correct
    static_checks = [c for c in checks if "http_static" in c["name"]]
    assert len(static_checks) == 1
    # Should NOT contain "9999static" (the old bug)
    assert "9999static" not in static_checks[0]["detail"]


# ---------------------------------------------------------------------------
# HTTP 422 handling tests
# ---------------------------------------------------------------------------


def test_http_get_422_passes(monkeypatch):
    """_http_get treats 422 as pass (route exists, missing params)."""
    import urllib.error
    import urllib.request

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            422,
            "Unprocessable Entity",
            {},
            None,
        )

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        fake_urlopen,
    )
    result = _http_get("http://127.0.0.1:9999/api/stats", "test_422")
    assert result["passed"] is True
    assert "422" in result["detail"]
    assert "validation" in result["detail"].lower()


def test_http_get_403_still_fails(monkeypatch):
    """_http_get still treats 403 as failure."""
    import urllib.error
    import urllib.request

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url,
            403,
            "Forbidden",
            {},
            None,
        )

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        fake_urlopen,
    )
    result = _http_get("http://127.0.0.1:9999/secret", "test_403")
    assert result["passed"] is False
    assert "403" in result["detail"]


# ---------------------------------------------------------------------------
# WebSocket functional testing
# ---------------------------------------------------------------------------


def test_build_ws_text_frame_structure():
    """_build_ws_text_frame produces valid masked RFC 6455 frame."""
    payload = b'{"type":"ping"}'
    frame = _build_ws_text_frame(payload)

    # First byte: FIN=1, opcode=0x01 (text) → 0x81
    assert frame[0] == 0x81
    # Second byte: mask=1, length for payload <= 125
    assert frame[1] & 0x80 == 0x80  # mask bit set
    length = frame[1] & 0x7F
    assert length == len(payload)
    # Bytes 2-5: masking key
    mask = frame[2:6]
    assert len(mask) == 4
    # Remaining: masked payload — unmask and verify
    masked_payload = frame[6:]
    assert len(masked_payload) == len(payload)
    unmasked = bytes(b ^ mask[i % 4] for i, b in enumerate(masked_payload))
    assert unmasked == payload


def test_ws_functional_check_connection_refused():
    """_ws_functional_check fails gracefully on unreachable port."""
    # Port 1 should never be listening
    result = _ws_functional_check(1, "/ws")
    assert result["passed"] is False
    assert result["name"] == "ws_functional:/ws"
