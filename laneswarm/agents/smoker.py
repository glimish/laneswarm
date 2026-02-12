"""Smoke-test agent: runtime validation of assembled projects.

After all tasks complete and integration validation passes, the smoker:
1. Detects the app type (Flask, FastAPI, aiohttp, Express, static)
2. Finds the entry point and port
3. Installs dependencies in an isolated venv
4. Starts the server as a subprocess
5. Runs HTTP/WebSocket smoke tests
6. Calls the LLM for diagnosis if tests fail
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..events import EventType
from ..prompts import SMOKER_DIAGNOSIS_PROMPT
from ..providers import Message
from ..task_graph import TaskGraph
from .base import BaseAgent

logger = logging.getLogger(__name__)

# Timeouts
SERVER_STARTUP_TIMEOUT = 20  # seconds to wait for server to bind port
HTTP_REQUEST_TIMEOUT = 5  # seconds per HTTP request
TOTAL_SMOKE_TIMEOUT = 90  # overall smoke test budget
INSTALL_TIMEOUT = 120  # pip install timeout


@dataclass
class AppInfo:
    """Detected application metadata."""

    framework: str = "unknown"  # aiohttp, flask, fastapi, django, express, static
    entry_point: str | None = None  # relative path to the main script
    port: int = 8080  # detected or default port
    routes: list[str] = field(default_factory=list)  # discovered HTTP routes
    route_methods: dict[str, list[str]] = field(default_factory=dict)  # path -> [methods]
    ws_endpoints: list[str] = field(default_factory=list)  # discovered WS endpoints
    static_dirs: list[str] = field(default_factory=list)  # static file directories
    requirements_file: str | None = None  # requirements.txt or similar
    is_python: bool = True  # vs Node.js


# Framework detection patterns
_FRAMEWORK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"from\s+aiohttp\s+import\s+web"), "aiohttp"),
    (re.compile(r"from\s+flask\s+import"), "flask"),
    (re.compile(r"from\s+fastapi\s+import"), "fastapi"),
    (re.compile(r"from\s+django\b"), "django"),
    (re.compile(r"(?:require|import)\s*\(?['\"]\s*express"), "express"),
    (re.compile(r"import\s+websockets"), "websockets"),
    (re.compile(r"from\s+websockets\b"), "websockets"),
]

# Entry point candidates (searched in order)
_PYTHON_ENTRY_POINTS = [
    "run.py",
    "main.py",
    "app.py",
    "server.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
]
_NODE_ENTRY_POINTS = ["index.js", "server.js", "app.js"]

# Port detection patterns
_PORT_PATTERNS = [
    # Handles both `port = 8080` and `port: int = 8080` (type annotations)
    re.compile(r"port\s*(?::\s*\w+\s*)?=\s*(\d{4,5})", re.IGNORECASE),
    re.compile(r"PORT\s*(?::\s*\w+\s*)?=\s*(\d{4,5})"),
    re.compile(r"\.listen\(\s*(\d{4,5})"),
    re.compile(r"host\s*=.*?port\s*=\s*(\d{4,5})"),
    # websockets.serve(..., host, port) — keyword arg
    re.compile(r"\.serve\([^)]*?,\s*['\"][^'\"]+['\"],\s*(\d{4,5})"),
]

# Route detection patterns
# IMPORTANT: All capture groups require `/` as first char to prevent
# false positives from dict.get('key') calls.
_ROUTE_PATTERNS = [
    # Flask/FastAPI: @app.route('/path'), @app.get('/path')
    re.compile(r"@\w+\.(?:route|get|post|put|delete)\(\s*['\"](/[^'\"]*)['\"]"),
    # aiohttp: app.router.add_get('/path', handler)
    re.compile(r"add_(?:get|post|put|delete|route)\(\s*['\"](/[^'\"]*)['\"]"),
    # Express: app.get('/path', ...) — require app./router. prefix
    re.compile(r"(?:app|router)\.(?:get|post|put|delete)\(\s*['\"](/[^'\"]*)['\"]"),
    # websockets-style: path == '/ws' checks in process_request
    re.compile(r"path\s*==\s*['\"](/[^'\"]+)['\"]"),
]

# WebSocket endpoint patterns
# All capture groups require `/` prefix to prevent false positives from
# type annotations (WebSocketServerProtocol) and constants.
_WS_PATTERNS = [
    # aiohttp: add_get('/ws', handler)
    re.compile(r"add_get\(\s*['\"](/[^'\"]*ws[^'\"]*)['\"]", re.IGNORECASE),
    # Flask/FastAPI: @app.websocket('/ws')
    re.compile(r"@\w+\.websocket\(\s*['\"](/[^'\"]+)['\"]"),
    # JS: new WebSocket('ws://host/endpoint') — direct URL string
    re.compile(r"new\s+WebSocket\(\s*['\"`]wss?://[^'\"]*?(/[^'\"]+)['\"`]"),
    # JS: new WebSocket(... + '/ws') — concatenation or template literal
    re.compile(r"new\s+WebSocket\(.*?['\"` ](/\w[^'\")]*)['\"`\)]"),
    # Python: path == '/ws' style routing
    re.compile(r"path\s*==\s*['\"](/[^'\"]*ws[^'\"]*)['\"]", re.IGNORECASE),
]

# Static directory patterns
_STATIC_PATTERNS = [
    re.compile(r"add_static\(\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"static_folder\s*=\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"StaticFiles.*?directory\s*=\s*['\"]([^'\"]+)['\"]"),
    # FastAPI: app.mount("/static", StaticFiles(...))
    re.compile(r"\.mount\(\s*['\"]([^'\"]+)['\"].*?StaticFiles"),
]

# APIRouter prefix detection (FastAPI)
_ROUTER_PREFIX_PATTERN = re.compile(r"APIRouter\(\s*prefix\s*=\s*['\"]([^'\"]+)['\"]")

# Directories to skip during scanning
_SKIP_DIRS = {
    ".flanes",
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".pytest_cache",
    ".env",
    ".tox",
}


class SmokerAgent(BaseAgent):
    """Runtime smoke-test agent for assembled projects."""

    def smoke_test(
        self,
        project_path: Path,
        task_graph: TaskGraph,
    ) -> dict:
        """Run smoke tests on the assembled project.

        Returns::

            {
                "passed": bool,
                "app_type": str,
                "entry_point": str | None,
                "checks": [{"name": str, "passed": bool, "detail": str}],
                "diagnosis": str | None,
                "summary": str,
            }
        """
        self.emit(EventType.SMOKE_TEST_STARTED)
        smoke_start = time.time()

        try:
            result = self._run_smoke_test(project_path, task_graph)
        except Exception as e:
            logger.warning("Smoke test crashed: %s", e, exc_info=True)
            result = {
                "passed": False,
                "app_type": "unknown",
                "entry_point": None,
                "checks": [{"name": "smoke_test", "passed": False, "detail": str(e)}],
                "diagnosis": None,
                "summary": f"Smoke test crashed: {e}",
            }

        # Enforce overall smoke test budget
        elapsed = time.time() - smoke_start
        if elapsed > TOTAL_SMOKE_TIMEOUT:
            logger.warning(
                "Smoke test exceeded budget: %.1fs > %ds",
                elapsed,
                TOTAL_SMOKE_TIMEOUT,
            )
            if result.get("passed"):
                result["passed"] = False
                result["checks"].append(
                    {
                        "name": "timeout_budget",
                        "passed": False,
                        "detail": (
                            f"Smoke test exceeded {TOTAL_SMOKE_TIMEOUT}s budget "
                            f"(took {elapsed:.1f}s)"
                        ),
                    }
                )
                result["summary"] = (
                    f"Smoke test budget exceeded ({elapsed:.1f}s > {TOTAL_SMOKE_TIMEOUT}s)"
                )

        elapsed = time.time() - smoke_start
        result["elapsed_seconds"] = round(elapsed, 1)

        if result["passed"]:
            self.emit(
                EventType.SMOKE_TEST_PASSED,
                summary=result["summary"],
                elapsed_seconds=elapsed,
            )
        else:
            self.emit(
                EventType.SMOKE_TEST_FAILED,
                summary=result["summary"],
                elapsed_seconds=elapsed,
            )

        return result

    def _run_smoke_test(
        self,
        project_path: Path,
        task_graph: TaskGraph,
    ) -> dict:
        """Inner smoke test implementation."""
        # Step 1: Detect app type
        app_info = _detect_app(project_path)
        logger.info(
            "Detected app: framework=%s entry=%s port=%d routes=%d ws=%d",
            app_info.framework,
            app_info.entry_point,
            app_info.port,
            len(app_info.routes),
            len(app_info.ws_endpoints),
        )

        if app_info.entry_point is None:
            return {
                "passed": True,  # No entry point — nothing to run
                "app_type": app_info.framework,
                "entry_point": None,
                "checks": [
                    {
                        "name": "app_detection",
                        "passed": True,
                        "detail": "No runnable entry point detected; skipping smoke tests",
                    }
                ],
                "diagnosis": None,
                "summary": "No entry point detected — smoke tests skipped.",
            }

        checks: list[dict] = []
        venv_path: Path | None = None
        server_proc: subprocess.Popen | None = None

        try:
            # Step 2: Set up isolated environment
            venv_path = _setup_env(project_path, app_info)
            if venv_path:
                checks.append(
                    {
                        "name": "env_setup",
                        "passed": True,
                        "detail": "Dependencies installed in isolated venv",
                    }
                )

            # Step 3: Start server
            server_proc = _start_server(project_path, app_info, venv_path)
            if server_proc is None:
                checks.append(
                    {
                        "name": "server_start",
                        "passed": False,
                        "detail": "Failed to start server subprocess",
                    }
                )
                return self._build_result(app_info, checks, project_path)

            # Step 4: Wait for port
            port_ready = _wait_for_port(app_info.port, SERVER_STARTUP_TIMEOUT)
            if not port_ready:
                # Capture server stderr for diagnosis
                stderr_text = ""
                try:
                    if server_proc.poll() is None:
                        # Process still running — kill to read stderr
                        server_proc.terminate()
                        try:
                            server_proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            server_proc.kill()
                    _, stderr_bytes = server_proc.communicate(
                        timeout=2,
                    )
                    stderr_text = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
                except Exception:
                    pass
                checks.append(
                    {
                        "name": "server_start",
                        "passed": False,
                        "detail": (
                            f"Server did not bind to port "
                            f"{app_info.port} within "
                            f"{SERVER_STARTUP_TIMEOUT}s. "
                            f"stderr: {stderr_text[:500]}"
                        ),
                    }
                )
                return self._build_result(app_info, checks, project_path)

            checks.append(
                {
                    "name": "server_start",
                    "passed": True,
                    "detail": f"Server started and listening on port {app_info.port}",
                }
            )

            # Step 4b: Fetch OpenAPI routes for FastAPI apps
            if app_info.framework == "fastapi":
                openapi_routes = _fetch_openapi_routes(app_info.port)
                if openapi_routes:
                    logger.info(
                        "OpenAPI spec: %d routes discovered",
                        len(openapi_routes),
                    )
                    app_info.routes = list(openapi_routes.keys())
                    app_info.route_methods = openapi_routes
                else:
                    logger.debug("OpenAPI fetch failed; using regex-detected routes")

            # Step 5: HTTP smoke tests
            http_checks = _run_http_checks(app_info.port, app_info)
            checks.extend(http_checks)

            # Step 6: WebSocket smoke tests
            if app_info.ws_endpoints:
                ws_checks = _run_ws_checks(app_info.port, app_info)
                checks.extend(ws_checks)

        finally:
            # Always clean up — capture stderr before killing for diagnosis
            server_stderr = ""
            if server_proc is not None:
                # Try to read stderr before killing (non-blocking)
                try:
                    if server_proc.stderr and server_proc.poll() is not None:
                        stderr_bytes = server_proc.stderr.read()
                        server_stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")[
                            :2000
                        ]
                except Exception:
                    pass
                _kill_server(server_proc)
                # If we didn't capture stderr above, try after kill
                if not server_stderr:
                    try:
                        _, stderr_bytes = server_proc.communicate(timeout=2)
                        server_stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")[
                            :2000
                        ]
                    except Exception:
                        pass
            if venv_path is not None:
                _cleanup_venv(venv_path)

        # Attach server stderr to failed checks for better diagnosis
        if server_stderr:
            for check in checks:
                if not check["passed"]:
                    check["detail"] += f" | server stderr: {server_stderr[:300]}"

        return self._build_result(app_info, checks, project_path)

    def _build_result(
        self,
        app_info: AppInfo,
        checks: list[dict],
        project_path: Path,
    ) -> dict:
        """Build the smoke test result dict, calling LLM for diagnosis if needed."""
        passed = all(c["passed"] for c in checks)
        failed_checks = [c for c in checks if not c["passed"]]

        diagnosis: str | None = None
        if failed_checks:
            diagnosis = self._diagnose(app_info, checks, project_path)

        # Summary
        total = len(checks)
        passed_count = sum(1 for c in checks if c["passed"])
        failed_count = total - passed_count

        if passed:
            summary = f"All {total} smoke checks passed ({app_info.framework} app)."
        else:
            summary = (
                f"{failed_count}/{total} smoke checks failed "
                f"({app_info.framework} app). "
                + "; ".join(c["name"] + ": " + c["detail"][:80] for c in failed_checks)
            )

        return {
            "passed": passed,
            "app_type": app_info.framework,
            "entry_point": app_info.entry_point,
            "checks": checks,
            "diagnosis": diagnosis,
            "summary": summary[:500],
        }

    def _diagnose(
        self,
        app_info: AppInfo,
        checks: list[dict],
        project_path: Path,
    ) -> str | None:
        """Call LLM to diagnose smoke test failures."""
        failed = [c for c in checks if not c["passed"]]
        if not failed:
            return None

        checks_text = "\n".join(
            f"- {c['name']}: {'PASS' if c['passed'] else 'FAIL'} — {c['detail']}" for c in checks
        )

        # Read the entry point file for context
        entry_content = ""
        if app_info.entry_point:
            ep = project_path / app_info.entry_point
            if ep.exists():
                try:
                    content = ep.read_text(errors="replace")
                    if len(content) > 5000:
                        content = content[:5000] + "\n... (truncated)"
                    entry_content = (
                        f"\n## Entry Point ({app_info.entry_point})\n```\n{content}\n```"
                    )
                except Exception:
                    pass

        prompt = (
            f"## App Info\n"
            f"Framework: {app_info.framework}\n"
            f"Entry point: {app_info.entry_point}\n"
            f"Port: {app_info.port}\n"
            f"Routes: {app_info.routes}\n"
            f"WS endpoints: {app_info.ws_endpoints}\n"
            f"\n## Smoke Test Results\n{checks_text}\n"
            f"{entry_content}\n"
            f"\nDiagnose the failures and suggest fixes."
        )

        try:
            response = self.call_llm_sync(
                messages=[Message(role="user", content=prompt)],
                max_tokens=1024,
                system=SMOKER_DIAGNOSIS_PROMPT,
                temperature=0.0,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning("LLM diagnosis failed: %s", e)
            return f"LLM diagnosis failed: {e}"


# ---------------------------------------------------------------------------
# App detection
# ---------------------------------------------------------------------------


def _detect_app(project_path: Path) -> AppInfo:
    """Scan project files to detect app type, entry point, port, routes."""
    info = AppInfo()

    # Collect source files
    py_files: list[Path] = []
    js_files: list[Path] = []
    html_files: list[Path] = []

    for f in sorted(project_path.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(project_path)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if f.suffix == ".py":
            py_files.append(f)
        elif f.suffix in (".js", ".ts"):
            js_files.append(f)
        elif f.suffix in (".html", ".htm"):
            html_files.append(f)

    # Read all source files and detect framework
    file_contents: dict[str, str] = {}
    for f in py_files + js_files:
        try:
            file_contents[str(f.relative_to(project_path))] = f.read_text(errors="replace")
        except Exception:
            pass

    # Detect framework
    for rel_path, content in file_contents.items():
        for pattern, framework in _FRAMEWORK_PATTERNS:
            if pattern.search(content):
                info.framework = framework
                info.is_python = framework != "express"
                break
        if info.framework != "unknown":
            break

    # If only HTML files and no framework detected, it's a static site
    if info.framework == "unknown" and html_files and not py_files and not js_files:
        info.framework = "static"
        return info

    # Find entry point
    entry_candidates = _PYTHON_ENTRY_POINTS if info.is_python else _NODE_ENTRY_POINTS
    for name in entry_candidates:
        if (project_path / name).is_file():
            info.entry_point = name
            break

    # If no top-level entry point, search for __main__.py or nested entry points
    if info.entry_point is None:
        for f in py_files:
            rel = str(f.relative_to(project_path)).replace("\\", "/")
            if rel.endswith("__main__.py"):
                info.entry_point = rel
                break

    # Detect requirements file
    for name in ["requirements.txt", "requirements.in"]:
        if (project_path / name).is_file():
            info.requirements_file = name
            break
    if info.requirements_file is None and (project_path / "pyproject.toml").is_file():
        info.requirements_file = "pyproject.toml"

    # Detect port, routes, WS endpoints, static dirs from all source files
    for _rel_path, content in file_contents.items():
        # Port
        for pattern in _PORT_PATTERNS:
            match = pattern.search(content)
            if match:
                info.port = int(match.group(1))

        # Detect APIRouter prefix for this file (FastAPI)
        prefix = ""
        prefix_match = _ROUTER_PREFIX_PATTERN.search(content)
        if prefix_match:
            prefix = prefix_match.group(1)

        # Routes — only keep values that look like URL paths (start with /)
        for pattern in _ROUTE_PATTERNS:
            for match in pattern.finditer(content):
                route = match.group(1)
                if not route.startswith("/"):
                    continue
                # Prepend router prefix (avoid double-prefixing)
                if prefix and not route.startswith(prefix):
                    route = prefix + route
                if route not in info.routes:
                    info.routes.append(route)

        # WebSocket endpoints — only keep path-like values (start with /)
        for pattern in _WS_PATTERNS:
            for match in pattern.finditer(content):
                ep = match.group(1)
                if ep.startswith("/") and ep not in info.ws_endpoints:
                    info.ws_endpoints.append(ep)

        # Static directories
        for pattern in _STATIC_PATTERNS:
            for match in pattern.finditer(content):
                sd = match.group(1)
                # Normalize to URL-path style (leading slash)
                if not sd.startswith("/"):
                    sd = "/" + sd
                if sd not in info.static_dirs:
                    info.static_dirs.append(sd)

    return info


def _fetch_openapi_routes(port: int) -> dict[str, list[str]] | None:
    """Fetch /openapi.json from a running FastAPI server and extract routes.

    Returns a dict mapping path -> list of HTTP methods (upper-case),
    or None on any failure (404, timeout, bad JSON).
    """
    import json
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/openapi.json"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        logger.debug("Failed to fetch OpenAPI spec from %s", url)
        return None

    paths = data.get("paths")
    if not isinstance(paths, dict):
        logger.debug("OpenAPI spec has no valid 'paths' key")
        return None

    # Standard HTTP methods defined by OpenAPI
    http_methods = {"get", "post", "put", "delete", "patch", "options", "head", "trace"}

    route_methods: dict[str, list[str]] = {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        methods = [m.upper() for m in path_item if m.lower() in http_methods]
        if methods:
            route_methods[path] = methods

    return route_methods if route_methods else None


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def _setup_env(project_path: Path, app_info: AppInfo) -> Path | None:
    """Create an isolated venv and install dependencies.

    Returns the venv path, or None if no deps to install.
    """
    if not app_info.requirements_file:
        return None

    req_path = project_path / app_info.requirements_file

    # Create temp venv
    venv_dir = Path(tempfile.mkdtemp(prefix="laneswarm-smoke-"))
    logger.debug("Creating smoke test venv at %s", venv_dir)

    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            timeout=60,
            check=True,
        )
    except Exception as e:
        logger.warning("Failed to create venv: %s", e)
        shutil.rmtree(venv_dir, ignore_errors=True)
        return None

    # Get pip path
    if sys.platform == "win32":
        pip_path = venv_dir / "Scripts" / "pip.exe"
    else:
        pip_path = venv_dir / "bin" / "pip"

    if not pip_path.exists():
        logger.warning("pip not found in venv at %s", pip_path)
        shutil.rmtree(venv_dir, ignore_errors=True)
        return None

    # Install dependencies
    try:
        if app_info.requirements_file == "pyproject.toml":
            # pip install . for pyproject.toml
            result = subprocess.run(
                [str(pip_path), "install", "."],
                capture_output=True,
                timeout=INSTALL_TIMEOUT,
                cwd=str(project_path),
            )
        else:
            result = subprocess.run(
                [str(pip_path), "install", "-r", str(req_path)],
                capture_output=True,
                timeout=INSTALL_TIMEOUT,
                cwd=str(project_path),
            )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"pip install failed (exit {result.returncode}): {stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("pip install timed out after %ds", INSTALL_TIMEOUT)
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise RuntimeError(f"pip install timed out after {INSTALL_TIMEOUT}s")
    except RuntimeError:
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise
    except Exception as e:
        logger.warning("pip install failed: %s", e)
        shutil.rmtree(venv_dir, ignore_errors=True)
        raise RuntimeError(f"pip install failed: {e}") from e

    return venv_dir


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def _check_port_available(port: int) -> None:
    """Raise RuntimeError if the port is already in use."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            raise RuntimeError(f"Port {port} already in use — cannot start smoke test server")
    except (ConnectionRefusedError, OSError, TimeoutError):
        pass  # Port is free — expected case


def _start_server(
    project_path: Path,
    app_info: AppInfo,
    venv_path: Path | None,
) -> subprocess.Popen | None:
    """Start the server as a subprocess."""
    if app_info.entry_point is None:
        return None

    # Pre-check: ensure port is available
    _check_port_available(app_info.port)

    # Determine the python/node executable
    if app_info.is_python:
        if venv_path:
            if sys.platform == "win32":
                exe = str(venv_path / "Scripts" / "python.exe")
            else:
                exe = str(venv_path / "bin" / "python")
        else:
            exe = sys.executable
        cmd = [exe, app_info.entry_point]
    else:
        cmd = ["node", app_info.entry_point]

    env = os.environ.copy()
    # Add project path to PYTHONPATH so local imports work
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_path) + (os.pathsep + existing if existing else "")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.debug("Started server (PID %d): %s", proc.pid, " ".join(cmd))
        return proc
    except Exception as e:
        logger.warning("Failed to start server: %s", e)
        return None


def _wait_for_port(port: int, timeout: float) -> bool:
    """Wait for a TCP port to accept connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


def _kill_server(proc: subprocess.Popen) -> None:
    """Kill the server process and all children."""
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception:
        pass
    logger.debug("Server process terminated")


def _cleanup_venv(venv_path: Path) -> None:
    """Remove the temporary venv."""
    try:
        shutil.rmtree(venv_path, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HTTP smoke checks
# ---------------------------------------------------------------------------


def _run_http_checks(port: int, app_info: AppInfo) -> list[dict]:
    """Run HTTP checks against the server, respecting per-route methods."""

    checks: list[dict] = []
    base_url = f"http://127.0.0.1:{port}"

    # Check 1: GET /
    checks.append(_http_get(base_url + "/", "http_root"))

    # Check 2: Discovered routes (skip / since we already checked it)
    for route in app_info.routes:
        if route == "/" or route.startswith("/ws"):
            continue
        # Only test GET-able routes (skip wildcards and params)
        if "{" in route or "<" in route or "*" in route:
            continue

        methods = app_info.route_methods.get(route)
        if methods and "GET" in methods:
            # GET is available — use existing GET check
            checks.append(_http_get(base_url + route, f"http_route:{route}"))
        elif methods:
            # No GET — use the first available method
            method = methods[0]
            checks.append(_http_request(base_url + route, method, f"http_route:{route}"))
        else:
            # No method info — fall back to GET (existing behavior)
            checks.append(_http_get(base_url + route, f"http_route:{route}"))

    # Check 3: Static assets (check if /static/ or similar serves files)
    for static_prefix in app_info.static_dirs:
        # Defensive join: ensure exactly one / between base and prefix
        url = base_url.rstrip("/") + "/" + static_prefix.lstrip("/")
        if not url.endswith("/"):
            url += "/"
        checks.append(_http_get(url, f"http_static:{static_prefix}"))

    return checks


def _http_get(url: str, check_name: str) -> dict:
    """Perform a single HTTP GET and return a check result."""
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=HTTP_REQUEST_TIMEOUT) as resp:
            status = resp.status
            body = resp.read(4096).decode("utf-8", errors="replace")

            if 200 <= status < 300:
                detail = f"HTTP {status}, {len(body)} bytes"
                if not body.strip():
                    detail = f"HTTP {status}, empty body (ok)"
                return {
                    "name": check_name,
                    "passed": True,
                    "detail": detail,
                }
            else:
                return {
                    "name": check_name,
                    "passed": False,
                    "detail": f"HTTP {status}: {body[:200]}",
                }
    except urllib.error.HTTPError as e:
        # 422 = FastAPI validation error — route exists but
        # requires query params we didn't supply
        if e.code == 422:
            return {
                "name": check_name,
                "passed": True,
                "detail": ("HTTP 422 (validation error — route exists, missing params)"),
            }
        return {
            "name": check_name,
            "passed": False,
            "detail": f"HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {
            "name": check_name,
            "passed": False,
            "detail": f"Request failed: {e}",
        }


def _http_request(url: str, method: str, check_name: str) -> dict:
    """Perform an HTTP request with the given method and return a check result.

    For POST/PUT/PATCH sends an empty JSON body.
    Accepts 2xx as pass, and 422 (Unprocessable Entity) as pass for non-GET
    methods — 422 means FastAPI received and validated the request (route works,
    just missing required fields).
    """
    import json
    import urllib.error
    import urllib.request

    body = None
    headers = {}
    if method.upper() in ("POST", "PUT", "PATCH"):
        body = json.dumps({}).encode("utf-8")
        headers["Content-Type"] = "application/json"

    try:
        req = urllib.request.Request(url, method=method.upper(), data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=HTTP_REQUEST_TIMEOUT) as resp:
            status = resp.status
            resp_body = resp.read(4096).decode("utf-8", errors="replace")

            if 200 <= status < 300:
                return {
                    "name": check_name,
                    "passed": True,
                    "detail": f"{method.upper()} HTTP {status}, {len(resp_body)} bytes",
                }
            else:
                return {
                    "name": check_name,
                    "passed": False,
                    "detail": f"{method.upper()} HTTP {status}: {resp_body[:200]}",
                }
    except urllib.error.HTTPError as e:
        # 422 = FastAPI validation error — route exists, just missing fields
        if e.code == 422:
            return {
                "name": check_name,
                "passed": True,
                "detail": f"{method.upper()} HTTP 422 (validation error — route exists)",
            }
        return {
            "name": check_name,
            "passed": False,
            "detail": f"{method.upper()} HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {
            "name": check_name,
            "passed": False,
            "detail": f"{method.upper()} request failed: {e}",
        }


# ---------------------------------------------------------------------------
# WebSocket smoke checks
# ---------------------------------------------------------------------------


def _run_ws_checks(port: int, app_info: AppInfo) -> list[dict]:
    """Run WebSocket handshake + functional checks."""
    checks: list[dict] = []

    for ws_ep in app_info.ws_endpoints:
        check = _ws_handshake(port, ws_ep)
        checks.append(check)

        # If handshake succeeded, also do a functional check
        if check["passed"]:
            func_check = _ws_functional_check(port, ws_ep)
            checks.append(func_check)

    return checks


def _ws_handshake(port: int, endpoint: str) -> dict:
    """Attempt a WebSocket handshake and return a check result.

    Uses a raw HTTP upgrade request — no websockets library needed.
    """
    import base64

    check_name = f"ws_handshake:{endpoint}"

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    # Generate WebSocket key
    ws_key = base64.b64encode(os.urandom(16)).decode("ascii")

    request_lines = [
        f"GET {endpoint} HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {ws_key}",
        "Sec-WebSocket-Version: 13",
        "",
        "",
    ]
    request_bytes = "\r\n".join(request_lines).encode("ascii")

    try:
        sock = socket.create_connection(("127.0.0.1", port), timeout=HTTP_REQUEST_TIMEOUT)
        try:
            sock.sendall(request_bytes)
            response = sock.recv(4096).decode("utf-8", errors="replace")

            if "101" in response and "Upgrade" in response:
                return {
                    "name": check_name,
                    "passed": True,
                    "detail": "WebSocket handshake successful (101 Switching Protocols)",
                }
            else:
                # Extract status line
                status_line = response.split("\r\n")[0] if response else "no response"
                return {
                    "name": check_name,
                    "passed": False,
                    "detail": f"WebSocket handshake failed: {status_line[:200]}",
                }
        finally:
            sock.close()
    except Exception as e:
        return {
            "name": check_name,
            "passed": False,
            "detail": f"WebSocket connection failed: {e}",
        }


def _build_ws_text_frame(payload: bytes) -> bytes:
    """Build a masked WebSocket text frame (RFC 6455).

    Client-to-server frames MUST be masked per the spec.
    """
    import struct

    frame = bytearray()
    # FIN=1, opcode=0x01 (text)
    frame.append(0x81)
    length = len(payload)
    # Mask bit = 1
    if length <= 125:
        frame.append(0x80 | length)
    elif length <= 65535:
        frame.append(0x80 | 126)
        frame.extend(struct.pack("!H", length))
    else:
        frame.append(0x80 | 127)
        frame.extend(struct.pack("!Q", length))
    # Masking key (4 random bytes)
    mask = os.urandom(4)
    frame.extend(mask)
    # Masked payload
    for i, b in enumerate(payload):
        frame.append(b ^ mask[i % 4])
    return bytes(frame)


def _ws_functional_check(port: int, endpoint: str) -> dict:
    """Attempt a WebSocket handshake, then send a text frame.

    After 101 Switching Protocols, sends a small text frame
    (``{"type":"ping"}``) and reads the response.  If the server
    immediately closes the connection or sends a Close frame,
    that is flagged.  Uses raw socket framing (RFC 6455), no
    external dependency needed.
    """
    import base64
    import struct

    check_name = f"ws_functional:{endpoint}"

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    ws_key = base64.b64encode(os.urandom(16)).decode("ascii")

    request_lines = [
        f"GET {endpoint} HTTP/1.1",
        f"Host: 127.0.0.1:{port}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {ws_key}",
        "Sec-WebSocket-Version: 13",
        "",
        "",
    ]
    request_bytes = "\r\n".join(request_lines).encode("ascii")

    try:
        sock = socket.create_connection(
            ("127.0.0.1", port),
            timeout=HTTP_REQUEST_TIMEOUT,
        )
        try:
            sock.sendall(request_bytes)
            response = sock.recv(4096).decode(
                "utf-8",
                errors="replace",
            )

            if "101" not in response or "Upgrade" not in response:
                status_line = response.split("\r\n")[0]
                return {
                    "name": check_name,
                    "passed": False,
                    "detail": (f"Handshake failed: {status_line[:200]}"),
                }

            # Handshake succeeded — send a text frame
            payload = b'{"type":"ping"}'
            frame = _build_ws_text_frame(payload)
            sock.sendall(frame)

            # Read response (wait up to 3 seconds)
            sock.settimeout(3.0)
            try:
                resp_data = sock.recv(4096)
            except socket.timeout:
                # No response — server didn't crash
                return {
                    "name": check_name,
                    "passed": True,
                    "detail": ("WS handshake OK, sent ping, no response in 3s (acceptable)"),
                }

            if not resp_data:
                return {
                    "name": check_name,
                    "passed": False,
                    "detail": ("WS server closed connection immediately after receiving message"),
                }

            # Check if it's a Close frame (opcode 0x08)
            if len(resp_data) >= 2:
                opcode = resp_data[0] & 0x0F
                if opcode == 0x08:
                    close_code = ""
                    payload_len = resp_data[1] & 0x7F
                    if payload_len >= 2:
                        close_code = str(struct.unpack("!H", resp_data[2:4])[0])
                    return {
                        "name": check_name,
                        "passed": False,
                        "detail": (
                            "WS server sent Close frame "
                            "after receiving message"
                            f" (code: {close_code})"
                        ),
                    }

            return {
                "name": check_name,
                "passed": True,
                "detail": (
                    f"WS handshake OK, message exchange succeeded ({len(resp_data)} bytes back)"
                ),
            }
        finally:
            sock.close()
    except Exception as e:
        return {
            "name": check_name,
            "passed": False,
            "detail": (f"WS functional check failed: {e}"),
        }
