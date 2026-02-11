"""Verification: lightweight smoke tests for generated code.

Runs compile checks, import checks, and pytest (if applicable)
against the workspace to catch errors before LLM review.

Also provides post-completion integration validation that checks
wiring between all assembled project files.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task_graph import Task, TaskGraph

logger = logging.getLogger(__name__)

# Maximum time (seconds) for any single verification subprocess
VERIFY_TIMEOUT = 30


def verify_workspace(
    workspace_path: Path,
    files_written: list[str],
    task_graph: TaskGraph | None = None,
    task: Task | None = None,
) -> dict:
    """Run smoke tests on the workspace after coder writes files.

    Returns:
        {
            "passed": bool,
            "compile_errors": list[dict],      # {file, error}
            "import_errors": list[dict],       # {module, error}
            "contract_violations": list[dict], # hard fails (interfaces, producer msgs)
            "contract_warnings": list[dict],   # soft warnings (consumer handlers)
            "pytest_output": str | None,
            "pytest_passed": bool | None,
            "summary": str,
        }
    """
    compile_errors: list[dict] = []
    import_errors: list[dict] = []
    contract_violations: list[dict] = []
    contract_warnings: list[dict] = []
    pytest_output: str | None = None
    pytest_passed: bool | None = None

    py_files = [f for f in files_written if f.endswith(".py")]

    # Step 1: py_compile on every written .py file
    for rel_path in py_files:
        full_path = workspace_path / rel_path
        if not full_path.exists():
            continue
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(full_path)],
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT,
                cwd=str(workspace_path),
            )
            if result.returncode != 0:
                error_text = (result.stderr or result.stdout).strip()
                compile_errors.append({"file": rel_path, "error": error_text})
        except subprocess.TimeoutExpired:
            compile_errors.append({"file": rel_path, "error": "Compile check timed out"})
        except Exception as e:
            compile_errors.append({"file": rel_path, "error": str(e)})

    # Step 2: Import check for each written .py module
    # Only attempt if no compile errors (imports would fail anyway)
    if not compile_errors:
        for rel_path in py_files:
            if rel_path.endswith("__init__.py"):
                continue  # Skip __init__.py to avoid circular imports
            # Convert path to module name: src/routes/projects.py -> src.routes.projects
            module_name = (
                rel_path.replace("/", ".").replace("\\", ".").removesuffix(".py")
            )
            try:
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module_name}"],
                    capture_output=True,
                    text=True,
                    timeout=VERIFY_TIMEOUT,
                    cwd=str(workspace_path),
                    env=_build_import_env(workspace_path),
                )
                if result.returncode != 0:
                    error_text = (result.stderr or result.stdout).strip()
                    # Only record if it's NOT a missing module from another task.
                    # ModuleNotFoundError / ImportError for cross-task deps is expected.
                    if not _is_cross_task_import_error(error_text):
                        import_errors.append(
                            {"module": module_name, "error": error_text}
                        )
            except subprocess.TimeoutExpired:
                import_errors.append(
                    {"module": module_name, "error": "Import check timed out"}
                )
            except Exception as e:
                import_errors.append({"module": module_name, "error": str(e)})

    # Step 3: Run pytest if test files were written
    # Only run on files that are actual test files (test_*.py or *_test.py),
    # not fixtures like conftest.py or __init__.py in test directories.
    test_files = [
        f for f in files_written
        if f.endswith(".py")
        and (
            f.split("/")[-1].startswith("test_")
            or f.split("\\")[-1].startswith("test_")
            or f.split("/")[-1].endswith("_test.py")
            or f.split("\\")[-1].endswith("_test.py")
        )
    ]
    if test_files and not compile_errors:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=short", "-q"] + test_files,
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT * 2,
                cwd=str(workspace_path),
                env=_build_import_env(workspace_path),
            )
            pytest_output = (result.stdout + result.stderr).strip()
            # Truncate very long output
            if len(pytest_output) > 2000:
                pytest_output = pytest_output[:2000] + "\n... (truncated)"
            # returncode 0 = all passed, 5 = no tests collected (not a failure)
            pytest_passed = result.returncode in (0, 5)
            if result.returncode == 5:
                pytest_output = None  # "no tests collected" is not an error
                pytest_passed = None  # Don't count it either way
        except subprocess.TimeoutExpired:
            pytest_output = "pytest timed out"
            pytest_passed = False
        except Exception as e:
            pytest_output = f"pytest failed to run: {e}"
            pytest_passed = None

    # Step 4: Contract compliance check (if contracts provided)
    if task_graph and task:
        contract_violations, contract_warnings = _check_contract_compliance(
            workspace_path, files_written, task, task_graph,
        )

    # Clean up verification artifacts so they don't pollute the workspace
    # (these cause spurious conflicts during Flanes integration)
    _cleanup_artifacts(workspace_path)

    # Build summary
    # Only hard failures (compile, import, producer contract violations) affect passed.
    # Consumer contract warnings are advisory — they appear in the summary but
    # do NOT block verification.  Post-completion validate_assembled_project()
    # catches real wiring gaps at the project level.
    passed = not compile_errors and not import_errors and not contract_violations
    if pytest_passed is not None:
        passed = passed and pytest_passed

    summary_parts = []
    if compile_errors:
        summary_parts.append(
            f"COMPILE ERRORS ({len(compile_errors)}):\n"
            + "\n".join(f"  {e['file']}: {e['error']}" for e in compile_errors)
        )
    if import_errors:
        summary_parts.append(
            f"IMPORT ERRORS ({len(import_errors)}):\n"
            + "\n".join(f"  {e['module']}: {e['error']}" for e in import_errors)
        )
    if contract_violations:
        summary_parts.append(
            f"CONTRACT VIOLATIONS ({len(contract_violations)}):\n"
            + "\n".join(
                f"  {v['name']}: {v['detail']}" for v in contract_violations
            )
        )
    if contract_warnings:
        summary_parts.append(
            f"CONTRACT WARNINGS ({len(contract_warnings)}):\n"
            + "\n".join(
                f"  {w['name']}: {w['detail']}" for w in contract_warnings
            )
        )
    if pytest_output and not pytest_passed:
        summary_parts.append(f"PYTEST FAILURES:\n  {pytest_output[:500]}")
    if not summary_parts:
        summary_parts.append("All verification checks passed.")

    return {
        "passed": passed,
        "compile_errors": compile_errors,
        "import_errors": import_errors,
        "contract_violations": contract_violations,
        "contract_warnings": contract_warnings,
        "pytest_output": pytest_output,
        "pytest_passed": pytest_passed,
        "summary": "\n".join(summary_parts),
    }


def _is_cross_task_import_error(error_text: str) -> bool:
    """Check if an error is from importing a module created by another task.

    These errors are expected in a multi-agent build and should be ignored.
    """
    cross_task_indicators = [
        "ModuleNotFoundError",
        "No module named",
        "cannot import name",
    ]
    return any(indicator in error_text for indicator in cross_task_indicators)


def _build_import_env(workspace_path: Path) -> dict:
    """Build environment for import checks with PYTHONPATH set."""
    env = os.environ.copy()
    # Add workspace to PYTHONPATH so local imports work
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(workspace_path) + (
        os.pathsep + existing if existing else ""
    )
    return env


def _cleanup_artifacts(workspace_path: Path) -> None:
    """Remove verification artifacts from the workspace.

    pytest and py_compile leave behind __pycache__ and .pytest_cache
    directories that cause spurious integration conflicts in Flanes.
    """
    import shutil

    for pattern in [".pytest_cache", "__pycache__"]:
        for artifact in workspace_path.rglob(pattern):
            try:
                if artifact.is_dir():
                    shutil.rmtree(artifact, ignore_errors=True)
            except Exception:
                pass  # Best-effort cleanup


def _check_contract_compliance(
    workspace_path: Path,
    files_written: list[str],
    task: Task,
    task_graph: TaskGraph,
) -> tuple[list[dict], list[dict]]:
    """Check that written files comply with interface and protocol contracts.

    Verifies:
    1. Shared interface names appear in the files that define them
    2. Protocol contract message types appear in producer files (hard fail)
    3. Protocol contract message types appear in consumer files (soft warning)

    Returns a tuple of (violations, warnings):
    - violations: hard failures (missing interfaces, producer messages) that
      block verification and trigger coder retries
    - warnings: soft issues (missing consumer handlers) that appear in the
      summary for awareness but do NOT block verification
    """
    violations: list[dict] = []
    warnings: list[dict] = []

    # Check 1: Shared interface names (hard fail)
    for iface in task_graph.shared_interfaces:
        iface_module = iface.get("module", "")
        if iface_module in files_written:
            file_path = workspace_path / iface_module
            if file_path.exists():
                try:
                    content = file_path.read_text(errors="replace")
                    expected_name = iface.get("name", "")
                    if expected_name and expected_name not in content:
                        violations.append({
                            "type": "missing_interface",
                            "name": expected_name,
                            "file": iface_module,
                            "detail": (
                                f"Expected shared interface '{expected_name}' "
                                f"in {iface_module} but it was not found"
                            ),
                        })
                except Exception:
                    pass

    # Check 2: Protocol contract message types for producers (hard fail)
    for pc in task_graph.protocol_contracts:
        if pc.get("producer_task") == task.task_id:
            msg_name = pc.get("name", "")
            if not msg_name:
                continue
            found = False
            for f in files_written:
                fp = workspace_path / f
                if fp.exists():
                    try:
                        content = fp.read_text(errors="replace")
                        if msg_name in content:
                            found = True
                            break
                    except Exception:
                        pass
            if not found:
                violations.append({
                    "type": "missing_protocol_message",
                    "name": msg_name,
                    "file": "",
                    "detail": (
                        f"Task should produce message type '{msg_name}' "
                        f"but it was not found in any written file"
                    ),
                })

    # Check 3: Protocol contract message types for consumers (soft warning)
    # Consumer checks are warnings, not failures, because:
    # - Frontend JS tasks may consume many message types (8-9+)
    # - Handlers may use dispatcher patterns that don't match literal strings
    # - Post-completion validate_assembled_project() catches real wiring gaps
    for pc in task_graph.protocol_contracts:
        if task.task_id in pc.get("consumer_tasks", []):
            msg_name = pc.get("name", "")
            if not msg_name:
                continue
            found = False
            for f in files_written:
                fp = workspace_path / f
                if fp.exists():
                    try:
                        content = fp.read_text(errors="replace")
                        if msg_name in content:
                            found = True
                            break
                    except Exception:
                        pass
            if not found:
                warnings.append({
                    "type": "missing_message_handler",
                    "name": msg_name,
                    "file": "",
                    "detail": (
                        f"Task should handle message type '{msg_name}' "
                        f"but it was not found in any written file"
                    ),
                })

    return violations, warnings


# ---------------------------------------------------------------------------
# Post-completion integration validation
# ---------------------------------------------------------------------------

# Skip these directories when scanning the assembled project
_SKIP_DIRS = {
    ".flanes", ".git", "__pycache__", "node_modules",
    ".venv", "venv", ".pytest_cache", ".env",
}


def validate_assembled_project(
    project_path: Path,
    task_graph: TaskGraph,
) -> dict:
    """Post-completion validation of the fully assembled project.

    Runs after ALL tasks have been promoted to main. Checks:
    1. Compile check: py_compile on every .py file
    2. Protocol compliance: message types in code match protocol_contracts
    3. Wiring validation: message types sent are also handled and vice versa
    4. Anti-pattern scan: TODO/FIXME, empty functions, placeholder text

    Returns::

        {
            "passed": bool,
            "compile_errors": list[dict],
            "protocol_violations": list[dict],
            "wiring_gaps": list[dict],
            "anti_patterns": list[dict],
            "summary": str,
        }
    """
    compile_errors: list[dict] = []
    protocol_violations: list[dict] = []
    wiring_gaps: list[dict] = []
    anti_patterns: list[dict] = []

    # Collect all source files
    all_py = _collect_source_files(project_path, "*.py")
    all_js = _collect_source_files(project_path, "*.js")
    all_source = all_py + all_js

    # --- Tier 1: Compile check on all .py files ---
    for py_file in all_py:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(py_file)],
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT,
                cwd=str(project_path),
            )
            if result.returncode != 0:
                rel = str(py_file.relative_to(project_path))
                error_text = (result.stderr or result.stdout).strip()
                compile_errors.append({"file": rel, "error": error_text})
        except (subprocess.TimeoutExpired, Exception) as e:
            rel = str(py_file.relative_to(project_path))
            compile_errors.append({"file": rel, "error": str(e)})

    # --- Tier 2: Protocol compliance ---
    if task_graph.protocol_contracts:
        protocol_violations = _check_protocol_compliance(
            project_path, all_source, task_graph,
        )

    # --- Tier 3: Wiring validation ---
    wiring_gaps = _check_wiring(project_path, all_source)

    # --- Tier 4: Anti-pattern scan ---
    anti_patterns = _scan_anti_patterns(project_path, all_source)

    # Build summary
    # Compile errors are hard failures; everything else is a warning
    passed = not compile_errors
    summary_parts: list[str] = []

    if compile_errors:
        summary_parts.append(
            f"COMPILE ERRORS ({len(compile_errors)}):\n"
            + "\n".join(
                f"  {e['file']}: {e['error'][:200]}"
                for e in compile_errors
            )
        )
    if protocol_violations:
        summary_parts.append(
            f"PROTOCOL VIOLATIONS ({len(protocol_violations)}):\n"
            + "\n".join(
                f"  {v['name']}: {v['detail']}"
                for v in protocol_violations
            )
        )
    if wiring_gaps:
        summary_parts.append(
            f"WIRING GAPS ({len(wiring_gaps)}):\n"
            + "\n".join(
                f"  {g['message_type']}: {g['detail']}"
                for g in wiring_gaps
            )
        )
    if anti_patterns:
        summary_parts.append(
            f"ANTI-PATTERNS ({len(anti_patterns)}):\n"
            + "\n".join(
                f"  {a['file']}: {a['detail']}"
                for a in anti_patterns[:20]  # Cap to avoid huge output
            )
        )
    if not summary_parts:
        summary_parts.append(
            "All integration validation checks passed."
        )

    return {
        "passed": passed,
        "compile_errors": compile_errors,
        "protocol_violations": protocol_violations,
        "wiring_gaps": wiring_gaps,
        "anti_patterns": anti_patterns,
        "summary": "\n".join(summary_parts),
    }


def _collect_source_files(
    project_path: Path, pattern: str,
) -> list[Path]:
    """Collect source files, skipping hidden/generated directories."""
    results = []
    for f in sorted(project_path.rglob(pattern)):
        if any(part in _SKIP_DIRS for part in f.relative_to(project_path).parts):
            continue
        if f.is_file():
            results.append(f)
    return results


def _check_protocol_compliance(
    project_path: Path,
    all_source: list[Path],
    task_graph: TaskGraph,
) -> list[dict]:
    """Check that protocol contract message types appear in source files.

    For each contract, verifies the exact message type string appears in
    at least one source file from the producer and at least one from the
    consumers.
    """
    violations: list[dict] = []

    # Build a content cache (read each file once)
    file_contents: dict[str, str] = {}
    for f in all_source:
        try:
            file_contents[str(f.relative_to(project_path))] = (
                f.read_text(errors="replace")
            )
        except Exception:
            pass

    all_content = "\n".join(file_contents.values())

    for pc in task_graph.protocol_contracts:
        msg_name = pc.get("name", "")
        if not msg_name:
            continue

        # Check: does the message type string appear anywhere in the project?
        if msg_name not in all_content:
            violations.append({
                "name": msg_name,
                "detail": (
                    f"Message type '{msg_name}' defined in protocol_contracts "
                    f"but not found in any source file"
                ),
            })
            continue

        # Check: does it appear in files that the wiring_map says it should?
        expected_files = set()
        for wm in task_graph.wiring_map:
            if wm.get("artifact") == msg_name:
                if wm.get("defined_in"):
                    expected_files.add(wm["defined_in"])
                expected_files.update(wm.get("used_in", []))

        for expected in expected_files:
            # Normalize path separators for comparison
            normalized = expected.replace("\\", "/")
            found_in_expected = False
            for rel_path, content in file_contents.items():
                if rel_path.replace("\\", "/") == normalized:
                    if msg_name in content:
                        found_in_expected = True
                    break
            if not found_in_expected and normalized in {
                p.replace("\\", "/") for p in file_contents
            }:
                violations.append({
                    "name": msg_name,
                    "detail": (
                        f"Message type '{msg_name}' expected in "
                        f"'{expected}' but not found there"
                    ),
                })

    return violations


# Regex patterns for detecting message sending/handling in different languages
_SEND_PATTERNS = [
    # Python: build_message('type', ...), broadcast(build_message('type', ...))
    re.compile(r"""build_message\(\s*['"](\w+)['"]"""),
    # Python: .send(json.dumps({'type': 'name', ...}))
    re.compile(r"""['"]type['"]\s*:\s*['"](\w+)['"]"""),
    # JS: ws.send(JSON.stringify({type: 'name', ...}))
    re.compile(r"""type\s*:\s*['"](\w+)['"]"""),
]

_HANDLE_PATTERNS = [
    # Python: if msg_type == 'name' / msg_type in ('name', ...)
    re.compile(r"""msg_type\s*(?:==|in\s*\()\s*.*?['"](\w+)['"]"""),
    # JS: case 'name':
    re.compile(r"""case\s+['"](\w+)['"]\s*:"""),
    # Python: elif message_type == 'name'
    re.compile(r"""message_type\s*==\s*['"](\w+)['"]"""),
]


def _check_wiring(
    project_path: Path,
    all_source: list[Path],
) -> list[dict]:
    """Scan source files for message type patterns and find wiring gaps.

    Builds sets of message types sent and handled, then flags:
    - Message types sent but never handled
    - Message types handled but never sent
    """
    sent_types: set[str] = set()
    handled_types: set[str] = set()

    # Common non-message identifiers to ignore
    ignore = {
        "type", "message", "data", "error", "success", "true", "false",
        "null", "undefined", "string", "number", "boolean", "object",
        "function", "return", "default", "break", "text", "json",
    }

    for f in all_source:
        try:
            content = f.read_text(errors="replace")
        except Exception:
            continue

        for pattern in _SEND_PATTERNS:
            for match in pattern.finditer(content):
                name = match.group(1)
                if name.lower() not in ignore:
                    sent_types.add(name)

        for pattern in _HANDLE_PATTERNS:
            for match in pattern.finditer(content):
                name = match.group(1)
                if name.lower() not in ignore:
                    handled_types.add(name)

    gaps: list[dict] = []

    # Sent but never handled
    for msg_type in sorted(sent_types - handled_types):
        gaps.append({
            "message_type": msg_type,
            "detail": f"Message type '{msg_type}' is sent but never handled",
        })

    # Handled but never sent (potential dead handler)
    for msg_type in sorted(handled_types - sent_types):
        gaps.append({
            "message_type": msg_type,
            "detail": (
                f"Message type '{msg_type}' has a handler but is "
                f"never sent (possible dead code or external source)"
            ),
        })

    return gaps


# Anti-pattern detection patterns
_ANTI_PATTERN_CHECKS = [
    # TODO/FIXME markers left in code
    (re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE), "todo_marker"),
    # Empty function bodies (just pass or ...)
    (re.compile(r"def\s+\w+\([^)]*\).*:\s*\n\s+pass\s*$", re.MULTILINE), "empty_function"),
    # raise NotImplementedError in non-abstract methods
    (re.compile(r"raise\s+NotImplementedError"), "not_implemented"),
    # Hardcoded localhost/127.0.0.1 outside of config
    (re.compile(r"""['"](?:localhost|127\.0\.0\.1)['"]"""), "hardcoded_host"),
]


def _scan_anti_patterns(
    project_path: Path,
    all_source: list[Path],
) -> list[dict]:
    """Scan for common anti-patterns in generated code.

    Returns a list of warnings (not failures).
    """
    results: list[dict] = []

    for f in all_source:
        try:
            content = f.read_text(errors="replace")
        except Exception:
            continue

        rel = str(f.relative_to(project_path)).replace("\\", "/")

        # Skip config files for hardcoded host check
        for pattern, kind in _ANTI_PATTERN_CHECKS:
            if kind == "hardcoded_host" and (
                "config" in rel.lower() or "settings" in rel.lower()
            ):
                continue
            for match in pattern.finditer(content):
                # Get line number for context
                line_no = content[:match.start()].count("\n") + 1
                snippet = match.group(0)[:80]
                results.append({
                    "file": rel,
                    "line": line_no,
                    "kind": kind,
                    "detail": f"{kind} at line {line_no}: {snippet}",
                })

    return results
