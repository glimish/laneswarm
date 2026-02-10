"""Verification: lightweight smoke tests for generated code.

Runs compile checks, import checks, and pytest (if applicable)
against the workspace to catch errors before LLM review.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum time (seconds) for any single verification subprocess
VERIFY_TIMEOUT = 30


def verify_workspace(
    workspace_path: Path,
    files_written: list[str],
) -> dict:
    """Run smoke tests on the workspace after coder writes files.

    Returns:
        {
            "passed": bool,
            "compile_errors": list[dict],  # {file, error}
            "import_errors": list[dict],   # {module, error}
            "pytest_output": str | None,
            "pytest_passed": bool | None,
            "summary": str,
        }
    """
    compile_errors: list[dict] = []
    import_errors: list[dict] = []
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

    # Clean up verification artifacts so they don't pollute the workspace
    # (these cause spurious conflicts during Flanes integration)
    _cleanup_artifacts(workspace_path)

    # Build summary
    passed = not compile_errors and not import_errors
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
    if pytest_output and not pytest_passed:
        summary_parts.append(f"PYTEST FAILURES:\n  {pytest_output[:500]}")
    if not summary_parts:
        summary_parts.append("All verification checks passed.")

    return {
        "passed": passed,
        "compile_errors": compile_errors,
        "import_errors": import_errors,
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
