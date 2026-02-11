"""Coder agent: implements a single task in an isolated Flanes workspace.

The coder agent:
1. Receives a task description and workspace path
2. Calls the LLM with a focused prompt (task + relevant files only)
3. Parses the LLM response to extract file operations
4. Writes files to the workspace
5. Records token usage via AgentSession
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..events import EventType
from ..flanes_bridge import build_focused_prompt
from ..prompts import CODER_SYSTEM_PROMPT
from ..providers import Message
from ..task_graph import Task, TaskGraph
from ..verification import verify_workspace
from .base import BaseAgent

logger = logging.getLogger(__name__)

# Maximum iterations of write → verify → fix within a single task execution
MAX_CODER_ITERATIONS = 3


class CoderAgent(BaseAgent):
    """Agent that implements code for a single task."""

    def run(
        self,
        task: Task,
        task_graph: TaskGraph,
        workspace_path: Path,
    ) -> dict:
        """Implement the task in the workspace with iterative verification.

        The coder calls the LLM, writes files, runs compile/import checks,
        and if verification fails, calls the LLM again with the error output.
        Up to MAX_CODER_ITERATIONS attempts.

        Returns a dict with:
        - success: bool
        - files_written: list[str]
        - tokens_in: int
        - tokens_out: int
        - error: str | None
        - verification: dict | None
        """
        self.emit(EventType.AGENT_WORKING, task_id=task.task_id, action="coding")

        total_tokens_in = 0
        total_tokens_out = 0
        all_files_written: list[str] = []
        last_error: str | None = None
        verification_result: dict | None = None

        # Build the initial prompt
        user_prompt = build_focused_prompt(workspace_path, task, task_graph)

        for iteration in range(MAX_CODER_ITERATIONS):
            messages = [Message(role="user", content=user_prompt)]

            try:
                response = self.call_llm_sync(
                    messages=messages,
                    max_tokens=16384,
                    system=CODER_SYSTEM_PROMPT,
                    temperature=0.0,
                )
            except Exception as e:
                self.log.error(
                    "LLM call failed for task '%s' (iter %d): %s",
                    task.task_id, iteration, e,
                )
                last_error = str(e)
                break

            total_tokens_in += response.usage.input_tokens
            total_tokens_out += response.usage.output_tokens

            # Parse and write files
            files_written = _extract_and_write_files(
                response.content, workspace_path, self.log
            )
            all_files_written = list(set(all_files_written + files_written))

            if not files_written:
                last_error = "Coder produced no files"
                self.log.warning(
                    "Task '%s' iteration %d: no files produced",
                    task.task_id, iteration,
                )
                if iteration < MAX_CODER_ITERATIONS - 1:
                    # Retry with a stronger nudge
                    user_prompt = (
                        "CRITICAL: Your previous response contained NO file blocks. "
                        "You MUST output code files using the `path/to/file.py`:\n"
                        "```python\n...\n``` format.\n\n"
                        "DO NOT explain or discuss. ONLY output file blocks.\n\n"
                        + build_focused_prompt(workspace_path, task, task_graph)
                    )
                    continue
                break

            # Verify the written files (including contract compliance if available)
            verification_result = verify_workspace(
                workspace_path, files_written,
                task_graph=task_graph, task=task,
            )

            if verification_result["passed"]:
                self.log.info(
                    "Task '%s' verified on iteration %d", task.task_id, iteration,
                )
                last_error = None
                break

            # Verification failed — build fix prompt for next iteration
            if iteration < MAX_CODER_ITERATIONS - 1:
                self.log.info(
                    "Task '%s' verification failed (iter %d), retrying: %s",
                    task.task_id, iteration,
                    verification_result["summary"][:200],
                )
                user_prompt = (
                    "## VERIFICATION FAILED\n\n"
                    "Your previous code had the following errors:\n\n"
                    f"{verification_result['summary']}\n\n"
                    "Please fix ALL errors and output the COMPLETE corrected files "
                    "using the same `path/to/file.py`:\n```python\n...\n``` format.\n\n"
                    "Output ONLY the corrected file blocks. No explanation needed.\n\n"
                    + build_focused_prompt(workspace_path, task, task_graph)
                )
            else:
                last_error = (
                    f"Verification failed after {MAX_CODER_ITERATIONS} iterations: "
                    + verification_result["summary"][:200]
                )

        self.emit(
            EventType.AGENT_FINISHED,
            task_id=task.task_id,
            files_written=len(all_files_written),
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
        )

        return {
            "success": len(all_files_written) > 0,
            "files_written": all_files_written,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "error": last_error,
            "verification": verification_result,
        }


def _extract_and_write_files(
    content: str,
    workspace_path: Path,
    log: logging.Logger | None = None,
) -> list[str]:
    """Extract file blocks from LLM response and write them to the workspace.

    Supports two formats:
    1. ```filepath: path/to/file.py
       <content>
       ```
    2. --- FILE: path/to/file.py ---
       <content>
       --- END FILE ---
    """
    _log = log or logger
    files_written = []

    # Pattern 1: ```filepath: path/to/file.py
    pattern1 = re.compile(
        r'```(?:filepath|file|path):\s*(.+?)\n(.*?)```',
        re.DOTALL,
    )

    # Pattern 2: --- FILE: path/to/file.py ---
    pattern2 = re.compile(
        r'---\s*FILE:\s*(.+?)\s*---\n(.*?)---\s*END\s*FILE\s*---',
        re.DOTALL,
    )

    # Pattern 3: Fenced code blocks with language hint and preceding path comment
    # Example: `path/to/file.py`:
    # ```python
    # <content>
    # ```
    pattern3 = re.compile(
        r'`([^`\n]+\.\w+)`[:\s]*\n```\w*\n(.*?)```',
        re.DOTALL,
    )

    for pattern in [pattern1, pattern2, pattern3]:
        for match in pattern.finditer(content):
            file_path = match.group(1).strip()
            file_content = match.group(2)

            # Security: prevent path traversal
            if ".." in file_path or file_path.startswith("/") or file_path.startswith("\\"):
                _log.warning("Skipping suspicious path: %s", file_path)
                continue

            # Write file
            full_path = workspace_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(file_content, encoding="utf-8")
            files_written.append(file_path)
            _log.info("Wrote file: %s (%d bytes)", file_path, len(file_content))

    return files_written
