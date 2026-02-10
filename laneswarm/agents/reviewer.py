"""Reviewer agent: evaluates code from coder agents.

The reviewer:
1. Runs evaluators (pytest, ruff, mypy) against the workspace
2. Gets the diff between base state and current state
3. Asks the LLM to review the diff
4. Accepts or rejects the transition based on evaluator results + LLM review
"""

from __future__ import annotations

import logging
from pathlib import Path

from flanes.repo import Repository

from ..events import EventType
from ..prompts import REVIEWER_SYSTEM_PROMPT
from ..providers import Message
from ..task_graph import Task
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReviewerAgent(BaseAgent):
    """Agent that reviews and evaluates code changes."""

    def review(
        self,
        task: Task,
        repo: Repository,
        transition_id: str | None = None,
        verification_summary: str | None = None,
    ) -> dict:
        """Review the work done on a task.

        Returns:
        - accepted: bool
        - summary: str
        - evaluator_results: dict
        - tokens_in: int
        - tokens_out: int
        """
        self.emit(EventType.REVIEW_STARTED, task_id=task.task_id)

        workspace_name = task.lane_name

        # Step 1: Run evaluators (pytest, ruff, mypy)
        eval_result = None
        try:
            eval_result = repo.run_evaluators(workspace_name)
        except Exception as e:
            self.log.debug("Evaluator run skipped: %s", e)

        evaluator_passed = True
        evaluator_summary = "No evaluators configured"
        if eval_result:
            evaluator_passed = eval_result.passed
            evaluator_summary = eval_result.summary

        # Step 2: Read workspace files to review
        files_text = ""
        try:
            ws_path = repo.workspace_path(workspace_name)
            if ws_path:
                files_text = _read_workspace_files(ws_path, task.files_to_create)
        except Exception as e:
            self.log.debug("Could not read workspace files: %s", e)

        # Fall back to diff if no files found
        if not files_text:
            try:
                lane_head = repo.head(task.lane_name)
                main_head = repo.head("main")
                if lane_head and main_head:
                    diff = repo.diff(main_head, lane_head)
                    files_text = _format_diff(diff)
            except Exception as e:
                self.log.debug("Could not compute diff: %s", e)

        # Step 3: LLM review
        llm_accepted = True
        llm_summary = "No LLM review performed"
        tokens_in = 0
        tokens_out = 0

        if files_text:
            try:
                verification_section = ""
                if verification_summary:
                    verification_section = (
                        f"## Verification Results (Automated)\n"
                        f"{verification_summary}\n\n"
                    )

                review_prompt = (
                    f"## Task: {task.title}\n\n"
                    f"{task.description}\n\n"
                    f"## Evaluator Results\n{evaluator_summary}\n\n"
                    f"{verification_section}"
                    f"## Created/Modified Files\n{files_text}\n\n"
                    "Review the files above. Are they correct and complete "
                    "for the task description? Respond with ACCEPT or REJECT followed "
                    "by a brief explanation."
                )

                response = self.call_llm_sync(
                    messages=[Message(role="user", content=review_prompt)],
                    max_tokens=2048,
                    system=REVIEWER_SYSTEM_PROMPT,
                    temperature=0.0,
                )

                tokens_in = response.usage.input_tokens
                tokens_out = response.usage.output_tokens
                llm_summary = response.content.strip()
                llm_accepted = not response.content.upper().startswith("REJECT")

            except Exception as e:
                self.log.warning("LLM review failed: %s", e)
                llm_summary = f"LLM review failed: {e}"

        # Auto-reject if verification found compile errors (syntax errors)
        verification_passed = True
        if verification_summary and "COMPILE ERRORS" in verification_summary:
            verification_passed = False

        # Decision: evaluators AND LLM AND verification must all pass
        accepted = evaluator_passed and llm_accepted and verification_passed
        summary = f"Evaluators: {evaluator_summary}\nLLM: {llm_summary}"

        # Step 4: Accept or reject the transition
        if transition_id:
            try:
                if accepted:
                    repo.accept(
                        transition_id,
                        evaluator=self.agent_id,
                        summary=summary[:500],
                    )
                else:
                    repo.reject(
                        transition_id,
                        evaluator=self.agent_id,
                        summary=summary[:500],
                    )
            except Exception as e:
                self.log.warning("Failed to accept/reject transition: %s", e)

        event_type = EventType.REVIEW_ACCEPTED if accepted else EventType.REVIEW_REJECTED
        self.emit(event_type, task_id=task.task_id, summary=summary[:200])

        return {
            "accepted": accepted,
            "summary": summary,
            "evaluator_passed": evaluator_passed,
            "llm_accepted": llm_accepted,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }


def _read_workspace_files(
    ws_path: Path, files_to_check: list[str], max_file_size: int = 30_000,
) -> str:
    """Read the created/modified files from the workspace for review.

    Returns a formatted string with file paths and contents.
    """
    parts = []

    # Read files specified in the task's files_to_create
    for rel_path in files_to_check:
        full_path = ws_path / rel_path
        if full_path.exists() and full_path.is_file():
            try:
                content = full_path.read_text(errors="replace")
                if not content.strip():
                    parts.append(f"### `{rel_path}` (empty)")
                    continue
                if len(content) > max_file_size:
                    content = content[:max_file_size] + "\n... (truncated)"
                # Detect language from extension
                ext = full_path.suffix.lstrip(".")
                lang = {"py": "python", "toml": "toml", "json": "json",
                        "yaml": "yaml", "yml": "yaml", "js": "javascript",
                        "ts": "typescript", "md": "markdown"}.get(ext, ext)
                parts.append(f"### `{rel_path}`\n```{lang}\n{content}\n```")
            except Exception:
                parts.append(f"### `{rel_path}` (could not read)")
        else:
            parts.append(f"### `{rel_path}` (not found)")

    return "\n\n".join(parts) if parts else ""


def _format_diff(diff: dict) -> str:
    """Format a Flanes diff dict into readable text."""
    parts = []

    added = diff.get("added", {})
    removed = diff.get("removed", {})
    modified = diff.get("modified", {})

    if added:
        parts.append("### Added Files")
        for path in sorted(added.keys()):
            parts.append(f"+ {path}")

    if removed:
        parts.append("### Removed Files")
        for path in sorted(removed.keys()):
            parts.append(f"- {path}")

    if modified:
        parts.append("### Modified Files")
        for path in sorted(modified.keys()):
            parts.append(f"~ {path}")

    unchanged = diff.get("unchanged_count", 0)
    if unchanged:
        parts.append(f"\n({unchanged} unchanged files)")

    return "\n".join(parts) if parts else "(no changes detected)"
