"""Fix planner: generates fix tasks from smoke test diagnosis.

Given a completed task graph and smoke test results (or a user
description of issues), makes a single LLM call to produce 1-N
small fix tasks that can be added to the existing graph.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import Config
from .model_selector import select_planner_model
from .planner import _extract_json
from .prompts import FIX_PLANNER_PROMPT
from .providers import Message, ProviderRegistry
from .task_graph import Task, TaskGraph

logger = logging.getLogger(__name__)


class FixPlanner:
    """Generates fix tasks from smoke test results."""

    def __init__(
        self,
        project_path: Path,
        config: Config,
        registry: ProviderRegistry,
    ) -> None:
        self.project_path = project_path
        self.config = config
        self.registry = registry
        self.model_id = select_planner_model(config)

    def plan_fixes(
        self,
        task_graph: TaskGraph,
        fix_description: str,
        smoke_result: dict | None = None,
    ) -> list[Task]:
        """Generate fix tasks.

        Returns a list of Task objects ready to be added to
        the graph via ``task_graph.add_tasks()``.
        """
        import asyncio

        return asyncio.run(
            self._plan_fixes_async(
                task_graph,
                fix_description,
                smoke_result,
            )
        )

    async def _plan_fixes_async(
        self,
        task_graph: TaskGraph,
        fix_description: str,
        smoke_result: dict | None = None,
    ) -> list[Task]:
        context = self._build_context(
            task_graph,
            fix_description,
            smoke_result,
        )

        response = await self.registry.complete(
            model_id=self.model_id,
            messages=[
                Message(role="user", content=context),
            ],
            max_tokens=4096,
            system=FIX_PLANNER_PROMPT,
            temperature=0.0,
        )

        return self._parse_fix_tasks(
            response.content,
            task_graph,
        )

    def _build_context(
        self,
        task_graph: TaskGraph,
        fix_description: str,
        smoke_result: dict | None,
    ) -> str:
        # Completed tasks summary
        tasks_summary = []
        for task in task_graph.tasks:
            tasks_summary.append(
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "status": task.status.value,
                    "files_to_create": task.files_to_create,
                    "files_written": task.files_written,
                }
            )

        parts = [
            f"## Fix Description\n\n{fix_description}",
            (f"## Completed Tasks\n\n```json\n{json.dumps(tasks_summary, indent=2)}\n```"),
        ]

        if smoke_result:
            parts.append(
                f"## Smoke Test Results\n\n```json\n{json.dumps(smoke_result, indent=2)}\n```"
            )

        return "\n\n".join(parts)

    def _parse_fix_tasks(
        self,
        content: str,
        task_graph: TaskGraph,
    ) -> list[Task]:
        """Parse LLM response into Task objects."""
        data = _extract_json(content)
        if data is None or "tasks" not in data:
            raise ValueError("Fix planner returned no parseable tasks")

        existing_ids = set(task_graph.task_ids)
        tasks: list[Task] = []
        counter = 1

        for td in data["tasks"]:
            # Generate unique fix-NNN ID
            fix_id = f"fix-{counter:03d}"
            while fix_id in existing_ids:
                counter += 1
                fix_id = f"fix-{counter:03d}"
            existing_ids.add(fix_id)
            counter += 1

            tasks.append(
                Task(
                    task_id=fix_id,
                    title=td.get("title", f"Fix {counter}"),
                    description=td.get("description", ""),
                    dependencies=td.get("dependencies", []),
                    files_to_create=td.get("files_to_modify", []),
                    files_to_read=td.get("files_to_read", []),
                    estimated_complexity=td.get("complexity", "low"),
                )
            )

        return tasks
