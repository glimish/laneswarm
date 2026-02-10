"""Planner agent: interactive CLI conversation that generates a task graph.

The planner:
1. Reads the user's project spec (from argument or interactive input)
2. Interviews the user for clarification
3. Generates folder structure and task dependency graph
4. Stores everything in Flanes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from flanes.repo import Repository
from flanes.state import AgentIdentity

from .config import Config
from .events import EventBus, EventType
from .flanes_bridge import init_project, store_task_graph
from .model_selector import select_planner_model
from .prompts import PLANNER_INTERVIEW_PROMPT, PLANNER_SYSTEM_PROMPT
from .providers import Message, ProviderRegistry
from .task_graph import Task, TaskGraph

logger = logging.getLogger(__name__)


class Planner:
    """Interactive planner that generates a task graph from a project spec."""

    def __init__(
        self,
        project_path: Path,
        config: Config,
        registry: ProviderRegistry,
        event_bus: EventBus | None = None,
    ) -> None:
        self.project_path = project_path
        self.config = config
        self.registry = registry
        self.event_bus = event_bus or EventBus()
        self.model_id = select_planner_model(config)
        self._conversation: list[Message] = []

    async def plan(self, spec: str) -> TaskGraph:
        """Generate a task graph from a project specification.

        This is the main entry point. For interactive mode, use plan_interactive().
        """
        # Initialize Flanes repo
        repo = init_project(self.project_path, self.config)

        self.event_bus.emit(EventType.PLAN_CREATED, spec_length=len(spec))

        # Send spec to the planner LLM
        self._conversation = [
            Message(
                role="user",
                content=(
                    f"Here is the project specification:\n\n{spec}\n\n"
                    "Generate a task graph for this project. "
                    "Return ONLY a single JSON code block (```json ... ```) "
                    "containing the full task graph with project_structure, "
                    "conventions, shared_interfaces, and tasks. "
                    "Do NOT include any explanation or discussion — "
                    "ONLY the JSON code block."
                ),
            ),
        ]

        response = await self.registry.complete(
            model_id=self.model_id,
            messages=self._conversation,
            max_tokens=16384,
            system=PLANNER_SYSTEM_PROMPT,
            temperature=0.0,
        )

        # Parse the task graph from the response
        task_graph = _parse_task_graph(response.content)

        # Validate
        errors = task_graph.validate()
        if errors:
            logger.warning("Task graph validation errors: %s", errors)

        # Create folder structure
        _create_folder_structure(self.project_path, response.content)

        # Store in Flanes
        store_task_graph(repo, task_graph)

        return task_graph

    def plan_sync(self, spec: str) -> TaskGraph:
        """Synchronous wrapper for plan()."""
        import asyncio
        import warnings

        # Suppress ResourceWarning from subprocess transports on Windows
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            return asyncio.run(self.plan(spec))

    async def interview(self) -> str:
        """Interactive interview to gather project specification.

        Returns the assembled specification string.
        """
        from rich.console import Console
        from rich.prompt import Prompt

        console = Console()
        console.print("\n[bold]Laneswarm Project Planner[/bold]\n")
        console.print(PLANNER_INTERVIEW_PROMPT)

        spec_parts = []
        conversation = []

        # Initial prompt
        user_input = Prompt.ask("\n[bold cyan]Your project description[/bold cyan]")
        spec_parts.append(user_input)

        # Ask follow-up questions
        conversation.append(Message(role="user", content=user_input))

        for _ in range(3):  # Max 3 rounds of follow-ups
            followup_prompt = (
                f"The user described their project:\n\n{user_input}\n\n"
                "Do you have any clarifying questions? If the description is clear enough, "
                "respond with READY. Otherwise ask 1-2 focused questions."
            )

            response = await self.registry.complete(
                model_id=self.model_id,
                messages=[Message(role="user", content=followup_prompt)],
                max_tokens=1024,
                system=PLANNER_SYSTEM_PROMPT,
                temperature=0.0,
            )

            if "READY" in response.content.upper():
                break

            console.print(f"\n[bold yellow]Planner:[/bold yellow] {response.content}")
            answer = Prompt.ask("\n[bold cyan]Your answer[/bold cyan]")
            spec_parts.append(f"Q: {response.content}\nA: {answer}")
            user_input = answer

        return "\n\n".join(spec_parts)


def _parse_task_graph(content: str) -> TaskGraph:
    """Parse a task graph from LLM response content.

    Extracts JSON from the response (may be wrapped in markdown code blocks).
    """
    logger.debug(
        "Raw planner response (%d chars): %.500s%s",
        len(content), content, "..." if len(content) > 500 else "",
    )

    data = _extract_json(content)

    if data is None:
        logger.error(
            "Could not extract JSON from planner response. "
            "Full LLM response (first 2000 chars): %s",
            content[:2000] if content else "(empty response)",
        )
        raise ValueError(
            "Could not parse task graph from LLM response: "
            "no valid JSON found in response"
        )

    tasks_data = data.get("tasks", [])
    tasks = []
    for t in tasks_data:
        tasks.append(Task(
            task_id=t["task_id"],
            title=t["title"],
            description=t["description"],
            dependencies=t.get("dependencies", []),
            files_to_create=t.get("files_to_create", []),
            files_to_read=t.get("files_to_read", []),
            estimated_complexity=t.get("estimated_complexity", "medium"),
            evaluators=t.get("evaluators", []),
        ))

    graph = TaskGraph(tasks)

    # Extract interface contracts from planner output
    graph.conventions = data.get("conventions", {})
    graph.shared_interfaces = data.get("shared_interfaces", [])

    return graph


def _extract_json(content: str) -> dict | None:
    """Extract the JSON object containing 'tasks' from LLM response content.

    Tries multiple strategies: ```json blocks, generic code blocks,
    brace matching, and finally the raw content.
    """
    import re

    # Strategy 1: ```json code block
    match = re.search(r'```json\s*\n(.*?)```', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2: generic code block containing "tasks"
    match = re.search(r'```\s*\n(.*?)```', content, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if '"tasks"' in candidate or '"project_structure"' in candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 3: brace matching for outermost JSON objects containing "tasks"
    # Try each top-level '{' in the content until we find one that parses
    search_start = 0
    while True:
        brace_start = content.find("{", search_start)
        if brace_start < 0:
            break
        depth = 0
        found_end = False
        for i in range(brace_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[brace_start : i + 1]
                    if '"tasks"' in candidate:
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                    found_end = True
                    search_start = i + 1
                    break
        if not found_end:
            break

    # Strategy 4: entire content as JSON
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    return None


def _create_folder_structure(project_path: Path, content: str) -> None:
    """Create the project folder structure from planner output."""
    data = _extract_json(content)
    if not data:
        return

    structure = data.get("project_structure", [])
    for path_str in structure:
        full_path = project_path / path_str
        if path_str.endswith("/"):
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_path.exists():
                full_path.touch()
