"""Planner agent: staged planning that generates a task graph.

The planner runs 5 sequential phases, each making a focused LLM call
and emitting events for real-time progress:

1. Analyze  — Understand the spec: objective, constraints, unknowns
2. Research — Identify relevant files, patterns, dependencies
3. Structure — Design phases, ordering, parallelism opportunities
4. Decompose — Break into concrete tasks with full interface contracts
5. Validate — Check for structural issues and fix them
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from .config import Config
from .events import EventBus, EventType
from .flanes_bridge import init_project, store_task_graph
from .model_selector import select_planner_model
from .prompts import (
    PLANNER_ANALYZE_PLAN_PROMPT,
    PLANNER_DECOMPOSE_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_VALIDATE_PROMPT,
)
from .providers import Message, ProviderRegistry
from .task_graph import Task, TaskGraph

logger = logging.getLogger(__name__)

# Phase definitions: (name, display label, prompt, max_tokens)
_PHASES = [
    ("analyze_plan", "Analyzing and planning", PLANNER_ANALYZE_PLAN_PROMPT, 4096),
    ("decompose", "Decomposing into tasks", PLANNER_DECOMPOSE_PROMPT, 16384),
    ("validate", "Validating task graph", PLANNER_VALIDATE_PROMPT, 4096),
]


class Planner:
    """Staged planner that generates a task graph from a project spec."""

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

        Runs 5 sequential phases, each emitting events for progress.
        For interactive mode, use plan_interactive().
        """
        # Initialize Flanes repo
        repo = init_project(self.project_path, self.config)

        self.event_bus.emit(EventType.PLANNING_STARTED, spec_length=len(spec))

        # Scan existing project files for context
        file_listing = _scan_project_files(self.project_path)

        # Phase 1: Analyze + Research + Structure (merged)
        context = f"## Project Specification\n\n{spec}"
        if file_listing:
            context += f"\n\n## Existing Project Files\n{file_listing}"
        merged = await self._run_phase(0, spec, context)

        # Extract sub-results for Phase 2 context
        analysis = merged.get("analysis", {})
        research = merged.get("research", {})
        structure = merged.get("structure", {})

        # Phase 2: Decompose
        decompose_context = (
            f"## Project Spec\n{spec}\n\n"
            f"## Analysis\n```json\n{json.dumps(analysis, indent=2)}\n```\n\n"
            f"## Research\n```json\n{json.dumps(research, indent=2)}\n```\n\n"
            f"## Plan Structure\n```json\n{json.dumps(structure, indent=2)}\n```"
        )
        task_graph_data = await self._run_phase(1, spec, decompose_context)

        # Phase 3: Validate
        validate_context = (
            f"## Task Graph to Validate\n```json\n{json.dumps(task_graph_data, indent=2)}\n```"
        )
        validation = await self._run_phase(2, spec, validate_context)

        # Use validated graph if available, otherwise use decompose output
        final_data = validation.get("task_graph", task_graph_data)
        if not final_data.get("tasks"):
            final_data = task_graph_data

        # Parse into TaskGraph
        task_graph = _parse_task_graph_data(final_data)

        # Validate structural integrity
        errors = task_graph.validate()
        if errors:
            logger.warning("Task graph structural errors: %s", errors)

        # Check wiring metadata (advisory)
        wiring_warnings = task_graph.validate_wiring()
        if wiring_warnings:
            logger.warning("Task graph wiring warnings: %s", wiring_warnings)

        # Create folder structure
        structure_list = final_data.get("project_structure", [])
        _create_folder_structure(self.project_path, structure_list)

        # Store in Flanes
        store_task_graph(repo, task_graph)

        self.event_bus.emit(
            EventType.PLANNING_COMPLETED,
            total_tasks=len(task_graph),
        )
        self.event_bus.emit(EventType.PLAN_CREATED, spec_length=len(spec))

        return task_graph

    async def _run_phase(
        self,
        phase_index: int,
        spec: str,
        user_content: str,
    ) -> dict:
        """Run a single planning phase: emit events, call LLM, parse JSON."""
        name, label, prompt, max_tokens = _PHASES[phase_index]
        total = len(_PHASES)

        self.event_bus.emit(
            EventType.PLANNING_PHASE_STARTED,
            phase=name,
            label=label,
            index=phase_index + 1,
            total=total,
        )

        messages = [Message(role="user", content=user_content)]

        # Emit elapsed-time progress events during the LLM call
        stop_timer = threading.Event()
        timer = threading.Thread(
            target=_elapsed_timer,
            args=(self.event_bus, name, stop_timer),
            daemon=True,
        )
        timer.start()

        try:
            response = await self.registry.complete(
                model_id=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                system=prompt,
                temperature=0.0,
            )
        finally:
            stop_timer.set()
            timer.join(timeout=1)

        result = _extract_json(response.content)
        if result is None:
            logger.warning(
                "Phase '%s' returned no parseable JSON, using empty dict",
                name,
            )
            result = {}

        self.event_bus.emit(
            EventType.PLANNING_PHASE_COMPLETED,
            phase=name,
            index=phase_index + 1,
            total=total,
        )

        return result

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

        from .prompts import PLANNER_INTERVIEW_PROMPT

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


def _elapsed_timer(
    event_bus: EventBus,
    phase_name: str,
    stop_event: threading.Event,
) -> None:
    """Emit elapsed-time events every 10s until stop_event is set."""
    start = time.time()
    while not stop_event.wait(timeout=10):
        elapsed = int(time.time() - start)
        event_bus.emit(
            EventType.PROGRESS_UPDATE,
            phase=phase_name,
            elapsed_seconds=elapsed,
        )


def _scan_project_files(project_path: Path) -> str:
    """Scan the project directory for existing files and return a listing."""
    skip_dirs = {
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
    files: list[str] = []
    try:
        for f in sorted(project_path.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(project_path)
            if any(part in skip_dirs for part in rel.parts):
                continue
            files.append(str(rel).replace("\\", "/"))
    except Exception:
        pass

    if not files:
        return ""
    return "\n".join(f"- {f}" for f in files[:200])


def _parse_task_graph_data(data: dict) -> TaskGraph:
    """Parse a task graph from a dict (already extracted JSON)."""
    tasks_data = data.get("tasks", [])
    tasks = []
    for t in tasks_data:
        tasks.append(
            Task(
                task_id=t["task_id"],
                title=t["title"],
                description=t["description"],
                dependencies=t.get("dependencies", []),
                files_to_create=t.get("files_to_create", []),
                files_to_read=t.get("files_to_read", []),
                estimated_complexity=t.get("estimated_complexity", "medium"),
                evaluators=t.get("evaluators", []),
            )
        )

    graph = TaskGraph(tasks)

    # Extract interface contracts from planner output
    graph.conventions = data.get("conventions", {})
    graph.shared_interfaces = data.get("shared_interfaces", [])

    # Extract protocol-level contracts (backward compatible)
    graph.protocol_contracts = data.get("protocol_contracts", [])
    graph.state_machines = data.get("state_machines", [])
    graph.wiring_map = data.get("wiring_map", [])

    return graph


def _extract_json(content: str) -> dict | None:
    """Extract the JSON object from LLM response content.

    Tries multiple strategies: ```json blocks, generic code blocks,
    brace matching, and finally the raw content.
    """
    import re

    # Strategy 1: ```json code block
    match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2: generic code block containing "tasks" or expected keys
    match = re.search(r"```\s*\n(.*?)```", content, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        if any(key in candidate for key in ('"tasks"', '"objective"', '"valid"', '"phases"')):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    # Strategy 3: brace matching for outermost JSON objects
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


def _create_folder_structure(project_path: Path, structure: list[str]) -> None:
    """Create the project folder structure from a list of paths."""
    for path_str in structure:
        full_path = project_path / path_str
        if path_str.endswith("/"):
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_path.exists():
                full_path.touch()
