"""Bridge between Laneswarm task graph and Flanes version control.

Provides helpers to:
- Create Flanes lanes and workspaces from a task graph
- Build focused prompts from workspace state (only relevant files)
- Aggregate costs across transitions
- Store/retrieve task graphs as Flanes intents
- Record per-lane costs (coder + reviewer + integrator)
"""

from __future__ import annotations

import logging
from pathlib import Path

from flanes.agent_sdk import AgentSession
from flanes.repo import Repository
from flanes.state import AgentIdentity, CostRecord

from .config import BudgetConfig, Config
from .task_graph import Task, TaskGraph

logger = logging.getLogger(__name__)

PLAN_TAG = "laneswarm-plan"
TASK_TAG = "laneswarm-task"


def init_project(project_path: Path, config: Config) -> Repository:
    """Initialize a Flanes repository for a new project.

    Creates the repo if it doesn't exist, or opens it if it does.
    """
    flanes_dir = project_path / ".flanes"
    if flanes_dir.exists():
        return Repository.find(project_path)
    return Repository.init(project_path)


TASK_GRAPH_FILENAME = ".laneswarm-tasks.json"


def store_task_graph(
    repo: Repository,
    task_graph: TaskGraph,
    agent_id: str = "planner",
) -> str:
    """Store the task graph as a file in the main workspace and commit.

    Writes the task graph JSON to a well-known file, then commits via
    quick_commit so Flanes tracks the change. Returns the transition ID.
    """
    agent = AgentIdentity(
        agent_id=agent_id,
        agent_type="planner",
        model="laneswarm",
    )

    graph_json = task_graph.to_json()

    # Write task graph to a file in the main workspace
    ws_path = repo.workspace_path("main")
    if ws_path is None:
        raise ValueError("Main workspace not found. Initialize the project first.")
    task_graph_file = ws_path / TASK_GRAPH_FILENAME
    task_graph_file.write_text(graph_json, encoding="utf-8")

    return repo.quick_commit(
        workspace="main",
        prompt=f"Laneswarm task graph: {len(task_graph)} tasks",
        agent=agent,
        tags=[PLAN_TAG],
        auto_accept=True,
        evaluator="auto",
    ).get("transition_id", "")


def load_task_graph(repo: Repository) -> TaskGraph | None:
    """Load the task graph from the main workspace file.

    Reads the well-known task graph JSON file from the main workspace.
    """
    ws_path = repo.workspace_path("main")
    if ws_path is None:
        return None

    task_graph_file = ws_path / TASK_GRAPH_FILENAME
    if not task_graph_file.exists():
        return None

    try:
        graph_json = task_graph_file.read_text()
        return TaskGraph.from_json(graph_json)
    except Exception:
        logger.warning("Failed to load task graph from %s", task_graph_file, exc_info=True)
        return None


def create_lanes_for_tasks(
    repo: Repository,
    task_graph: TaskGraph,
    config: Config,
) -> None:
    """Create a Flanes lane and workspace for each task in the graph.

    Each lane is forked from the current main head. Budgets are set
    based on task complexity.

    NOTE: This creates all lanes eagerly from the *current* main head.
    For tasks with dependencies, prefer create_lane_for_task() which
    forks from the latest main (after dependencies are promoted).
    """
    for task in task_graph.tasks:
        create_lane_for_task(repo, task, config)


def create_lane_for_task(
    repo: Repository,
    task: Task,
    config: Config,
) -> None:
    """Create a Flanes lane and workspace for a single task.

    Forks from the current main head, so call this AFTER dependencies
    have been promoted to main to get their files in the workspace.
    """
    main_head = repo.head()
    if main_head is None:
        raise ValueError("Main lane has no head state. Initialize the project first.")

    lane_name = task.lane_name

    # Skip if lane already exists
    lanes = repo.lanes()
    if any(l.get("name") == lane_name for l in lanes):
        logger.info("Lane '%s' already exists, skipping", lane_name)
        return

    # Create lane forked from main
    repo.create_lane(lane_name, base=main_head)

    # Set budget based on complexity
    budget = _budget_for_complexity(task.estimated_complexity, config.default_budget)
    try:
        repo.set_budget(
            lane_name,
            max_tokens_in=budget.max_tokens_in,
            max_tokens_out=budget.max_tokens_out,
            max_api_calls=budget.max_api_calls,
            max_wall_time_ms=budget.max_wall_time_ms,
        )
    except Exception:
        logger.debug("Budget setting not available, skipping", exc_info=True)

    logger.info(
        "Created lane '%s' for task '%s' (complexity: %s)",
        lane_name, task.task_id, task.estimated_complexity,
    )


def _budget_for_complexity(complexity: str, default: BudgetConfig) -> BudgetConfig:
    """Scale budget based on task complexity."""
    multipliers = {"low": 0.5, "medium": 1.0, "high": 2.0}
    mult = multipliers.get(complexity, 1.0)

    return BudgetConfig(
        max_tokens_in=int(default.max_tokens_in * mult),
        max_tokens_out=int(default.max_tokens_out * mult),
        max_api_calls=int(default.max_api_calls * mult),
        max_wall_time_ms=int(default.max_wall_time_ms * mult),
    )


def create_agent_session(
    repo_path: Path,
    task: Task,
    agent_id: str,
    model: str,
) -> AgentSession:
    """Create a Flanes AgentSession for a task.

    The session operates on the task's lane and workspace.
    """
    return AgentSession(
        repo_path=repo_path,
        agent_id=agent_id,
        agent_type="coder",
        model=model,
        lane=task.lane_name,
        workspace=task.lane_name,
    )


def build_focused_prompt(
    workspace_path: Path,
    task: Task,
    task_graph: TaskGraph,
    max_file_size: int = 50_000,
    max_auto_include_size: int = 2000,
) -> str:
    """Build a focused prompt for a coder agent.

    Includes:
    - Interface contracts and conventions from the planner
    - Task description and review feedback (if retrying)
    - Actual file contents from dependency tasks (read from workspace)
    - Small project files auto-included for context
    - Directory structure for orientation
    """
    parts = []

    # --- Section 1: Interface contracts (always first) ---
    if task_graph.conventions or task_graph.shared_interfaces:
        contract_parts = []
        if task_graph.conventions:
            contract_parts.append("### Naming Conventions")
            for key, value in task_graph.conventions.items():
                contract_parts.append(f"- **{key}**: {value}")
        if task_graph.shared_interfaces:
            contract_parts.append("\n### Shared Interfaces")
            contract_parts.append(
                "You MUST use these EXACT names and signatures when referencing "
                "shared symbols. Do NOT invent your own names."
            )
            for iface in task_graph.shared_interfaces:
                contract_parts.append(
                    f"- **{iface.get('name', '?')}** ({iface.get('type', '')}) "
                    f"in `{iface.get('module', '')}`: "
                    f"`{iface.get('signature', '')}`"
                )

        # Protocol contracts (filtered to this task)
        if task_graph.protocol_contracts:
            relevant_contracts = [
                c for c in task_graph.protocol_contracts
                if c.get("producer_task") == task.task_id
                or task.task_id in c.get("consumer_tasks", [])
            ]
            if relevant_contracts:
                contract_parts.append("\n### Protocol Contracts (Your Task)")
                contract_parts.append(
                    "You MUST use these EXACT message types and data shapes. "
                    "Do NOT invent your own message names or change the data "
                    "structure."
                )
                for pc in relevant_contracts:
                    role = (
                        "PRODUCE"
                        if pc.get("producer_task") == task.task_id
                        else "CONSUME"
                    )
                    channel = pc.get("channel", "unknown")
                    data_shape = pc.get("data_shape", {})
                    contract_parts.append(
                        f"- **`{pc['name']}`** [{role}] via {channel}: "
                        f"`{data_shape}`"
                    )

        # State machines (filtered to this task)
        if task_graph.state_machines:
            relevant_sms = [
                sm for sm in task_graph.state_machines
                if task.task_id in sm.get("relevant_tasks", [])
            ]
            if relevant_sms:
                contract_parts.append("\n### State Machines")
                contract_parts.append(
                    "Your code must respect these state definitions and "
                    "transitions exactly."
                )
                for sm in relevant_sms:
                    states_str = ", ".join(
                        f"`{s}`" for s in sm.get("states", [])
                    )
                    contract_parts.append(
                        f"- **{sm['name']}** in "
                        f"`{sm.get('module', '')}`: "
                        f"states=[{states_str}]"
                    )
                    for tr in sm.get("transitions", []):
                        contract_parts.append(
                            f"  - `{tr.get('from', '?')}` → "
                            f"`{tr.get('to', '?')}` on "
                            f"{tr.get('trigger', '?')} "
                            f"(message: `{tr.get('message_type', 'N/A')}`)"
                        )

        parts.append(
            "## CRITICAL: Interface Contracts\n\n"
            "These conventions were defined by the project planner. "
            "ALL agents must follow them exactly.\n\n"
            + "\n".join(contract_parts)
        )

    # --- Section 2: Task description ---
    parts.append(f"## Task: {task.title}\n\n{task.description}")

    # Include review feedback from previous attempt (if retrying)
    if task.retries > 0 and task.last_review_feedback:
        parts.append(
            f"## IMPORTANT: Previous Attempt Was Rejected (attempt {task.retries})\n\n"
            f"Your previous implementation was rejected by the reviewer. "
            f"You MUST fix the issues described below:\n\n"
            f"{task.last_review_feedback}\n\n"
            f"Make sure to address ALL of the reviewer's concerns in this attempt."
        )

    # Files to create
    if task.files_to_create:
        parts.append(
            "## Files to Create\n" + "\n".join(f"- `{f}`" for f in task.files_to_create)
        )

    # --- Section 3: Dependency task outputs (actual file contents) ---
    if task.dependencies:
        dep_file_parts = []
        seen_files: set[str] = set()
        for dep_id in task.dependencies:
            try:
                dep_task = task_graph.get_task(dep_id)
                dep_file_parts.append(
                    f"### Dependency: {dep_task.title} ({dep_task.task_id})"
                )
                for file_path in dep_task.files_to_create:
                    if file_path in seen_files:
                        continue
                    seen_files.add(file_path)
                    full_path = workspace_path / file_path
                    if full_path.exists() and full_path.is_file():
                        content = full_path.read_text(errors="replace")
                        if len(content) > max_file_size:
                            content = content[:max_file_size] + "\n... (truncated)"
                        lang = _lang_for_ext(full_path.suffix)
                        dep_file_parts.append(
                            f"`{file_path}`:\n```{lang}\n{content}\n```"
                        )
                    else:
                        dep_file_parts.append(
                            f"`{file_path}`: *(not yet available in workspace)*"
                        )
            except KeyError:
                dep_file_parts.append(f"- {dep_id} (details not available)")
        if dep_file_parts:
            parts.append(
                "## Code From Completed Dependencies\n\n"
                "These files were created by dependency tasks and are already "
                "in the workspace. Use the EXACT names and patterns shown.\n\n"
                + "\n\n".join(dep_file_parts)
            )

    # --- Section 4: Explicitly requested files ---
    if task.files_to_read:
        file_parts = []
        for file_path in task.files_to_read:
            full_path = workspace_path / file_path
            if full_path.exists() and full_path.is_file():
                content = full_path.read_text(errors="replace")
                if len(content) > max_file_size:
                    content = content[:max_file_size] + "\n... (truncated)"
                lang = _lang_for_ext(full_path.suffix)
                file_parts.append(f"### `{file_path}`\n```{lang}\n{content}\n```")
            else:
                file_parts.append(f"### `{file_path}`\n(file not found)")
        if file_parts:
            parts.append("## Relevant Files\n" + "\n\n".join(file_parts))

    # --- Section 5: Auto-include small project files ---
    already_included = set(task.files_to_read) | set(task.files_to_create)
    if task.dependencies:
        for dep_id in task.dependencies:
            try:
                dep_task = task_graph.get_task(dep_id)
                already_included.update(dep_task.files_to_create)
            except KeyError:
                pass
    auto_parts = _auto_include_small_files(
        workspace_path, already_included, max_auto_include_size
    )
    if auto_parts:
        parts.append(
            "## Additional Project Context\n\n"
            "These small files from the workspace are included for reference.\n\n"
            + "\n\n".join(auto_parts)
        )

    # --- Section 6: Directory structure ---
    parts.append(
        "## Project Structure\n```\n" + _dir_tree(workspace_path) + "\n```"
    )

    return "\n\n".join(parts)


def _lang_for_ext(ext: str) -> str:
    """Map file extension to markdown language tag."""
    return {
        ".py": "python", ".toml": "toml", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".js": "javascript",
        ".ts": "typescript", ".html": "html", ".css": "css",
        ".sql": "sql", ".sh": "bash", ".md": "markdown",
    }.get(ext.lower(), ext.lstrip("."))


def _auto_include_small_files(
    workspace_path: Path,
    already_included: set[str],
    max_size: int = 2000,
) -> list[str]:
    """Auto-include small .py files from the workspace as extra context.

    Targets __init__.py files, models.py, schemas.py, config.py, and similar
    interface-defining files. Skips files already included in the prompt.
    """
    priority_names = {
        "__init__.py", "models.py", "schemas.py", "config.py",
        "extensions.py", "types.py", "constants.py", "app.py",
        "database.py",
    }

    results = []
    try:
        for py_file in sorted(workspace_path.rglob("*.py")):
            rel = py_file.relative_to(workspace_path)
            rel_str = str(rel).replace("\\", "/")

            # Skip hidden dirs, __pycache__, venv, etc.
            if any(
                part.startswith(".") or part in {
                    "__pycache__", "venv", ".venv", "node_modules", ".env",
                }
                for part in rel.parts
            ):
                continue
            if rel_str in already_included:
                continue
            if py_file.name not in priority_names:
                continue
            if not py_file.is_file():
                continue
            content = py_file.read_text(errors="replace")
            if len(content) > max_size or not content.strip():
                continue
            results.append(f"`{rel_str}`:\n```python\n{content}\n```")
    except Exception:
        pass  # Filesystem errors should not break prompt building

    return results


def _dir_tree(path: Path, prefix: str = "", max_depth: int = 4, depth: int = 0) -> str:
    """Generate a simple directory tree string."""
    if depth >= max_depth:
        return prefix + "..."

    lines = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return prefix + "(permission denied)"

    # Filter out hidden and ignored dirs
    skip = {".flanes", ".git", "__pycache__", "node_modules", ".venv", "venv"}
    entries = [e for e in entries if e.name not in skip]

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        lines.append(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            subtree = _dir_tree(entry, prefix + extension, max_depth, depth + 1)
            if subtree:
                lines.append(subtree)

    return "\n".join(lines)


def aggregate_costs(
    repo: Repository,
    lane: str | None = None,
    task_graph: TaskGraph | None = None,
) -> dict:
    """Aggregate costs across transitions.

    If *task_graph* is provided, sums costs from all task lanes **and**
    main. Otherwise falls back to the specified *lane* (default main).
    """
    total_tokens_in = 0
    total_tokens_out = 0
    total_api_calls = 0
    total_wall_time_ms = 0.0

    lanes_to_check: list[str] = []
    if task_graph:
        lanes_to_check = [t.lane_name for t in task_graph.tasks]
        lanes_to_check.append("main")
    else:
        lanes_to_check = [lane or "main"]

    seen_ids: set[str] = set()
    for ln in lanes_to_check:
        try:
            history = repo.history(lane=ln, limit=1000)
        except Exception:
            continue
        for transition in history:
            tid = transition.get("id", "")
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            cost = transition.get("cost", {})
            total_tokens_in += cost.get("tokens_in", 0)
            total_tokens_out += cost.get("tokens_out", 0)
            total_api_calls += cost.get("api_calls", 0)
            total_wall_time_ms += cost.get("wall_time_ms", 0.0)

    return {
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "api_calls": total_api_calls,
        "wall_time_ms": total_wall_time_ms,
        "wall_time_s": total_wall_time_ms / 1000.0,
    }


def record_pipeline_costs(
    repo: Repository,
    coder_transition_id: str | None,
    reviewer_tokens: dict,
    wall_time_ms: float,
) -> None:
    """Record the full pipeline costs on the Flanes transition.

    Adds reviewer token usage to the coder's transition on the task
    lane (coder tokens are already recorded by ``session.work()``).
    Integrator costs are recorded separately via the ``cost`` parameter
    on ``repo.promote()``.

    Uses the first-class ``repo.update_transition_cost()`` API.
    """
    if not coder_transition_id:
        return

    try:
        # Merge reviewer tokens into the coder's transition.
        reviewer_in = reviewer_tokens.get("tokens_in", 0)
        reviewer_out = reviewer_tokens.get("tokens_out", 0)
        if reviewer_in or reviewer_out:
            repo.update_transition_cost(
                coder_transition_id,
                CostRecord(
                    tokens_in=reviewer_in,
                    tokens_out=reviewer_out,
                    wall_time_ms=wall_time_ms,
                    api_calls=1,
                ),
                merge=True,
            )
        elif wall_time_ms > 0:
            # No reviewer tokens, but still record pipeline wall time
            repo.update_transition_cost(
                coder_transition_id,
                CostRecord(wall_time_ms=wall_time_ms),
                merge=True,
            )
    except Exception:
        logger.warning(
            "Failed to record pipeline costs on transition '%s'",
            coder_transition_id, exc_info=True,
        )
