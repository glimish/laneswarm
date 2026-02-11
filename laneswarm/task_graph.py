"""Task graph: DAG of tasks with dependency tracking and topological scheduling.

Tasks represent units of work that agents implement. Each task maps to a
Flanes lane + workspace. The TaskGraph manages the dependency DAG, determines
which tasks are ready for execution, and tracks completion status.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    """Status of a task in the graph."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A single task in the dependency graph.

    Each task maps to one Flanes lane and workspace.
    """

    task_id: str
    title: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    files_to_create: list[str] = field(default_factory=list)
    files_to_read: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"  # "low", "medium", "high"
    model_override: str | None = None  # Override config model for this task
    evaluators: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    max_retries: int = 3
    lane_name: str = ""
    # Populated after execution
    transition_id: str | None = None
    error_message: str | None = None
    last_review_feedback: str | None = None  # Reviewer feedback for retry
    tokens_used: int = 0
    wall_time_ms: float = 0.0
    # Dashboard-enriched fields (populated during execution)
    current_phase: str = ""  # "coding", "reviewing", "promoting", "completed"
    agent_steps: list[dict] = field(default_factory=list)  # [{phase, iteration, timestamp, summary}]
    files_written: list[str] = field(default_factory=list)  # Actual files produced by coder
    verification_result: dict | None = None  # Last verification output
    review_summary: str = ""  # Reviewer's LLM feedback (accepted/rejected + reason)

    def __post_init__(self) -> None:
        if not self.lane_name:
            self.lane_name = f"task-{self.task_id}"

    def to_dict(self) -> dict:
        """Serialize to dict for storage as Flanes intent metadata."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "dependencies": self.dependencies,
            "files_to_create": self.files_to_create,
            "files_to_read": self.files_to_read,
            "estimated_complexity": self.estimated_complexity,
            "model_override": self.model_override,
            "evaluators": self.evaluators,
            "status": self.status.value,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "lane_name": self.lane_name,
            "transition_id": self.transition_id,
            "error_message": self.error_message,
            "last_review_feedback": self.last_review_feedback,
            "tokens_used": self.tokens_used,
            "wall_time_ms": self.wall_time_ms,
            "current_phase": self.current_phase,
            "agent_steps": self.agent_steps,
            "files_written": self.files_written,
            "verification_result": self.verification_result,
            "review_summary": self.review_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        """Deserialize from dict."""
        return cls(
            task_id=data["task_id"],
            title=data["title"],
            description=data["description"],
            dependencies=data.get("dependencies", []),
            files_to_create=data.get("files_to_create", []),
            files_to_read=data.get("files_to_read", []),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            model_override=data.get("model_override"),
            evaluators=data.get("evaluators", []),
            status=TaskStatus(data.get("status", "pending")),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
            lane_name=data.get("lane_name", ""),
            transition_id=data.get("transition_id"),
            error_message=data.get("error_message"),
            last_review_feedback=data.get("last_review_feedback"),
            tokens_used=data.get("tokens_used", 0),
            wall_time_ms=data.get("wall_time_ms", 0.0),
            current_phase=data.get("current_phase", ""),
            agent_steps=data.get("agent_steps", []),
            files_written=data.get("files_written", []),
            verification_result=data.get("verification_result"),
            review_summary=data.get("review_summary", ""),
        )


class TaskGraph:
    """Directed acyclic graph of tasks with dependency tracking.

    Provides topological scheduling: tasks become ready when all their
    dependencies are completed.
    """

    def __init__(self, tasks: list[Task] | None = None) -> None:
        self._tasks: dict[str, Task] = {}
        # Planner-generated interface contracts (injected into every coder prompt)
        self.conventions: dict[str, str] = {}
        self.shared_interfaces: list[dict] = []
        # Protocol-level contracts for cross-task communication
        # Each entry: {name, direction, producer_task, consumer_tasks, data_shape, channel}
        self.protocol_contracts: list[dict] = []
        # State machine definitions for stateful components
        # Each entry: {name, module, states, transitions, relevant_tasks}
        self.state_machines: list[dict] = []
        # Producer→consumer wiring map for cross-task artifacts
        # Each entry: {artifact, artifact_type, produced_by, consumed_by, defined_in, used_in}
        self.wiring_map: list[dict] = []
        if tasks:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: Task) -> None:
        """Add a task to the graph."""
        if task.task_id in self._tasks:
            raise ValueError(f"Task '{task.task_id}' already exists")
        self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Task:
        """Get a task by ID."""
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        return self._tasks[task_id]

    @property
    def tasks(self) -> list[Task]:
        """All tasks in the graph."""
        return list(self._tasks.values())

    @property
    def task_ids(self) -> list[str]:
        """All task IDs."""
        return list(self._tasks.keys())

    def validate(self) -> list[str]:
        """Validate structural integrity of the graph.

        Returns a list of errors (empty if valid).  Only checks
        structural issues (missing dependencies, cycles) that would
        prevent execution.  Wiring metadata issues (bad task IDs in
        protocol_contracts / wiring_map) are advisory — call
        ``validate_wiring()`` separately to get those warnings.
        """
        errors = []

        # Check for missing dependencies
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in self._tasks:
                    errors.append(
                        f"Task '{task.task_id}' depends on '{dep_id}' which doesn't exist"
                    )

        # Check for cycles
        if not errors:
            cycle = self._detect_cycle()
            if cycle:
                errors.append(f"Dependency cycle detected: {' -> '.join(cycle)}")

        return errors

    def validate_wiring(self) -> list[str]:
        """Validate protocol contracts and wiring map references.

        Checks that all task IDs referenced in contracts exist in the graph.
        Returns a list of error messages.
        """
        errors = []

        for pc in self.protocol_contracts:
            name = pc.get("name", "?")
            producer = pc.get("producer_task", "")
            if producer and producer not in self._tasks:
                errors.append(
                    f"Protocol contract '{name}' references non-existent "
                    f"producer task '{producer}'"
                )
            for consumer in pc.get("consumer_tasks", []):
                if consumer not in self._tasks:
                    errors.append(
                        f"Protocol contract '{name}' references non-existent "
                        f"consumer task '{consumer}'"
                    )

        for sm in self.state_machines:
            name = sm.get("name", "?")
            for task_id in sm.get("relevant_tasks", []):
                if task_id not in self._tasks:
                    errors.append(
                        f"State machine '{name}' references non-existent "
                        f"task '{task_id}'"
                    )

        for wm in self.wiring_map:
            artifact = wm.get("artifact", "?")
            produced_by = wm.get("produced_by", "")
            if produced_by and produced_by not in self._tasks:
                errors.append(
                    f"Wiring map entry '{artifact}' references non-existent "
                    f"producer task '{produced_by}'"
                )
            for consumer in wm.get("consumed_by", []):
                if consumer not in self._tasks:
                    errors.append(
                        f"Wiring map entry '{artifact}' references non-existent "
                        f"consumer task '{consumer}'"
                    )

        return errors

    def _detect_cycle(self) -> list[str] | None:
        """Detect cycles using DFS. Returns cycle path or None."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in self._tasks}
        parent: dict[str, str | None] = {tid: None for tid in self._tasks}

        def dfs(node: str) -> list[str] | None:
            color[node] = GRAY
            for dep_id in self._tasks[node].dependencies:
                if dep_id not in self._tasks:
                    continue
                if color[dep_id] == GRAY:
                    # Found cycle — reconstruct path
                    cycle = [dep_id, node]
                    current = node
                    while parent[current] is not None and parent[current] != dep_id:
                        current = parent[current]  # type: ignore[assignment]
                        cycle.append(current)
                    cycle.reverse()
                    return cycle
                if color[dep_id] == WHITE:
                    parent[dep_id] = node
                    result = dfs(dep_id)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for tid in self._tasks:
            if color[tid] == WHITE:
                result = dfs(tid)
                if result:
                    return result
        return None

    def get_ready_tasks(self) -> list[Task]:
        """Return tasks whose dependencies are all completed and status is pending."""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self._tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self._tasks
            )
            if deps_met:
                ready.append(task)
        return ready

    def mark_in_progress(self, task_id: str) -> None:
        """Mark a task as in progress."""
        task = self.get_task(task_id)
        task.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, task_id: str, transition_id: str | None = None) -> None:
        """Mark a task as completed."""
        task = self.get_task(task_id)
        task.status = TaskStatus.COMPLETED
        if transition_id:
            task.transition_id = transition_id

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed."""
        task = self.get_task(task_id)
        task.retries += 1
        if task.retries >= task.max_retries:
            task.status = TaskStatus.FAILED
            task.error_message = error
        else:
            # Reset to pending for retry
            task.status = TaskStatus.PENDING
            task.error_message = error

    def all_complete(self) -> bool:
        """Check if all tasks are completed or failed."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            for task in self._tasks.values()
        )

    def progress(self) -> dict[str, int]:
        """Return progress summary."""
        counts: dict[str, int] = {s.value: 0 for s in TaskStatus}
        for task in self._tasks.values():
            counts[task.status.value] += 1
        counts["total"] = len(self._tasks)
        return counts

    def topological_order(self) -> list[str]:
        """Return task IDs in topological order (dependencies first)."""
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id in self._tasks:
                    in_degree[task.task_id] += 1  # task depends on dep_id

        # Kahn's algorithm
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []

        # Build reverse adjacency: dep_id -> [tasks that depend on it]
        dependents: dict[str, list[str]] = {tid: [] for tid in self._tasks}
        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id in self._tasks:
                    dependents[dep_id].append(task.task_id)

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            order.append(node)
            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return order

    def to_json(self) -> str:
        """Serialize entire graph to JSON."""
        data = {
            "conventions": self.conventions,
            "shared_interfaces": self.shared_interfaces,
            "tasks": [task.to_dict() for task in self._tasks.values()],
        }
        # Include protocol-level contracts if present
        if self.protocol_contracts:
            data["protocol_contracts"] = self.protocol_contracts
        if self.state_machines:
            data["state_machines"] = self.state_machines
        if self.wiring_map:
            data["wiring_map"] = self.wiring_map
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, data: str) -> TaskGraph:
        """Deserialize graph from JSON."""
        parsed = json.loads(data)
        tasks = [Task.from_dict(t) for t in parsed["tasks"]]
        graph = cls(tasks)
        graph.conventions = parsed.get("conventions", {})
        graph.shared_interfaces = parsed.get("shared_interfaces", [])
        graph.protocol_contracts = parsed.get("protocol_contracts", [])
        graph.state_machines = parsed.get("state_machines", [])
        graph.wiring_map = parsed.get("wiring_map", [])
        return graph

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks
