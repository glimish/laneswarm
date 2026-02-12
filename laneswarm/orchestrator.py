"""Orchestrator: main loop managing task execution with ThreadPoolExecutor.

The orchestrator:
1. Reads the task graph (from Flanes or direct input)
2. Schedules ready tasks (dependencies met) for parallel execution
3. Spawns coder agents in worker threads
4. Runs reviewer agents on completed work
5. Promotes accepted work into main via integrator agents
6. Reports progress via the event bus
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from flanes.agent_sdk import AgentSession
from flanes.repo import Repository

from .agents.coder import CoderAgent
from .agents.integrator import IntegratorAgent
from .agents.reviewer import ReviewerAgent
from .config import Config
from .events import EventBus, EventType
from .flanes_bridge import create_agent_session, create_lane_for_task, record_pipeline_costs
from .model_selector import (
    select_integrator_model,
    select_model,
    select_reviewer_model,
    select_smoker_model,
)
from .providers import ProviderRegistry
from .task_graph import Task, TaskGraph, TaskStatus

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator for Laneswarm task execution."""

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
        self._repo = Repository.find(project_path)

    def run(self, task_graph: TaskGraph) -> dict:
        """Execute all tasks in the graph.

        Returns a summary dict with counts and costs.
        """
        self.event_bus.emit(EventType.RUN_STARTED, total_tasks=len(task_graph))
        start_time = time.time()

        # Validate structural integrity (missing deps, cycles)
        errors = task_graph.validate()
        if errors:
            raise ValueError(f"Invalid task graph: {'; '.join(errors)}")

        # Log wiring warnings (advisory — don't block execution)
        wiring_warnings = task_graph.validate_wiring()
        if wiring_warnings:
            logger.warning(
                "Task graph wiring warnings (non-blocking): %s",
                "; ".join(wiring_warnings),
            )

        # Main execution loop
        total_tokens_in = 0
        total_tokens_out = 0
        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            active_futures: dict[Future, Task] = {}

            while not task_graph.all_complete():
                # Find ready tasks and submit them
                ready = task_graph.get_ready_tasks()
                for task in ready:
                    if task.task_id not in {t.task_id for t in active_futures.values()}:
                        # Create lane lazily — forks from latest main so
                        # promoted dependency files are included.
                        create_lane_for_task(self._repo, task, self.config)
                        task_graph.mark_in_progress(task.task_id)
                        future = pool.submit(self._execute_task, task, task_graph)
                        active_futures[future] = task
                        self.event_bus.emit(
                            EventType.TASK_STARTED,
                            task_id=task.task_id,
                            model=select_model(task, self.config),
                        )

                if not active_futures:
                    # No tasks ready and no tasks running — check for deadlock
                    pending = [t for t in task_graph.tasks if t.status == TaskStatus.PENDING]
                    if pending:
                        logger.error(
                            "Deadlock: %d tasks pending but none ready. Check dependency graph.",
                            len(pending),
                        )
                        for t in pending:
                            task_graph.mark_failed(t.task_id, "Deadlocked")
                            failed += 1
                    break

                # Wait for at least one task to complete
                done_futures = []
                for future in as_completed(active_futures):
                    done_futures.append(future)
                    break  # Process one at a time to check for new ready tasks

                for future in done_futures:
                    task = active_futures.pop(future)
                    try:
                        result = future.result()
                        task_tokens = result.get("tokens_in", 0) + result.get("tokens_out", 0)
                        task.tokens_used += task_tokens
                        task.wall_time_ms += result.get("wall_time_ms", 0.0)
                        total_tokens_in += result.get("tokens_in", 0)
                        total_tokens_out += result.get("tokens_out", 0)

                        if result.get("success"):
                            completed += 1
                            self.event_bus.emit(
                                EventType.TASK_COMPLETED,
                                task_id=task.task_id,
                                tokens=task_tokens,
                                wall_time_ms=task.wall_time_ms,
                                files_written=task.files_written,
                            )
                        else:
                            error = result.get("error") or "Unknown error"
                            logger.warning(
                                "Task '%s' failed: %s",
                                task.task_id,
                                error,
                            )
                            task_graph.mark_failed(task.task_id, error)
                            if task.status == TaskStatus.PENDING:
                                # Will be retried
                                self.event_bus.emit(
                                    EventType.TASK_RETRYING,
                                    task_id=task.task_id,
                                    retry=task.retries,
                                    error=error,
                                )
                            else:
                                failed += 1
                                self.event_bus.emit(
                                    EventType.TASK_FAILED,
                                    task_id=task.task_id,
                                    error=error,
                                )
                    except Exception as e:
                        logger.error("Task '%s' raised exception: %s", task.task_id, e)
                        task_graph.mark_failed(task.task_id, str(e))
                        failed += 1
                        self.event_bus.emit(
                            EventType.TASK_FAILED,
                            task_id=task.task_id,
                            error=str(e),
                        )

                # Emit progress
                self.event_bus.emit(
                    EventType.PROGRESS_UPDATE,
                    **task_graph.progress(),
                )

        # Post-completion integration validation
        integration_result = None
        if completed > 0:
            integration_result = self._validate_assembled_project(task_graph)

        # Post-completion smoke tests (advisory — results logged but don't block)
        smoke_result = None
        if completed > 0:
            smoke_result = self._run_smoke_tests(task_graph)

        # Persist smoke result so `laneswarm fix` can use it
        if smoke_result is not None:
            self._persist_smoke_result(smoke_result)

        elapsed = time.time() - start_time

        summary = {
            "completed": completed,
            "failed": failed,
            "total": len(task_graph),
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "elapsed_seconds": elapsed,
        }

        if integration_result is not None:
            summary["integration_validation"] = integration_result
        if smoke_result is not None:
            summary["smoke_test"] = smoke_result

        self.event_bus.emit(EventType.RUN_COMPLETED, **summary)
        return summary

    def _validate_assembled_project(
        self,
        task_graph: TaskGraph,
    ) -> dict | None:
        """Run integration validation on the assembled project.

        Called after all tasks complete. Checks compile errors, protocol
        compliance, wiring gaps, and anti-patterns.
        """
        from .verification import validate_assembled_project

        self.event_bus.emit(EventType.INTEGRATION_VALIDATION_STARTED)

        try:
            main_ws = self._repo.workspace_path("main")
            if main_ws is None:
                logger.warning("No main workspace found, skipping integration validation")
                return None

            result = validate_assembled_project(main_ws, task_graph)

            if result.get("passed"):
                self.event_bus.emit(
                    EventType.INTEGRATION_VALIDATION_PASSED,
                    summary=result.get("summary", ""),
                )
            else:
                self.event_bus.emit(
                    EventType.INTEGRATION_VALIDATION_FAILED,
                    summary=result.get("summary", ""),
                    compile_errors=len(result.get("compile_errors", [])),
                    protocol_violations=len(result.get("protocol_violations", [])),
                    wiring_gaps=len(result.get("wiring_gaps", [])),
                    anti_patterns=len(result.get("anti_patterns", [])),
                )

            return result
        except Exception as e:
            logger.warning("Post-completion validation failed: %s", e)
            return None

    def _run_smoke_tests(
        self,
        task_graph: TaskGraph,
    ) -> dict | None:
        """Run runtime smoke tests on the assembled project.

        Called after all tasks complete. Starts the server, hits it with
        HTTP/WS requests, and reports results. Advisory only — failures
        are logged but don't affect the run summary pass/fail.
        """
        from .agents.smoker import SmokerAgent

        try:
            main_ws = self._repo.workspace_path("main")
            if main_ws is None:
                logger.warning("No main workspace found, skipping smoke tests")
                return None

            smoker_model = select_smoker_model(self.config)
            smoker = SmokerAgent(
                repo_path=self.project_path,
                agent_id="smoker",
                agent_type="smoker",
                model_id=smoker_model,
                registry=self.registry,
                event_bus=self.event_bus,
            )

            result = smoker.smoke_test(main_ws, task_graph)
            return result
        except Exception as e:
            logger.warning("Smoke test failed: %s", e)
            return None

    def _persist_smoke_result(self, result: dict) -> None:
        """Save smoke test result to disk for ``laneswarm fix``."""
        import json

        try:
            main_ws = self._repo.workspace_path("main")
            if main_ws is None:
                return
            smoke_file = main_ws / ".laneswarm-smoke.json"
            smoke_file.write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(
                "Failed to persist smoke result: %s",
                e,
            )

    def _execute_task(self, task: Task, task_graph: TaskGraph) -> dict:
        """Execute a single task: code → review → promote.

        Runs in a worker thread. Creates its own Repository instance
        for thread safety.
        """
        task_start = time.time()
        result = self._execute_task_inner(task, task_graph)
        wall_time_ms = (time.time() - task_start) * 1000
        result["wall_time_ms"] = wall_time_ms

        # Record costs on Flanes transitions so flanes history shows them
        if result.get("coder_transition_id"):
            try:
                thread_repo = Repository(self.project_path)
                record_pipeline_costs(
                    repo=thread_repo,
                    coder_transition_id=result.get("coder_transition_id"),
                    reviewer_tokens=result.get("reviewer_tokens", {}),
                    wall_time_ms=wall_time_ms,
                )
            except Exception as e:
                logger.debug("Failed to record pipeline costs: %s", e)

        return result

    def _execute_task_inner(self, task: Task, task_graph: TaskGraph) -> dict:
        """Inner implementation of task execution (code → review → promote)."""
        # Each thread gets its own Repository
        thread_repo = Repository(self.project_path)
        model_id = select_model(task, self.config)

        total_tokens_in = 0
        total_tokens_out = 0

        # Per-step token tracking for Flanes cost recording
        coder_tokens: dict = {"tokens_in": 0, "tokens_out": 0}
        reviewer_tokens: dict = {"tokens_in": 0, "tokens_out": 0}
        integrator_tokens: dict = {"tokens_in": 0, "tokens_out": 0}

        # Step 1: Code
        task.current_phase = "coding"
        task.agent_steps.append(
            {
                "phase": "coding",
                "iteration": task.retries,
                "timestamp": time.time(),
                "summary": f"Starting coder with model {model_id}",
            }
        )

        session = create_agent_session(
            self.project_path,
            task,
            f"coder-{task.task_id}",
            model_id,
        )

        coder = CoderAgent(
            repo_path=self.project_path,
            agent_id=f"coder-{task.task_id}",
            agent_type="coder",
            model_id=model_id,
            registry=self.registry,
            event_bus=self.event_bus,
            lane=task.lane_name,
            workspace=task.lane_name,
        )

        workspace_path = thread_repo.workspace_path(task.lane_name)
        if workspace_path is None:
            return {"success": False, "error": f"No workspace for lane '{task.lane_name}'"}

        with session.work(
            prompt=f"Implement: {task.title}",
            tags=["laneswarm-task", task.task_id],
            auto_accept=False,
        ) as work_ctx:
            code_result = coder.run(task, task_graph, work_ctx.path)

            # Record token usage in the session
            work_ctx.record_tokens(
                tokens_in=code_result.get("tokens_in", 0),
                tokens_out=code_result.get("tokens_out", 0),
            )

        coder_tokens = {
            "tokens_in": code_result.get("tokens_in", 0),
            "tokens_out": code_result.get("tokens_out", 0),
        }
        total_tokens_in += coder_tokens["tokens_in"]
        total_tokens_out += coder_tokens["tokens_out"]

        # Enrich task with coder results for dashboard
        task.files_written = code_result.get("files_written", [])
        task.verification_result = code_result.get("verification")
        task.agent_steps.append(
            {
                "phase": "coding",
                "iteration": task.retries,
                "timestamp": time.time(),
                "summary": (
                    f"Coder finished: {len(task.files_written)} files, "
                    f"{'success' if code_result.get('success') else 'failed'}"
                ),
            }
        )

        # Get the transition ID from the work context
        coder_transition_id = None
        if work_ctx.result:
            coder_transition_id = work_ctx.result.get("transition_id")

        if not code_result.get("success"):
            logger.debug(
                "Task '%s' coder failed: %s",
                task.task_id,
                code_result.get("error") or "Coder produced no files",
            )
            return {
                "success": False,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "error": code_result.get("error") or "Coder produced no files",
            }

        logger.debug(
            "Task '%s' coder succeeded: %d files",
            task.task_id,
            len(code_result.get("files_written", [])),
        )

        # Extract verification summary from coder (coder does internal verification)
        verification_summary = None
        coder_verification = code_result.get("verification")
        if coder_verification:
            verification_summary = coder_verification.get("summary")
            if coder_verification.get("passed"):
                logger.debug("Task '%s' verification passed", task.task_id)
            else:
                logger.debug(
                    "Task '%s' verification issues: %s",
                    task.task_id,
                    (verification_summary or "")[:200],
                )

        # Step 2: Review
        task.current_phase = "reviewing"
        task.agent_steps.append(
            {
                "phase": "reviewing",
                "iteration": task.retries,
                "timestamp": time.time(),
                "summary": "Starting reviewer",
            }
        )

        reviewer_model = select_reviewer_model(self.config)
        reviewer = ReviewerAgent(
            repo_path=self.project_path,
            agent_id=f"reviewer-{task.task_id}",
            agent_type="reviewer",
            model_id=reviewer_model,
            registry=self.registry,
            event_bus=self.event_bus,
        )

        review_result = reviewer.review(
            task,
            thread_repo,
            coder_transition_id,
            verification_summary=verification_summary,
            task_graph=task_graph,
        )
        reviewer_tokens = {
            "tokens_in": review_result.get("tokens_in", 0),
            "tokens_out": review_result.get("tokens_out", 0),
        }
        total_tokens_in += reviewer_tokens["tokens_in"]
        total_tokens_out += reviewer_tokens["tokens_out"]

        # Enrich task with review results for dashboard
        task.review_summary = review_result.get("summary", "")
        accepted = review_result.get("accepted", False)
        task.agent_steps.append(
            {
                "phase": "reviewing",
                "iteration": task.retries,
                "timestamp": time.time(),
                "summary": f"Review {'accepted' if accepted else 'rejected'}: {task.review_summary[:100]}",
            }
        )

        if not accepted:
            # Store reviewer feedback so the coder can use it on retry
            llm_feedback = review_result.get("summary", "")
            task.last_review_feedback = llm_feedback[:500]
            logger.debug(
                "Task '%s' review rejected: %s",
                task.task_id,
                llm_feedback[:300],
            )
            return {
                "success": False,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "error": f"Review rejected: {llm_feedback[:200]}",
                "coder_transition_id": coder_transition_id,
                "reviewer_tokens": reviewer_tokens,
            }

        logger.debug("Task '%s' review accepted", task.task_id)

        # Step 3: Promote to main
        task.current_phase = "promoting"
        task.agent_steps.append(
            {
                "phase": "promoting",
                "iteration": task.retries,
                "timestamp": time.time(),
                "summary": "Starting promote to main",
            }
        )

        integrator_model = select_integrator_model(self.config)
        integrator = IntegratorAgent(
            repo_path=self.project_path,
            agent_id=f"integrator-{task.task_id}",
            agent_type="integrator",
            model_id=integrator_model,
            registry=self.registry,
            event_bus=self.event_bus,
        )

        promote_result = integrator.promote(task, thread_repo)
        integrator_tokens = {
            "tokens_in": promote_result.get("tokens_in", 0),
            "tokens_out": promote_result.get("tokens_out", 0),
        }
        total_tokens_in += integrator_tokens["tokens_in"]
        total_tokens_out += integrator_tokens["tokens_out"]

        promote_transition_id = promote_result.get("transition_id")

        if promote_result.get("success"):
            task.current_phase = "completed"
            task.agent_steps.append(
                {
                    "phase": "promoting",
                    "iteration": task.retries,
                    "timestamp": time.time(),
                    "summary": "Promoted to main successfully",
                }
            )
            logger.debug("Task '%s' promoted to main", task.task_id)
            task_graph.mark_completed(task.task_id, coder_transition_id)
            return {
                "success": True,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "coder_transition_id": coder_transition_id,
                "promote_transition_id": promote_transition_id,
                "coder_tokens": coder_tokens,
                "reviewer_tokens": reviewer_tokens,
                "integrator_tokens": integrator_tokens,
            }
        else:
            task.agent_steps.append(
                {
                    "phase": "promoting",
                    "iteration": task.retries,
                    "timestamp": time.time(),
                    "summary": f"Promote failed: {promote_result.get('error', 'conflicts')}",
                }
            )
            logger.debug(
                "Task '%s' promote failed: %s",
                task.task_id,
                promote_result.get("error", "conflicts"),
            )
            return {
                "success": False,
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "error": f"Promote failed: {promote_result.get('error', 'conflicts')}",
                "coder_transition_id": coder_transition_id,
                "reviewer_tokens": reviewer_tokens,
            }
