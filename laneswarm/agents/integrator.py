"""Integrator agent: promotes feature lanes into main.

The integrator:
1. Calls repo.promote() to integrate a task's work into main
2. If clean: accepts the promotion
3. If conflicts: invokes the LLM to resolve them
"""

from __future__ import annotations

import logging
from pathlib import Path

from flanes.repo import Repository
from flanes.state import AgentIdentity, CostRecord

from ..events import EventType
from ..prompts import INTEGRATOR_SYSTEM_PROMPT
from ..providers import Message
from ..task_graph import Task
from .base import BaseAgent

logger = logging.getLogger(__name__)


class IntegratorAgent(BaseAgent):
    """Agent that promotes feature lanes into main and resolves conflicts."""

    def promote(
        self,
        task: Task,
        repo: Repository,
        target_lane: str = "main",
        auto_accept: bool = True,
    ) -> dict:
        """Promote a task's lane into the target lane.

        Returns:
        - success: bool
        - status: str ("promoted", "conflicts", "resolved", "failed")
        - conflicts: list[dict] | None
        - tokens_in: int
        - tokens_out: int
        """
        self.emit(EventType.PROMOTE_STARTED, task_id=task.task_id)

        agent = AgentIdentity(
            agent_id=self.agent_id,
            agent_type="integrator",
            model=self.model_id,
        )

        try:
            result = repo.promote(
                workspace=task.lane_name,
                target_lane=target_lane,
                prompt=f"Promote task '{task.title}' ({task.task_id}) to {target_lane}",
                agent=agent,
                tags=["laneswarm-promote", task.task_id],
                auto_accept=auto_accept,
                force=False,
            )
        except Exception as e:
            self.log.error("Promote failed for task '%s': %s", task.task_id, e)
            self.emit(EventType.PROMOTE_CONFLICT, task_id=task.task_id, error=str(e))
            return {
                "success": False,
                "status": "failed",
                "conflicts": None,
                "tokens_in": 0,
                "tokens_out": 0,
                "error": str(e),
            }

        # Check result
        if result.get("has_conflicts"):
            conflicts = result.get("conflicts", [])
            self.emit(
                EventType.PROMOTE_CONFLICT,
                task_id=task.task_id,
                conflict_count=len(conflicts),
            )

            # Try to resolve conflicts
            resolution = self._resolve_conflicts(task, repo, result, target_lane)
            return resolution

        # Materialize the updated main workspace to disk
        try:
            repo.workspace_update(target_lane)
        except Exception as e:
            self.log.warning("Failed to materialize main workspace: %s", e)

        # Clean promote
        self.emit(EventType.PROMOTE_COMPLETED, task_id=task.task_id, status="clean")

        return {
            "success": True,
            "status": "promoted",
            "conflicts": None,
            "tokens_in": 0,
            "tokens_out": 0,
            "transition_id": result.get("transition_id"),
        }

    def _resolve_conflicts(
        self,
        task: Task,
        repo: Repository,
        conflict_result: dict,
        target_lane: str,
    ) -> dict:
        """Attempt to resolve conflicts using the LLM.

        Strategy: Force-promote (agent's version wins) when the LLM
        confirms the agent's changes are correct for the task.
        """
        conflicts = conflict_result.get("conflicts", [])
        conflict_paths = [c["path"] for c in conflicts]

        self.log.info(
            "Resolving %d conflicts for task '%s': %s",
            len(conflicts), task.task_id, conflict_paths,
        )

        # Ask LLM whether to force promote
        conflict_desc = "\n".join(
            f"- {c['path']}: lane={c['lane_action']}, target={c['target_action']}"
            for c in conflicts
        )

        prompt = (
            f"## Task: {task.title}\n\n"
            f"{task.description}\n\n"
            f"## Conflicts\n"
            f"The following files were modified on both the task lane "
            f"and the {target_lane} lane:\n\n"
            f"{conflict_desc}\n\n"
            f"Should we force-promote (keep the task's version of these files) "
            f"or abort? The task lane contains the implementation of this specific "
            f"task. Respond with FORCE or ABORT followed by a brief explanation."
        )

        try:
            response = self.call_llm_sync(
                messages=[Message(role="user", content=prompt)],
                max_tokens=1024,
                system=INTEGRATOR_SYSTEM_PROMPT,
                temperature=0.0,
            )
        except Exception as e:
            self.log.error("LLM conflict resolution failed: %s", e)
            return {
                "success": False,
                "status": "conflicts",
                "conflicts": conflicts,
                "tokens_in": 0,
                "tokens_out": 0,
                "error": str(e),
            }

        should_force = response.content.upper().startswith("FORCE")

        if should_force:
            # Force promote
            agent = AgentIdentity(
                agent_id=self.agent_id,
                agent_type="integrator",
                model=self.model_id,
            )
            integrator_cost = CostRecord(
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                api_calls=1,
            )
            try:
                force_result = repo.promote(
                    workspace=task.lane_name,
                    target_lane=target_lane,
                    prompt=f"Force-promote task '{task.title}' (conflicts resolved by LLM)",
                    agent=agent,
                    tags=["laneswarm-promote", "force-resolved", task.task_id],
                    cost=integrator_cost,
                    auto_accept=True,
                    force=True,
                )
                # Materialize the updated main workspace to disk
                try:
                    repo.workspace_update(target_lane)
                except Exception as e:
                    self.log.warning("Failed to materialize main workspace: %s", e)

                self.emit(EventType.PROMOTE_COMPLETED, task_id=task.task_id, status="force")
                return {
                    "success": True,
                    "status": "resolved",
                    "conflicts": conflicts,
                    "tokens_in": response.usage.input_tokens,
                    "tokens_out": response.usage.output_tokens,
                    "transition_id": force_result.get("transition_id"),
                }
            except Exception as e:
                self.log.error("Force promote failed: %s", e)
                return {
                    "success": False,
                    "status": "failed",
                    "conflicts": conflicts,
                    "tokens_in": response.usage.input_tokens,
                    "tokens_out": response.usage.output_tokens,
                    "error": str(e),
                }
        else:
            # Abort
            self.log.info("LLM decided to abort promote for task '%s'", task.task_id)
            return {
                "success": False,
                "status": "conflicts",
                "conflicts": conflicts,
                "tokens_in": response.usage.input_tokens,
                "tokens_out": response.usage.output_tokens,
            }
