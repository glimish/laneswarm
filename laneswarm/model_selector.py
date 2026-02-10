"""Model selection logic.

Chooses the best model for a task based on:
- Task complexity (low/medium/high)
- Budget remaining on the lane
- Config-defined model assignments
- Retry count (downgrade on repeated failures)
"""

from __future__ import annotations

import logging

from .config import Config
from .task_graph import Task

logger = logging.getLogger(__name__)

# Cost tiers: cheaper models first
MODEL_COST_TIERS = [
    # (role_key, description)
    ("coder_low", "low-cost model for simple tasks"),
    ("coder_medium", "standard model for typical tasks"),
    ("coder_high", "premium model for complex tasks"),
]


def select_model(task: Task, config: Config) -> str:
    """Select the best model for a task.

    Priority:
    1. Task-level model override
    2. Complexity-based selection from config
    3. Downgrade on retries (save budget for another attempt with simpler model)
    """
    # Task-level override takes precedence
    if task.model_override:
        return task.model_override

    # Map complexity to config role
    model_id = config.model_for_complexity(task.estimated_complexity)

    # On retries, consider downgrading to save budget
    if task.retries > 0:
        model_id = _downgrade_for_retry(task, config, model_id)

    return model_id


def _downgrade_for_retry(task: Task, config: Config, current_model: str) -> str:
    """Downgrade model on retries to conserve budget.

    On first retry: keep same model
    On second retry: go one tier down
    On third+ retry: use cheapest model
    """
    if task.retries < 2:
        return current_model

    # Get the cost tier index for the current model
    tier_keys = [t[0] for t in MODEL_COST_TIERS]
    current_tier = _find_tier(current_model, config, tier_keys)

    if task.retries >= 3:
        # Use cheapest model
        target_tier = 0
    else:
        # Go one tier down
        target_tier = max(0, current_tier - 1)

    downgraded = config.model_for_role(tier_keys[target_tier])
    if downgraded != current_model:
        logger.info(
            "Downgrading model for task '%s' (retry %d): %s -> %s",
            task.task_id, task.retries, current_model, downgraded,
        )
    return downgraded


def _find_tier(model_id: str, config: Config, tier_keys: list[str]) -> int:
    """Find which cost tier a model belongs to."""
    for i, key in enumerate(tier_keys):
        if config.model_for_role(key) == model_id:
            return i
    # Default to medium tier
    return 1


def select_reviewer_model(config: Config) -> str:
    """Select model for review tasks."""
    return config.model_for_role("reviewer")


def select_integrator_model(config: Config, has_conflicts: bool = False) -> str:
    """Select model for integration tasks.

    Uses a more capable model when conflicts need resolution.
    """
    if has_conflicts:
        return config.model_for_role("coder_high")
    return config.model_for_role("integrator")


def select_planner_model(config: Config) -> str:
    """Select model for the planner agent."""
    return config.model_for_role("planner")
