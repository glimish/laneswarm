"""Configuration loading for Laneswarm.

Reads .laneswarm.toml from the project directory. Supports:
- Provider configs (API keys, subscriptions, cloud platforms)
- Model assignments per agent role
- Concurrency and budget settings
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from .providers import ProviderConfig

CONFIG_FILENAME = ".laneswarm.toml"

# Default model assignments per role
DEFAULT_MODELS = {
    "planner": "anthropic/claude-opus-4-6",
    "coder_high": "anthropic/claude-opus-4-6",
    "coder_medium": "anthropic/claude-sonnet-4-5-20250929",
    "coder_low": "anthropic/claude-haiku-4-5-20251001",
    "reviewer": "anthropic/claude-sonnet-4-5-20250929",
    "integrator": "anthropic/claude-sonnet-4-5-20250929",
    "smoker": "anthropic/claude-haiku-4-5-20251001",
}


@dataclass
class BudgetConfig:
    """Budget limits for task execution."""

    max_tokens_in: int = 500_000
    max_tokens_out: int = 200_000
    max_api_calls: int = 100
    max_wall_time_ms: int = 3_600_000  # 1 hour


@dataclass
class Config:
    """Laneswarm configuration."""

    # Provider configs keyed by provider name
    providers: dict[str, ProviderConfig] = field(default_factory=dict)

    # Model assignments per role (provider/model-name format)
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))

    # Concurrency
    max_workers: int = 4
    max_retries: int = 3

    # Default budget for new lanes
    default_budget: BudgetConfig = field(default_factory=BudgetConfig)

    # Evaluators to run on each workspace
    evaluators: list[dict[str, Any]] = field(default_factory=list)

    # Project path
    project_path: Path = field(default_factory=lambda: Path("."))

    def model_for_role(self, role: str) -> str:
        """Get the model ID for a given agent role."""
        if role in self.models:
            return self.models[role]
        # Fall back to coder_medium for unknown roles
        return self.models.get("coder_medium", DEFAULT_MODELS["coder_medium"])

    def model_for_complexity(self, complexity: str) -> str:
        """Get the model ID based on task complexity."""
        role = f"coder_{complexity}"
        return self.model_for_role(role)


def load_config(project_path: Path | None = None) -> Config:
    """Load configuration from .laneswarm.toml in the project directory.

    Falls back to defaults if no config file exists.
    """
    path = project_path or Path(".")
    config_file = path / CONFIG_FILENAME

    if not config_file.exists():
        return Config(project_path=path)

    if tomllib is None:
        raise ImportError(
            "tomli is required for Python < 3.11. "
            "Install it with: pip install tomli"
        )

    with open(config_file, "rb") as f:
        raw = tomllib.load(f)

    return _parse_config(raw, path)


def _parse_config(raw: dict[str, Any], project_path: Path) -> Config:
    """Parse raw TOML dict into Config."""
    config = Config(project_path=project_path)

    # Parse providers
    for name, provider_data in raw.get("providers", {}).items():
        config.providers[name] = ProviderConfig(
            name=name,
            auth=provider_data.get("auth", "api_key"),
            api_key_env=provider_data.get("api_key_env", ""),
            api_key=provider_data.get("api_key", ""),
            base_url=provider_data.get("base_url", ""),
            vertex_project=provider_data.get("vertex_project", ""),
            vertex_region=provider_data.get("vertex_region", ""),
            bedrock_region=provider_data.get("bedrock_region", ""),
            azure_endpoint=provider_data.get("azure_endpoint", ""),
            azure_api_key_env=provider_data.get("azure_api_key_env", ""),
            subscription_token_env=provider_data.get("subscription_token_env", ""),
            extra={
                k: v for k, v in provider_data.items()
                if k not in {
                    "auth", "api_key_env", "api_key", "base_url",
                    "vertex_project", "vertex_region", "bedrock_region",
                    "azure_endpoint", "azure_api_key_env", "subscription_token_env",
                }
            },
        )

    # Parse models
    models_data = raw.get("models", {})
    for role, model_id in models_data.items():
        config.models[role] = model_id

    # Parse concurrency
    config.max_workers = raw.get("max_workers", config.max_workers)
    config.max_retries = raw.get("max_retries", config.max_retries)

    # Parse default budget
    budget_data = raw.get("budget", {})
    if budget_data:
        config.default_budget = BudgetConfig(
            max_tokens_in=budget_data.get("max_tokens_in", 500_000),
            max_tokens_out=budget_data.get("max_tokens_out", 200_000),
            max_api_calls=budget_data.get("max_api_calls", 100),
            max_wall_time_ms=budget_data.get("max_wall_time_ms", 3_600_000),
        )

    # Parse evaluators
    config.evaluators = raw.get("evaluators", [])

    return config
