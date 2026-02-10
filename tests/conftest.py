"""Shared test fixtures for Laneswarm tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from flanes.repo import Repository

from laneswarm.config import Config
from laneswarm.events import EventBus
from laneswarm.providers import LLMResponse, ProviderConfig, ProviderRegistry, TokenUsage
from laneswarm.task_graph import Task, TaskGraph


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Repository:
    """Create a temporary Flanes repository with initial files."""
    (tmp_path / "README.md").write_text("# Test Project\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")

    repo = Repository.init(tmp_path)
    return repo


@pytest.fixture
def sample_task_graph() -> TaskGraph:
    """Create a sample task graph for testing."""
    tasks = [
        Task(
            task_id="001",
            title="Project setup",
            description="Create configuration files and project structure.",
            dependencies=[],
            files_to_create=["src/config.py"],
            estimated_complexity="low",
        ),
        Task(
            task_id="002",
            title="Database models",
            description="Implement database models for users and posts.",
            dependencies=["001"],
            files_to_create=["src/models.py"],
            files_to_read=["src/config.py"],
            estimated_complexity="medium",
        ),
        Task(
            task_id="003",
            title="API routes",
            description="Create REST API endpoints.",
            dependencies=["001"],
            files_to_create=["src/routes.py"],
            files_to_read=["src/config.py"],
            estimated_complexity="medium",
        ),
        Task(
            task_id="004",
            title="Frontend",
            description="Build the web frontend.",
            dependencies=["002", "003"],
            files_to_create=["src/app.py"],
            files_to_read=["src/routes.py", "src/models.py"],
            estimated_complexity="high",
        ),
    ]
    return TaskGraph(tasks)


@pytest.fixture
def config() -> Config:
    """Create a test configuration."""
    return Config(
        providers={
            "mock": ProviderConfig(name="mock"),
        },
        models={
            "planner": "mock/test-model",
            "coder_high": "mock/test-model",
            "coder_medium": "mock/test-model",
            "coder_low": "mock/test-model",
            "reviewer": "mock/test-model",
            "integrator": "mock/test-model",
        },
        max_workers=2,
    )


@pytest.fixture
def event_bus() -> EventBus:
    """Create a test event bus."""
    return EventBus()


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or ["Mock response"]
        self._call_count = 0
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "mock"

    async def complete(self, messages, model, max_tokens=4096, tools=None,
                       temperature=0.0, system=None) -> LLMResponse:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        self.calls.append({
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
        })
        return LLMResponse(
            content=self._responses[idx],
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            model=model,
        )

    def list_models(self) -> list[str]:
        return ["test-model"]


@pytest.fixture
def mock_provider() -> MockProvider:
    """Create a mock LLM provider."""
    return MockProvider()


@pytest.fixture
def mock_registry(mock_provider: MockProvider) -> ProviderRegistry:
    """Create a provider registry with a mock provider."""
    registry = ProviderRegistry()
    registry.register(mock_provider)
    return registry
