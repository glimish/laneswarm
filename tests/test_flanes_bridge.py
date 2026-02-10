"""Tests for the Flanes bridge module."""

from pathlib import Path

import pytest

from flanes.repo import Repository

from laneswarm.config import BudgetConfig, Config
from laneswarm.flanes_bridge import (
    _budget_for_complexity,
    _dir_tree,
    build_focused_prompt,
    create_lanes_for_tasks,
    init_project,
)
from laneswarm.task_graph import Task, TaskGraph


def test_init_project(tmp_path):
    config = Config()
    repo = init_project(tmp_path, config)
    assert isinstance(repo, Repository)
    assert (tmp_path / ".flanes").exists()


def test_budget_for_complexity():
    default = BudgetConfig(max_tokens_in=100_000)

    low = _budget_for_complexity("low", default)
    assert low.max_tokens_in == 50_000

    medium = _budget_for_complexity("medium", default)
    assert medium.max_tokens_in == 100_000

    high = _budget_for_complexity("high", default)
    assert high.max_tokens_in == 200_000


def test_create_lanes_for_tasks(tmp_repo):
    graph = TaskGraph([
        Task(task_id="001", title="Setup", description="...", dependencies=[]),
        Task(task_id="002", title="Models", description="...", dependencies=["001"]),
    ])

    config = Config()
    create_lanes_for_tasks(tmp_repo, graph, config)

    lanes = tmp_repo.lanes()
    lane_names = {l["name"] for l in lanes}
    assert "task-001" in lane_names
    assert "task-002" in lane_names


def test_build_focused_prompt(tmp_path):
    # Create some files in the workspace
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.py").write_text("DB_URL = 'sqlite:///test.db'")

    task = Task(
        task_id="002",
        title="Database models",
        description="Create user and post models.",
        dependencies=["001"],
        files_to_create=["src/models.py"],
        files_to_read=["src/config.py"],
    )

    graph = TaskGraph([
        Task(task_id="001", title="Setup", description="Init project."),
        task,
    ])

    prompt = build_focused_prompt(tmp_path, task, graph)

    assert "Database models" in prompt
    assert "Create user and post models" in prompt
    assert "src/models.py" in prompt
    assert "DB_URL" in prompt  # from config.py
    assert "Setup" in prompt  # dependency summary


def test_dir_tree(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / "tests").mkdir()

    tree = _dir_tree(tmp_path)
    assert "src" in tree
    assert "main.py" in tree
    assert "tests" in tree
