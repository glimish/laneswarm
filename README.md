# Laneswarm

Multi-agent autonomous coding system powered by [Flanes](https://pypi.org/project/flanes/) version control.

## Install

```bash
pip install laneswarm
```

## Quick Start

```bash
# Plan a project
laneswarm plan "Build a todo app with authentication"

# Execute the tasks
laneswarm run

# Check status
laneswarm status
```

## Dashboard

Laneswarm includes a real-time web dashboard that streams agent activity as tasks execute.

```bash
# Run tasks with the dashboard
laneswarm run --dashboard

# Or launch the dashboard standalone (view current task graph)
laneswarm serve

# Dashboard + execution together
laneswarm serve --run
```

Opens at `http://localhost:8420` by default. Use `--port` to change.

**What you see:**
- **Task grid** — live cards for each task with status, current phase, and iteration count
- **Activity feed** — streaming event log from all agents, color-coded and filterable
- **Orchestrator panel** — worker utilization, dependency graph, overall progress
- **Task detail** — click any task to see agent steps timeline, files written, verification results, and review feedback

The dashboard connects via WebSocket and auto-reconnects if the connection drops.

## Configuration

Copy `.laneswarm.toml.example` to `.laneswarm.toml` and configure your LLM providers.
