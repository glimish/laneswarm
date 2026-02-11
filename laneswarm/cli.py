"""CLI entry point for Laneswarm.

Commands:
- laneswarm plan <spec>     — Generate a task graph from a project spec
- laneswarm run             — Execute tasks from the task graph
- laneswarm status          — Show current task status and progress
- laneswarm report          — Generate a cost/timeline report
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .config import Config, load_config
from .events import EventBus, EventType, SwarmEvent
from .flanes_bridge import aggregate_costs, load_task_graph, store_task_graph
from .orchestrator import Orchestrator
from .planner import Planner
from .providers import ProviderConfig, ProviderRegistry
from .task_graph import TaskGraph, TaskStatus

console = Console()
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _setup_registry(config: Config) -> ProviderRegistry:
    """Set up the provider registry from config."""
    registry = ProviderRegistry()

    # Register configured providers
    for name, provider_config in config.providers.items():
        try:
            provider = _create_provider(name, provider_config)
            registry.register(provider)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize {name} provider: {e}[/yellow]")

    # Auto-detect providers from environment if none configured
    if not config.providers:
        import os

        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from .providers.anthropic import AnthropicProvider

                provider = AnthropicProvider(
                    ProviderConfig(name="anthropic", api_key_env="ANTHROPIC_API_KEY")
                )
                registry.register(provider)
            except Exception:
                pass

        if os.environ.get("OPENAI_API_KEY"):
            try:
                from .providers.openai import OpenAIProvider

                provider = OpenAIProvider(
                    ProviderConfig(name="openai", api_key_env="OPENAI_API_KEY")
                )
                registry.register(provider)
            except Exception:
                pass

        # Try Claude Agent SDK (uses claude CLI — supports subscription auth)
        if "anthropic" not in registry.providers:
            try:
                from .providers.claude_sdk import ClaudeSDKProvider, is_claude_sdk_available

                if is_claude_sdk_available():
                    provider = ClaudeSDKProvider()
                    registry.register(provider)
                    console.print(
                        "[dim]Using Claude CLI for Anthropic models "
                        "(subscription auth via claude login)[/dim]"
                    )
            except Exception:
                pass

    # Apply model fallbacks: if a configured model's provider isn't available,
    # remap to an available provider
    _apply_model_fallbacks(config, registry)

    if not registry.providers:
        console.print(
            "[red]No LLM providers available. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
            "or create a .laneswarm.toml config file.[/red]"
        )

    return registry


# Fallback model mappings: provider -> [(fallback_provider, {model -> model})]
# Tried in order — first available provider wins.
_FALLBACK_CHAIN: dict[str, list[tuple[str, dict[str, str]]]] = {
    "anthropic": [
        # Prefer Claude SDK (same models, subscription auth)
        (
            "claude",
            {
                "claude-opus-4-6": "claude-opus-4-6",
                "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
                "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
            },
        ),
        # Fall back to OpenAI
        (
            "openai",
            {
                "claude-opus-4-6": "gpt-4o",
                "claude-sonnet-4-5-20250929": "gpt-4o",
                "claude-haiku-4-5-20251001": "gpt-4o-mini",
            },
        ),
    ],
    "openai": [
        (
            "anthropic",
            {
                "gpt-4o": "claude-sonnet-4-5-20250929",
                "gpt-4o-mini": "claude-haiku-4-5-20251001",
            },
        ),
        (
            "claude",
            {
                "gpt-4o": "claude-sonnet-4-5-20250929",
                "gpt-4o-mini": "claude-haiku-4-5-20251001",
            },
        ),
    ],
}


def _apply_model_fallbacks(config: Config, registry: ProviderRegistry) -> None:
    """Remap model assignments when a provider isn't available."""
    available = set(registry.providers.keys())
    if not available:
        return

    remapped: list[tuple[str, str, str]] = []  # (role, original, fallback)

    for role, model_id in list(config.models.items()):
        if "/" not in model_id:
            continue
        provider_name, model_name = model_id.split("/", 1)
        if provider_name in available:
            continue

        # Provider not available — try fallback chain in order
        chain = _FALLBACK_CHAIN.get(provider_name, [])
        found = False
        for fallback_provider, model_map in chain:
            if fallback_provider not in available:
                continue
            fallback_model = model_map.get(model_name)
            if fallback_model:
                fallback_id = f"{fallback_provider}/{fallback_model}"
                config.models[role] = fallback_id
                remapped.append((role, model_id, fallback_id))
                found = True
                break

        if found:
            continue

        # No specific fallback — use first available provider's default model
        first_available = sorted(available)[0]
        first_provider = registry.providers[first_available]
        models = first_provider.list_models()
        if models:
            fallback_id = f"{first_available}/{models[0]}"
            config.models[role] = fallback_id
            remapped.append((role, model_id, fallback_id))

    # Print a single consolidated message instead of one per role
    if remapped:
        # Group by fallback provider
        fallback_providers = {fb.split("/")[0] for _, _, fb in remapped}
        provider_str = ", ".join(sorted(fallback_providers))
        unavail = {orig.split("/")[0] for _, orig, _ in remapped}
        unavail_str = ", ".join(sorted(unavail))
        # Escape Rich markup — brackets are interpreted as style tags
        console.print(
            f"[yellow]Provider '{unavail_str}' not available. "
            f"Using '{provider_str}' for all roles.[/yellow]"
        )


def _create_provider(name: str, config: ProviderConfig):
    """Create a provider instance from config."""
    if name == "anthropic":
        from .providers.anthropic import AnthropicProvider

        return AnthropicProvider(config)
    elif name == "openai":
        from .providers.openai import OpenAIProvider

        return OpenAIProvider(config)
    elif name == "google":
        from .providers.google import GoogleProvider

        return GoogleProvider(config)
    elif name == "ollama":
        from .providers.ollama import OllamaProvider

        return OllamaProvider(config)
    elif name == "claude":
        from .providers.claude_sdk import ClaudeSDKProvider

        return ClaudeSDKProvider(config)
    else:
        raise ValueError(f"Unknown provider: {name}")


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--project", "-p", type=click.Path(exists=True), default=".", help="Project directory"
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, project: str) -> None:
    """Laneswarm: Multi-agent autonomous coding system."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["project_path"] = Path(project).resolve()
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("spec", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read spec from file")
@click.option("--interactive", "-i", is_flag=True, help="Interactive interview mode")
@click.pass_context
def plan(ctx: click.Context, spec: str | None, file: str | None, interactive: bool) -> None:
    """Generate a task graph from a project specification."""
    project_path = ctx.obj["project_path"]
    config = load_config(project_path)
    registry = _setup_registry(config)
    event_bus = EventBus()

    # Show planning phase progress in real-time
    def _on_planning_event(event: SwarmEvent) -> None:
        if event.event_type == EventType.PLANNING_PHASE_STARTED:
            idx = event.data.get("index", "?")
            total = event.data.get("total", "?")
            label = event.data.get("label", event.data.get("phase", ""))
            console.print(f"  [cyan]Phase {idx}/{total}:[/cyan] {label}...")
        elif event.event_type == EventType.PLANNING_PHASE_COMPLETED:
            idx = event.data.get("index", "?")
            phase = event.data.get("phase", "")
            console.print(f"  [green]Phase {idx} ({phase}) done[/green]")

    event_bus.subscribe(_on_planning_event)

    planner = Planner(project_path, config, registry, event_bus)

    if interactive:
        spec_text = asyncio.run(planner.interview())
    elif file:
        spec_text = Path(file).read_text()
    elif spec:
        spec_text = spec
    else:
        console.print("[red]Error: Provide a spec string, --file, or use --interactive[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Planning project at {project_path}...[/bold]\n")

    try:
        task_graph = planner.plan_sync(spec_text)
    except Exception as e:
        console.print(f"[red]Planning failed: {e}[/red]")
        sys.exit(1)

    # Display the task graph
    _display_task_graph(task_graph)
    console.print(f"\n[green]Plan created with {len(task_graph)} tasks.[/green]")
    console.print("Run [bold]laneswarm run[/bold] to start execution.")


@main.command()
@click.option("--workers", "-w", type=int, help="Number of parallel workers")
@click.option("--dashboard", "-d", is_flag=True, help="Launch live dashboard in browser")
@click.option("--port", type=int, default=8420, help="Dashboard port (default: 8420)")
@click.pass_context
def run(ctx: click.Context, workers: int | None, dashboard: bool, port: int) -> None:
    """Execute tasks from the task graph."""
    project_path = ctx.obj["project_path"]
    config = load_config(project_path)
    if workers:
        config.max_workers = workers
    registry = _setup_registry(config)

    # Set up event bus with CLI display
    event_bus = EventBus()
    progress_state: dict = {}

    def on_event(event: SwarmEvent) -> None:
        progress_state[event.event_type] = event

    event_bus.subscribe(on_event)

    # Load task graph from Flanes
    from flanes.repo import Repository

    repo = Repository.find(project_path)
    task_graph = load_task_graph(repo)

    if task_graph is None:
        console.print("[red]No task graph found. Run 'laneswarm plan' first.[/red]")
        sys.exit(1)

    # Start dashboard server if requested
    dashboard_server = None
    if dashboard:
        try:
            from .dashboard.server import DashboardServer

            dashboard_server = DashboardServer(event_bus, task_graph)
            dashboard_server.serve_background(host="0.0.0.0", port=port)
            url = f"http://localhost:{port}"
            console.print(f"[bold blue]Dashboard:[/bold blue] {url}")

            import webbrowser

            webbrowser.open(url)
        except ImportError:
            console.print(
                "[yellow]Warning: websockets not installed. "
                "Install with: pip install websockets>=12.0[/yellow]"
            )

    console.print(
        f"\n[bold]Running {len(task_graph)} tasks with {config.max_workers} workers...[/bold]\n"
    )

    orchestrator = Orchestrator(project_path, config, registry, event_bus)

    # On Windows, Claude CLI subprocesses may send CTRL_C_EVENT to the
    # process group when they exit, which Click interprets as an abort.
    # Temporarily ignore SIGINT during the entire orchestration + summary
    # phase so that spurious/delayed signals don't kill the run.
    import signal

    prev_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        summary = orchestrator.run(task_graph)
    except Exception as e:
        signal.signal(signal.SIGINT, prev_handler)
        console.print(f"[red]Execution failed: {e}[/red]")
        sys.exit(1)

    # Persist updated task graph (with completed/failed statuses) back to disk
    try:
        store_task_graph(repo, task_graph, agent_id="orchestrator")
    except Exception as e:
        logger.warning("Failed to persist task graph: %s", e)

    # Display summary
    console.print("\n[bold]Execution Complete[/bold]\n")
    console.print(f"  Completed: [green]{summary['completed']}[/green]")
    console.print(f"  Failed:    [red]{summary['failed']}[/red]")
    console.print(f"  Total:     {summary['total']}")
    console.print(f"  Tokens:    {summary['total_tokens']:,}")
    console.print(f"  Time:      {summary['elapsed_seconds']:.1f}s")

    # Smoke test results
    smoke = summary.get("smoke_test")
    if smoke is not None:
        if smoke.get("passed"):
            console.print(
                f"  Smoke:     [green]passed[/green] "
                f"({smoke.get('app_type', '?')} app, "
                f"{len(smoke.get('checks', []))} checks)"
            )
        else:
            console.print(f"  Smoke:     [red]failed[/red] ({smoke.get('app_type', '?')} app)")
            for check in smoke.get("checks", []):
                if not check.get("passed"):
                    console.print(
                        f"             [red]-[/red] {check['name']}: {check['detail'][:80]}"
                    )
            if smoke.get("diagnosis"):
                console.print("\n  [bold]Diagnosis:[/bold]")
                for line in smoke["diagnosis"].split("\n"):
                    console.print(f"    {line}")

    # Restore SIGINT handler only after all output is done
    signal.signal(signal.SIGINT, prev_handler)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current task status and progress."""
    project_path = ctx.obj["project_path"]

    from flanes.repo import Repository

    try:
        repo = Repository.find(project_path)
    except Exception:
        console.print("[red]No Flanes repository found at this path.[/red]")
        sys.exit(1)

    task_graph = load_task_graph(repo)
    config = load_config(project_path)

    if task_graph is None:
        console.print("[yellow]No task graph found. Run 'laneswarm plan' first.[/yellow]")
        sys.exit(0)

    _display_task_graph(task_graph)

    # Show cost summary
    costs = aggregate_costs(repo, task_graph=task_graph)
    if costs["total_tokens"] > 0:
        console.print("\n[bold]Costs[/bold]")
        console.print(f"  Tokens: {costs['total_tokens']:,}")
        console.print(f"  Time:   {costs['wall_time_s']:.1f}s")


@main.command()
@click.pass_context
def report(ctx: click.Context) -> None:
    """Generate a cost/timeline report."""
    project_path = ctx.obj["project_path"]

    from flanes.repo import Repository

    try:
        repo = Repository.find(project_path)
    except Exception:
        console.print("[red]No Flanes repository found at this path.[/red]")
        sys.exit(1)

    task_graph = load_task_graph(repo)
    if task_graph is None:
        console.print("[yellow]No task graph found.[/yellow]")
        sys.exit(0)

    # Full report
    console.print("\n[bold]Laneswarm Report[/bold]\n")

    # Task summary
    progress = task_graph.progress()
    console.print(f"Tasks: {progress['total']} total")
    console.print(f"  Completed:   {progress['completed']}")
    console.print(f"  Failed:      {progress['failed']}")
    console.print(f"  In Progress: {progress['in_progress']}")
    console.print(f"  Pending:     {progress['pending']}")

    # Per-task costs
    console.print("\n[bold]Per-Task Details[/bold]")
    table = Table()
    table.add_column("Task", style="cyan")
    table.add_column("Status")
    table.add_column("Tokens", justify="right")
    table.add_column("Time", justify="right")

    for task in task_graph.tasks:
        status_style = {
            TaskStatus.COMPLETED: "[green]completed[/green]",
            TaskStatus.FAILED: "[red]failed[/red]",
            TaskStatus.IN_PROGRESS: "[yellow]running[/yellow]",
            TaskStatus.PENDING: "pending",
            TaskStatus.BLOCKED: "[dim]blocked[/dim]",
        }.get(task.status, str(task.status.value))

        table.add_row(
            task.title[:40],
            status_style,
            f"{task.tokens_used:,}" if task.tokens_used else "-",
            f"{task.wall_time_ms / 1000:.1f}s" if task.wall_time_ms else "-",
        )

    console.print(table)

    # Overall costs (aggregate across all task lanes + main)
    costs = aggregate_costs(repo, task_graph=task_graph)
    console.print("\n[bold]Total Costs[/bold]")
    console.print(f"  Input tokens:  {costs['tokens_in']:,}")
    console.print(f"  Output tokens: {costs['tokens_out']:,}")
    console.print(f"  Total tokens:  {costs['total_tokens']:,}")
    console.print(f"  API calls:     {costs['api_calls']}")
    console.print(f"  Wall time:     {costs['wall_time_s']:.1f}s")


@main.command()
@click.option("--port", type=int, default=8420, help="Port to serve on (default: 8420)")
@click.option("--run", "run_tasks", is_flag=True, help="Also execute the task graph")
@click.option("--workers", "-w", type=int, help="Number of parallel workers (with --run)")
@click.option("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
@click.pass_context
def serve(ctx: click.Context, port: int, run_tasks: bool, workers: int | None, host: str) -> None:
    """Launch the real-time web dashboard.

    Without --run, serves the dashboard showing the current/last task graph.
    With --run, starts task execution and streams progress to the dashboard.
    """
    project_path = ctx.obj["project_path"]

    try:
        from .dashboard.server import DashboardServer
    except ImportError:
        console.print(
            "[red]websockets library not installed.[/red]\n"
            "Install with: [bold]pip install websockets>=12.0[/bold]"
        )
        sys.exit(1)

    # Load task graph
    from flanes.repo import Repository

    try:
        repo = Repository.find(project_path)
    except Exception:
        console.print("[red]No Flanes repository found at this path.[/red]")
        sys.exit(1)

    task_graph = load_task_graph(repo)
    event_bus = EventBus()

    url = f"http://localhost:{port}"
    console.print("\n[bold blue]Laneswarm Dashboard[/bold blue]")
    console.print(f"  URL: [link={url}]{url}[/link]")

    if run_tasks:
        if task_graph is None:
            console.print("[red]No task graph found. Run 'laneswarm plan' first.[/red]")
            sys.exit(1)

        config = load_config(project_path)
        if workers:
            config.max_workers = workers
        registry = _setup_registry(config)

        console.print(f"  Tasks: {len(task_graph)}")
        console.print(f"  Workers: {config.max_workers}")

        # Start dashboard server in background
        server = DashboardServer(event_bus, task_graph)
        server.serve_background(host=host, port=port)

        import webbrowser

        webbrowser.open(url)

        # Run the orchestrator (blocking)
        orchestrator = Orchestrator(project_path, config, registry, event_bus)

        import signal

        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            summary = orchestrator.run(task_graph)
        except Exception as e:
            signal.signal(signal.SIGINT, prev_handler)
            console.print(f"\n[red]Execution failed: {e}[/red]")
            sys.exit(1)

        signal.signal(signal.SIGINT, prev_handler)

        # Persist updated task graph
        try:
            store_task_graph(repo, task_graph, agent_id="orchestrator")
        except Exception as e:
            logger.warning("Failed to persist task graph: %s", e)

        # Display summary
        console.print("\n[bold]Execution Complete[/bold]\n")
        console.print(f"  Completed: [green]{summary['completed']}[/green]")
        console.print(f"  Failed:    [red]{summary['failed']}[/red]")
        console.print(f"  Tokens:    {summary['total_tokens']:,}")
        console.print(f"  Time:      {summary['elapsed_seconds']:.1f}s")

        # Smoke test results
        smoke = summary.get("smoke_test")
        if smoke is not None:
            if smoke.get("passed"):
                console.print(
                    f"  Smoke:     [green]passed[/green] ({smoke.get('app_type', '?')} app)"
                )
            else:
                console.print(f"  Smoke:     [red]failed[/red] ({smoke.get('app_type', '?')} app)")

        console.print(f"\n  Dashboard still running at {url}")
        console.print("  Press Ctrl+C to stop.")

        # Keep the dashboard running after execution completes
        try:
            import signal as _sig

            _sig.signal(_sig.SIGINT, _sig.default_int_handler)
            while True:
                import time as _time

                _time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[dim]Dashboard stopped.[/dim]")
    else:
        # Serve-only mode: show current state
        if task_graph is not None:
            console.print(f"  Tasks: {len(task_graph)}")
        else:
            console.print("  [dim]No task graph loaded[/dim]")

        server = DashboardServer(event_bus, task_graph)

        import webbrowser

        webbrowser.open(url)

        console.print("\n  Press Ctrl+C to stop.\n")
        try:
            server.serve(host=host, port=port)
        except KeyboardInterrupt:
            console.print("\n[dim]Dashboard stopped.[/dim]")


def _display_task_graph(task_graph: TaskGraph) -> None:
    """Display the task graph as a rich table."""
    table = Table(title="Task Graph")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Title", width=35)
    table.add_column("Status", width=12)
    table.add_column("Complexity", width=10)
    table.add_column("Dependencies", width=20)

    for task in task_graph.tasks:
        status_style = {
            TaskStatus.COMPLETED: "[green]completed[/green]",
            TaskStatus.FAILED: "[red]failed[/red]",
            TaskStatus.IN_PROGRESS: "[yellow]running[/yellow]",
            TaskStatus.PENDING: "pending",
            TaskStatus.BLOCKED: "[dim]blocked[/dim]",
        }.get(task.status, str(task.status.value))

        deps = ", ".join(task.dependencies) if task.dependencies else "-"

        table.add_row(
            task.task_id,
            task.title[:35],
            status_style,
            task.estimated_complexity,
            deps[:20],
        )

    console.print(table)


if __name__ == "__main__":
    main()
