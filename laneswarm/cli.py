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
from rich.live import Live
from rich.table import Table

from .config import Config, load_config
from .events import EventBus, EventType, SwarmEvent
from .flanes_bridge import aggregate_costs, init_project, load_task_graph, store_task_graph
from .model_selector import select_model
from .orchestrator import Orchestrator
from .planner import Planner
from .providers import ProviderConfig, ProviderRegistry
from .task_graph import TaskGraph, TaskStatus

console = Console()


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
        ("claude", {
            "claude-opus-4-6": "claude-opus-4-6",
            "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
        }),
        # Fall back to OpenAI
        ("openai", {
            "claude-opus-4-6": "gpt-4o",
            "claude-sonnet-4-5-20250929": "gpt-4o",
            "claude-haiku-4-5-20251001": "gpt-4o-mini",
        }),
    ],
    "openai": [
        ("anthropic", {
            "gpt-4o": "claude-sonnet-4-5-20250929",
            "gpt-4o-mini": "claude-haiku-4-5-20251001",
        }),
        ("claude", {
            "gpt-4o": "claude-sonnet-4-5-20250929",
            "gpt-4o-mini": "claude-haiku-4-5-20251001",
        }),
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
@click.option("--project", "-p", type=click.Path(exists=True), default=".",
              help="Project directory")
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
@click.pass_context
def run(ctx: click.Context, workers: int | None) -> None:
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

    console.print(
        f"\n[bold]Running {len(task_graph)} tasks "
        f"with {config.max_workers} workers...[/bold]\n"
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
        console.print(f"\n[bold]Costs[/bold]")
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
    console.print(f"\n[bold]Total Costs[/bold]")
    console.print(f"  Input tokens:  {costs['tokens_in']:,}")
    console.print(f"  Output tokens: {costs['tokens_out']:,}")
    console.print(f"  Total tokens:  {costs['total_tokens']:,}")
    console.print(f"  API calls:     {costs['api_calls']}")
    console.print(f"  Wall time:     {costs['wall_time_s']:.1f}s")


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
