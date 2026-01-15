"""Additional CLI commands for Kappa."""

from pathlib import Path

import anyio
import typer
from rich.console import Console
from rich.table import Table

from src.cli.main import app

console = Console()


@app.command()
def logs(
    session_id: str | None = typer.Argument(
        None,
        help="Session ID to view logs for",
    ),
    tail: int = typer.Option(
        50,
        "--tail",
        "-n",
        help="Number of lines to show",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Follow log output",
    ),
) -> None:
    """
    View logs from Kappa execution.
    """
    log_dir = Path("logs")

    if not log_dir.exists():
        console.print("[yellow]No logs directory found[/yellow]")
        return

    log_files = sorted(log_dir.glob("kappa_*.log"), reverse=True)

    if not log_files:
        console.print("[yellow]No log files found[/yellow]")
        return

    latest_log = log_files[0]
    console.print(f"[dim]Reading from {latest_log}[/dim]\n")

    with open(latest_log) as f:
        lines = f.readlines()
        for line in lines[-tail:]:
            console.print(line.rstrip())


@app.command()
def resolve(
    conflict_id: str = typer.Argument(..., help="Conflict ID to resolve"),
    strategy: str = typer.Option(
        "merge",
        "--strategy",
        "-s",
        help="Resolution strategy: merge, newer, manual",
    ),
) -> None:
    """
    Manually resolve a conflict.
    """
    console.print(f"Resolving conflict {conflict_id} with strategy: {strategy}")

    async def do_resolve() -> None:
        from src.conflict.resolver import ConflictResolver

        resolver = ConflictResolver()

        # TODO: Load conflict from database
        conflict = {
            "id": conflict_id,
            "conflict_type": strategy,
            "file_path": "unknown",
        }

        try:
            resolution = await resolver.resolve(conflict)
            console.print(f"[green]Resolved: {resolution}[/green]")
        except ValueError as e:
            console.print(f"[red]Failed to resolve: {e}[/red]")

    anyio.run(do_resolve)


@app.command()
def metrics(
    project_id: str | None = typer.Argument(
        None,
        help="Project ID to show metrics for",
    ),
    export: Path | None = typer.Option(
        None,
        "--export",
        "-e",
        help="Export metrics to file",
    ),
) -> None:
    """
    Show execution metrics.
    """
    console.print("[bold]Execution Metrics[/bold]\n")

    # TODO: Load actual metrics from database
    table = Table()
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Total Tasks", "0")
    table.add_row("Successful", "0")
    table.add_row("Failed", "0")
    table.add_row("Total Duration", "0s")
    table.add_row("Tokens Used", "0")

    console.print(table)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration",
    ),
    set_value: str | None = typer.Option(
        None,
        "--set",
        help="Set a configuration value (key=value)",
    ),
) -> None:
    """
    View or modify Kappa configuration.
    """
    if show:
        from src.core.config import get_settings

        settings = get_settings()

        table = Table(title="Current Configuration")
        table.add_column("Setting")
        table.add_column("Value")

        table.add_row("Log Level", settings.kappa_log_level)
        table.add_row("Max Sessions", str(settings.kappa_max_parallel_sessions))
        table.add_row("Session Timeout", f"{settings.kappa_session_timeout}s")
        table.add_row("Debug Mode", str(settings.kappa_debug))
        table.add_row("Working Dir", settings.kappa_working_dir)

        console.print(table)

    elif set_value:
        console.print("[yellow]Configuration setting not yet implemented[/yellow]")

    else:
        console.print("Use --show to view configuration or --set to modify")


@app.command()
def clean(
    all_data: bool = typer.Option(
        False,
        "--all",
        help="Remove all data including database",
    ),
    logs_only: bool = typer.Option(
        False,
        "--logs",
        help="Remove only log files",
    ),
    cache_only: bool = typer.Option(
        False,
        "--cache",
        help="Remove only cache files",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """
    Clean up Kappa data and temporary files.
    """
    if not any([all_data, logs_only, cache_only]):
        console.print("Specify what to clean: --all, --logs, or --cache")
        return

    if all_data and not force:
        confirm = typer.confirm("This will remove ALL data. Continue?")
        if not confirm:
            console.print("[yellow]Aborted[/yellow]")
            return

    import shutil

    if logs_only or all_data:
        log_dir = Path("logs")
        if log_dir.exists():
            shutil.rmtree(log_dir)
            console.print("[green]Removed logs directory[/green]")

    if cache_only or all_data:
        cache_dirs = [Path(".kappa"), Path("__pycache__")]
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                console.print(f"[green]Removed {cache_dir}[/green]")

    console.print("[green]Cleanup complete[/green]")
