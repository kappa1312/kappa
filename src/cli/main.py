"""Main CLI entry point using Typer."""

from pathlib import Path

import anyio
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src import __version__

app = typer.Typer(
    name="kappa",
    help="Kappa OS - Autonomous Development Operating System",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]Kappa OS[/bold blue] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    Kappa OS - Build complete software projects 10x faster.

    Orchestrates multiple parallel Claude Code sessions to transform
    specifications into production-ready code.
    """
    pass


@app.command()
def run(
    spec: str = typer.Argument(..., help="Project specification or path to spec file"),
    project_path: Path = typer.Option(
        Path("."),
        "--project",
        "-p",
        help="Path to project directory",
    ),
    project_name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name (defaults to directory name)",
    ),
    _max_sessions: int = typer.Option(
        5,
        "--max-sessions",
        "-s",
        help="Maximum parallel sessions",
    ),
    _debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug mode",
    ),
) -> None:
    """
    Run Kappa to build a project from specification.

    Example:
        kappa run "Build a REST API with user authentication" -p ./my-api
    """
    # Check if spec is a file path
    spec_path = Path(spec)
    if spec_path.exists() and spec_path.is_file():
        spec = spec_path.read_text()
        console.print(f"[dim]Loaded specification from {spec_path}[/dim]")

    console.print(
        Panel(
            f"[bold]Specification:[/bold]\n{spec[:200]}{'...' if len(spec) > 200 else ''}",
            title="[bold blue]Kappa OS[/bold blue]",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Initializing Kappa...", total=None)

        async def execute() -> None:
            from src.core.orchestrator import Kappa

            # Override settings if needed
            kappa = Kappa()

            result = await kappa.run(
                spec=spec,
                project_path=str(project_path.resolve()),
                project_name=project_name,
            )

            # Display results
            if result.get("status") == "completed":
                console.print("\n[bold green]Project completed successfully![/bold green]")
            else:
                console.print(f"\n[bold red]Project failed: {result.get('error')}[/bold red]")

            console.print(f"\n[dim]Output:[/dim]\n{result.get('final_output', 'No output')}")

        anyio.run(execute)


@app.command()
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Path to initialize project",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name",
    ),
) -> None:
    """
    Initialize a new Kappa project directory.

    Creates the basic structure and configuration files.
    """
    project_path = path.resolve()
    project_name = name or project_path.name

    console.print(f"Initializing project: [bold]{project_name}[/bold]")

    # Create directories
    dirs = ["src", "tests", "docs", ".claude/skills"]
    for d in dirs:
        (project_path / d).mkdir(parents=True, exist_ok=True)

    # Create basic files
    (project_path / "README.md").write_text(f"# {project_name}\n\nA Kappa-generated project.\n")
    (project_path / ".gitignore").write_text("__pycache__/\n*.pyc\n.env\n.venv/\n")

    console.print(f"[green]Project initialized at {project_path}[/green]")


@app.command()
def status(
    project_id: str | None = typer.Argument(
        None,
        help="Project ID to check status",
    ),
) -> None:
    """
    Check status of running or completed projects.
    """
    if project_id:
        console.print(f"Checking status for project: {project_id}")
        # TODO: Implement actual status check
        console.print("[yellow]Status check not yet implemented[/yellow]")
    else:
        console.print("[bold]Recent Projects[/bold]")
        console.print("[dim]No projects found[/dim]")


@app.command()
def decompose(
    spec: str = typer.Argument(..., help="Specification to decompose"),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for task breakdown",
    ),
) -> None:
    """
    Decompose a specification into tasks without executing.

    Useful for previewing the task breakdown.
    """
    console.print("[bold]Decomposing specification...[/bold]")

    async def do_decompose() -> None:
        from src.core.orchestrator import Kappa

        kappa = Kappa()
        tasks = await kappa.decompose(spec)

        # Display tasks
        table = Table(title="Task Breakdown")
        table.add_column("Wave", style="cyan")
        table.add_column("Name", style="bold")
        table.add_column("Category")
        table.add_column("Dependencies")

        for task in tasks:
            deps = ", ".join(task.get("dependencies", [])) or "-"
            table.add_row(
                str(task.get("wave", 0)),
                task.get("name", "Unknown"),
                task.get("category", "unknown"),
                deps[:30] + "..." if len(deps) > 30 else deps,
            )

        console.print(table)

        if output:
            import json

            output.write_text(json.dumps(tasks, indent=2))
            console.print(f"[green]Saved to {output}[/green]")

    anyio.run(do_decompose)


@app.command()
def health() -> None:
    """
    Check system health and dependencies.
    """
    console.print("[bold]Running health checks...[/bold]\n")

    async def do_health_check() -> None:
        from src.monitoring.health_check import check_health

        result = await check_health()

        # Display results
        status_colors = {
            "healthy": "green",
            "degraded": "yellow",
            "unhealthy": "red",
            "unknown": "dim",
        }

        overall = result["status"]
        color = status_colors.get(overall, "white")
        console.print(f"Overall Status: [{color}]{overall.upper()}[/{color}]\n")

        for check in result["checks"]:
            name = check["name"]
            status = check["status"]
            message = check["message"]
            color = status_colors.get(status, "white")
            console.print(f"  {name}: [{color}]{status}[/{color}] - {message}")

    anyio.run(do_health_check)


@app.command()
def chat() -> None:
    """
    Start interactive chat with Kappa OS.

    Guides you through project ideation to development with
    a conversational interface.

    Example:
        kappa chat
    """

    async def start() -> None:
        from src.chat.interface import start_chat_cli

        await start_chat_cli()

    anyio.run(start)


@app.command()
def dashboard(
    port: int = typer.Option(8000, "--port", "-p", help="Port for dashboard server"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
) -> None:
    """
    Start the Kappa OS dashboard server.

    Provides a REST API and WebSocket interface for real-time
    project monitoring and management.

    Example:
        kappa dashboard
        kappa dashboard --port 3000
    """
    import uvicorn

    from src.api.main import app as api_app

    console.print(
        Panel(
            f"[bold]Dashboard:[/bold] http://localhost:{port}\n"
            f"[bold]API Docs:[/bold]  http://localhost:{port}/docs\n"
            f"[bold]Health:[/bold]    http://localhost:{port}/health",
            title="[bold cyan]Kappa OS Dashboard[/bold cyan]",
            border_style="cyan",
        )
    )

    uvicorn.run(api_app, host=host, port=port, log_level="info")


@app.command()
def build(
    requirements: str = typer.Argument(..., help="Requirements text or path to requirements file"),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Output workspace directory",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Project name",
    ),
    _watch: bool = typer.Option(
        False,
        "--watch",
        help="Watch progress in real-time",
    ),
) -> None:
    """
    Build a project from requirements.

    Accepts either inline requirements text or a path to a requirements file.

    Example:
        kappa build "Build a REST API with authentication"
        kappa build ./requirements.md --name my-api
    """
    # Check if requirements is a file path
    req_path = Path(requirements)
    if req_path.exists() and req_path.is_file():
        requirements_text = req_path.read_text()
        console.print(f"[dim]Loaded requirements from {req_path}[/dim]")
    else:
        requirements_text = requirements

    console.print(
        Panel(
            f"[bold]Requirements:[/bold]\n{requirements_text[:300]}{'...' if len(requirements_text) > 300 else ''}",
            title="[bold blue]Kappa Build[/bold blue]",
            border_style="blue",
        )
    )

    async def execute() -> None:
        from src.core.orchestrator import Kappa

        kappa = Kappa(workspace=str(workspace) if workspace else None)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building project...", total=None)

            result = await kappa.execute(
                requirements=requirements_text,
                project_name=name,
            )

            progress.update(task, completed=True)

        # Display results
        if result.get("status") == "completed":
            console.print("\n[bold green]Build completed successfully![/bold green]")
            console.print(f"[dim]Workspace: {result.get('workspace_path')}[/dim]")
        else:
            console.print(f"\n[bold red]Build failed: {result.get('error')}[/bold red]")

        if result.get("final_output"):
            console.print(f"\n[dim]Output:[/dim]\n{result.get('final_output')[:500]}")

    anyio.run(execute)


if __name__ == "__main__":
    app()
