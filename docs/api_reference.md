# Kappa OS API Reference

## Core Module

### Kappa Class

The main orchestrator class for running Kappa.

```python
from src.core.orchestrator import Kappa

kappa = Kappa()
```

#### Constructor

```python
def __init__(self, settings: Settings | None = None) -> None:
    """Initialize Kappa orchestrator.

    Args:
        settings: Optional settings override. Uses default if not provided.
    """
```

#### Methods

##### run

```python
async def run(
    self,
    spec: str,
    project_path: str | Path,
    project_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> KappaState:
    """Execute the full Kappa pipeline.

    Args:
        spec: Natural language project specification.
        project_path: Path to the project directory.
        project_name: Optional project name.
        config: Optional configuration overrides.

    Returns:
        Final KappaState with results.

    Example:
        >>> result = await kappa.run(
        ...     "Build a REST API",
        ...     "./my-project"
        ... )
    """
```

##### decompose

```python
async def decompose(self, spec: str) -> list[dict[str, Any]]:
    """Decompose a specification into tasks without execution.

    Args:
        spec: Natural language project specification.

    Returns:
        List of task dictionaries.

    Example:
        >>> tasks = await kappa.decompose("Build a calculator")
        >>> len(tasks)
        5
    """
```

##### status

```python
async def status(self, project_id: str) -> KappaState | None:
    """Get the current status of a project.

    Args:
        project_id: Project UUID.

    Returns:
        Current state if found, None otherwise.
    """
```

##### cancel

```python
async def cancel(self, project_id: str) -> bool:
    """Cancel a running project execution.

    Args:
        project_id: Project UUID to cancel.

    Returns:
        True if cancelled successfully.
    """
```

### Settings Class

Configuration management using Pydantic Settings.

```python
from src.core.config import Settings, get_settings

settings = get_settings()
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `anthropic_api_key` | `SecretStr` | Anthropic API key |
| `database_url` | `PostgresDsn` | PostgreSQL connection URL |
| `kappa_log_level` | `str` | Logging level |
| `kappa_max_parallel_sessions` | `int` | Max concurrent sessions |
| `kappa_session_timeout` | `int` | Session timeout (seconds) |
| `kappa_debug` | `bool` | Debug mode flag |

### KappaState TypedDict

State object that flows through the LangGraph pipeline.

```python
from src.core.state import KappaState, create_initial_state

state = create_initial_state(
    specification="Build an API",
    project_path="/path/to/project"
)
```

## Decomposition Module

### TaskGenerator

Generates tasks from specifications.

```python
from src.decomposition.task_generator import TaskGenerator

generator = TaskGenerator()
tasks = await generator.generate("Build a blog")
```

#### Methods

##### generate

```python
async def generate(
    self,
    specification: str,
    context: dict[str, Any] | None = None,
) -> list[Task]:
    """Generate tasks from a specification.

    Args:
        specification: Natural language project specification.
        context: Optional context with existing project info.

    Returns:
        List of Task objects.
    """
```

### DependencyResolver

Resolves task dependencies and assigns waves.

```python
from src.decomposition.dependency_resolver import DependencyResolver

resolver = DependencyResolver()
graph = resolver.build_graph(tasks)
waves = resolver.assign_waves(tasks, graph)
```

#### Methods

##### build_graph

```python
def build_graph(self, tasks: list[Task]) -> dict[str, list[str]]:
    """Build a dependency graph from tasks.

    Args:
        tasks: List of Task objects.

    Returns:
        Dictionary mapping task_id -> [dependency_ids].
    """
```

##### assign_waves

```python
def assign_waves(
    self,
    tasks: list[Task],
    graph: dict[str, list[str]],
) -> list[list[str]]:
    """Assign tasks to execution waves.

    Args:
        tasks: List of Task objects.
        graph: Dependency graph.

    Returns:
        List of waves (each wave is list of task_ids).
    """
```

### Task Model

```python
from src.decomposition.models import Task, TaskCategory, Complexity

task = Task(
    name="Create User model",
    description="Define User SQLAlchemy model",
    category=TaskCategory.DATA_MODEL,
    complexity=Complexity.MEDIUM,
    dependencies=["setup-task"],
    wave=1,
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique task ID (auto-generated) |
| `name` | `str` | Task name |
| `description` | `str` | Detailed description |
| `category` | `TaskCategory` | Task category |
| `complexity` | `Complexity` | LOW, MEDIUM, or HIGH |
| `dependencies` | `list[str]` | IDs of dependent tasks |
| `wave` | `int` | Execution wave |
| `file_targets` | `list[str]` | Target files |

## Sessions Module

### TerminalSession

Claude session using the Agent SDK.

```python
from src.sessions.terminal import TerminalSession
from src.sessions.base import SessionConfig

config = SessionConfig(
    max_turns=50,
    timeout_seconds=3600,
    working_directory="./project",
)

async with TerminalSession(config=config) as session:
    result = await session.execute("Create a hello world script")
```

#### Methods

##### execute

```python
async def execute(self, prompt: str) -> TaskResult:
    """Execute a prompt in the session.

    Args:
        prompt: Task prompt to execute.

    Returns:
        TaskResult with execution outcome.
    """
```

### SessionRouter

Manages parallel session execution.

```python
from src.sessions.router import SessionRouter

router = SessionRouter(max_sessions=5, timeout=3600)
results = await router.execute_tasks(tasks, state)
```

#### Methods

##### execute_tasks

```python
async def execute_tasks(
    self,
    tasks: list[dict[str, Any]],
    state: KappaState,
) -> list[TaskResult]:
    """Execute multiple tasks in parallel.

    Args:
        tasks: List of task dictionaries.
        state: Current Kappa state.

    Returns:
        List of TaskResult objects.
    """
```

## Knowledge Module

### ContextManager

Cross-session context sharing.

```python
from src.knowledge.context_manager import ContextManager
from src.knowledge.database import get_db_session

async with get_db_session() as db:
    manager = ContextManager(db)
    await manager.store_decision(
        project_id="project-1",
        category="architecture",
        decision="Use REST API",
    )
```

#### Methods

##### store_context

```python
async def store_context(
    self,
    session_id: str,
    context_type: str,
    key: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ContextSnapshot:
    """Store a context snapshot."""
```

##### store_decision

```python
async def store_decision(
    self,
    project_id: str,
    category: str,
    decision: str,
    rationale: str | None = None,
    alternatives: list[str] | None = None,
) -> Decision:
    """Store an architectural decision."""
```

##### get_decisions

```python
async def get_decisions(
    self,
    project_id: str,
    category: str | None = None,
) -> list[Decision]:
    """Get decisions for a project."""
```

## Conflict Module

### ConflictDetector

Detects conflicts between session outputs.

```python
from src.conflict.detector import ConflictDetector

detector = ConflictDetector()
conflicts = await detector.detect(task_results)
```

### ConflictResolver

Resolves detected conflicts.

```python
from src.conflict.resolver import ConflictResolver

resolver = ConflictResolver()
resolution = await resolver.resolve(conflict)
```

## Monitoring Module

### HealthChecker

System health checks.

```python
from src.monitoring.health_check import check_health

health = await check_health()
print(health["status"])  # "healthy", "degraded", or "unhealthy"
```

### MetricsCollector

Performance metrics collection.

```python
from src.monitoring.metrics import MetricsCollector

collector = MetricsCollector()
collector.start_collection()
collector.record_task_completion("task-1", 45.2, success=True)
summary = collector.get_summary()
```

## CLI Commands

### kappa run

```bash
kappa run SPEC [OPTIONS]

Arguments:
  SPEC  Project specification or path to spec file

Options:
  -p, --project PATH       Project directory
  -n, --name TEXT         Project name
  -s, --max-sessions INT  Max parallel sessions
  -d, --debug             Enable debug mode
```

### kappa decompose

```bash
kappa decompose SPEC [OPTIONS]

Arguments:
  SPEC  Specification to decompose

Options:
  -o, --output PATH  Output file for task breakdown
```

### kappa health

```bash
kappa health

Runs system health checks and displays status.
```

### kappa status

```bash
kappa status [PROJECT_ID]

Check status of running or completed projects.
```

### kappa logs

```bash
kappa logs [OPTIONS]

Options:
  --tail, -n INT  Number of lines to show
  --follow, -f    Follow log output
```

## Exceptions

```python
from src.core.orchestrator import (
    KappaError,
    DecompositionError,
    SessionError,
    ConflictError,
)

try:
    result = await kappa.run(spec, path)
except DecompositionError:
    print("Failed to decompose specification")
except SessionError:
    print("Claude session failed")
except ConflictError:
    print("Unresolvable conflict")
except KappaError:
    print("General Kappa error")
```
