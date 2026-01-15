# Kappa OS Development Guide

## Product Vision

Kappa OS is an autonomous development operating system that orchestrates multiple parallel Claude Code sessions to build complete software projects 10x faster than traditional manual coding.

**Core Value Proposition:** Transform "weeks of coding" into "hours of orchestration"

**Target Users:** Development teams, solo developers, and enterprises who need to rapidly prototype, build, and ship production-quality software.

## Architecture Philosophy

### Master-Clone Pattern

Kappa uses a hierarchical orchestration model:

```
┌────────────────────────────────────────────────────────────────┐
│                    KAPPA ORCHESTRATOR (Master)                  │
│                                                                 │
│  Responsibilities:                                              │
│  - Parse and decompose requirements                             │
│  - Generate dependency graph                                    │
│  - Spawn and manage clone sessions                              │
│  - Coordinate context sharing                                   │
│  - Resolve conflicts and merge results                          │
└────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Clone S1 │   │ Clone S2 │   │ Clone S3 │
        │          │   │          │   │          │
        │ Task A   │   │ Task B   │   │ Task C   │
        │ Task D   │   │ Task E   │   │ Task F   │
        └──────────┘   └──────────┘   └──────────┘
              │               │               │
              └───────────────┴───────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Shared PostgreSQL │
                    │  Knowledge Base   │
                    └───────────────────┘
```

### Key Design Principles

1. **Terminal-First**: 80% of development work happens in the terminal. Kappa leverages Claude Code's terminal automation capabilities for file operations, git commands, testing, and deployment.

2. **Wave-Based Execution**: Tasks execute in parallel waves based on their dependency graph. Independent tasks run simultaneously; dependent tasks wait for prerequisites.

3. **Production Quality**: All generated code must:
   - Compile without errors
   - Pass type checking (mypy strict)
   - Include appropriate tests
   - Follow project code style
   - Be deployment-ready

4. **Skills Over Rigid Agents**: Use auto-invoked Claude skills for domain knowledge instead of hardcoded agent workflows. Skills provide flexible, contextual guidance.

5. **Context is King**: Cross-session context sharing via PostgreSQL eliminates redundant exploration and ensures consistent architectural decisions.

## Technology Stack

### Core Dependencies (2026 Latest)

| Package | Version | Purpose |
|---------|---------|---------|
| `python` | ^3.11 | Runtime (async/await native) |
| `claude-agent-sdk` | ^0.1.19 | Terminal automation, file ops |
| `langgraph` | ^1.0.0 | Durable state machine orchestration |
| `sqlalchemy[asyncio]` | ^2.0.45 | Async ORM and query builder |
| `asyncpg` | ^0.31.0 | Fast async PostgreSQL (5x psycopg3) |
| `pydantic` | ^2.12.0 | Data validation and settings |
| `pydantic-settings` | ^2.7.0 | Environment configuration |
| `typer` | ^0.21.0 | CLI framework |
| `rich` | ^13.9.0 | Terminal formatting |
| `loguru` | ^0.7.3 | Structured logging |

### Why These Choices

- **claude-agent-sdk** over raw anthropic: Provides the full Claude Code toolset (file operations, bash execution, web search) with proper session management
- **asyncpg** over psycopg2: 5x faster, native asyncio support, better connection pooling
- **LangGraph 1.0**: GA release with durable state, automatic recovery, human-in-the-loop
- **Pydantic v2**: 50x faster validation, better TypedDict support

## Project Structure

```
kappa/
├── src/
│   ├── core/           # Orchestrator, state, config
│   ├── graph/          # LangGraph nodes, edges, builder
│   ├── decomposition/  # Task parsing and generation
│   ├── sessions/       # Claude session management
│   ├── knowledge/      # PostgreSQL context sharing
│   ├── conflict/       # Conflict detection and resolution
│   ├── prompts/        # Dynamic prompt templates
│   ├── monitoring/     # Health checks and metrics
│   └── cli/            # Typer CLI commands
├── tests/
│   ├── unit/           # Fast, isolated tests
│   ├── integration/    # Database and session tests
│   └── e2e/            # Full workflow tests
├── scripts/            # Setup and utility scripts
└── docs/               # Documentation
```

## Development Workflow

### Code Style

```python
# Formatting: Black (line length 100)
poetry run black src tests

# Linting: Ruff (replaces Flake8 + isort)
poetry run ruff check src tests --fix

# Type Checking: Mypy strict mode
poetry run mypy src
```

### Docstring Format (Google Style)

```python
async def decompose_requirements(
    spec: str,
    context: ProjectContext,
) -> list[Task]:
    """Decompose a project specification into executable tasks.

    Parses the natural language specification and generates a
    dependency-ordered list of tasks suitable for parallel execution.

    Args:
        spec: Natural language project specification.
        context: Current project context including existing code,
            architecture decisions, and constraints.

    Returns:
        List of Task objects ordered by dependency wave.

    Raises:
        DecompositionError: If the specification cannot be parsed.
        DependencyError: If circular dependencies are detected.

    Example:
        >>> tasks = await decompose_requirements(
        ...     "Build a REST API with auth",
        ...     context
        ... )
        >>> len(tasks)
        12
    """
```

### Testing Standards

```python
# Unit tests: Fast, isolated, mock external dependencies
@pytest.mark.unit
async def test_task_generator_creates_valid_tasks():
    generator = TaskGenerator()
    tasks = await generator.generate(mock_spec)
    assert all(task.is_valid() for task in tasks)

# Integration tests: Real database, mock Claude sessions
@pytest.mark.integration
async def test_context_manager_persists_decisions(db_session):
    manager = ContextManager(db_session)
    await manager.store_decision("Use REST over GraphQL")
    decisions = await manager.get_decisions()
    assert "REST" in decisions[0].content

# E2E tests: Full workflow with real sessions
@pytest.mark.e2e
async def test_full_project_generation():
    kappa = Kappa()
    result = await kappa.run("Build a calculator CLI")
    assert result.success
    assert result.tests_passed
```

### Minimum Coverage: 80%

```bash
poetry run pytest --cov=src --cov-fail-under=80
```

## Git Workflow

### Branch Naming

```bash
feature/decomposition-parser     # New features
fix/session-timeout-handling     # Bug fixes
refactor/async-database-layer    # Code improvements
docs/api-reference               # Documentation
```

### Commit Messages (Conventional Commits)

```bash
feat(decomposition): add task generator with dependency resolution
fix(sessions): handle timeout during long-running tasks
refactor(knowledge): migrate to asyncpg for better performance
docs(readme): add architecture diagram
test(conflict): add integration tests for merge strategies
```

### Pull Request Template

```markdown
## Summary
Brief description of changes

## Changes
- [ ] Change 1
- [ ] Change 2

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Key Patterns

### Async Everything

```python
# Always use async for I/O operations
async def fetch_context(session_id: str) -> Context:
    async with get_db_session() as db:
        return await db.get(Context, session_id)

# Use anyio for concurrency primitives
async def run_parallel_tasks(tasks: list[Task]) -> list[Result]:
    async with anyio.create_task_group() as tg:
        results = []
        for task in tasks:
            tg.start_soon(execute_task, task, results)
    return results
```

### Pydantic Models for Everything

```python
from pydantic import BaseModel, Field

class Task(BaseModel):
    """A single executable task in the decomposition."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    wave: int = Field(ge=0)
    estimated_complexity: Literal["low", "medium", "high"]

    model_config = ConfigDict(frozen=True)
```

### Dependency Injection

```python
from functools import lru_cache

@lru_cache
def get_settings() -> Settings:
    return Settings()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
```

## Error Handling

### Custom Exceptions

```python
class KappaError(Exception):
    """Base exception for all Kappa errors."""

class DecompositionError(KappaError):
    """Failed to decompose requirements."""

class SessionError(KappaError):
    """Claude session error."""

class ConflictError(KappaError):
    """Unresolvable code conflict."""
```

### Graceful Degradation

```python
async def execute_with_retry(
    task: Task,
    max_retries: int = 3,
) -> Result:
    for attempt in range(max_retries):
        try:
            return await execute_task(task)
        except SessionError as e:
            if attempt == max_retries - 1:
                raise
            await anyio.sleep(2 ** attempt)  # Exponential backoff
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | - | Anthropic API key |
| `DATABASE_URL` | Yes | - | PostgreSQL connection URL |
| `KAPPA_LOG_LEVEL` | No | INFO | Logging level |
| `KAPPA_MAX_PARALLEL_SESSIONS` | No | 5 | Max concurrent sessions |
| `KAPPA_SESSION_TIMEOUT` | No | 3600 | Session timeout (seconds) |
| `KAPPA_DEBUG` | No | false | Enable debug mode |

## Quick Reference

### Run Development Server

```bash
poetry install
cp .env.example .env
# Edit .env with your credentials
poetry run kappa --help
```

### Run Tests

```bash
poetry run pytest                    # All tests
poetry run pytest tests/unit         # Unit only
poetry run pytest -x                 # Stop on first failure
poetry run pytest -k "decomposition" # Filter by name
```

### Database Operations

```bash
# Setup
psql -d kappa_db -f scripts/setup_db.sql

# Migrations (when needed)
poetry run alembic upgrade head
```

### Code Quality

```bash
poetry run black src tests           # Format
poetry run ruff check src tests      # Lint
poetry run mypy src                  # Type check
```
