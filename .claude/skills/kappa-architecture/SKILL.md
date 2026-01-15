# Kappa Architecture Skill

## Activation

This skill activates when working on:
- Core orchestration logic
- Session management
- Task decomposition
- System design decisions

## Architecture Overview

### Master-Clone Pattern

Kappa uses a hierarchical orchestration model where a central orchestrator (master) manages multiple parallel Claude Code sessions (clones):

```
Master Orchestrator
├── Decomposition Engine → Parse specs into tasks
├── Dependency Resolver → Build execution graph
├── Session Manager → Spawn/manage clones
├── Context Manager → Share knowledge
└── Conflict Resolver → Merge results
```

### Data Flow

```
User Spec → Parser → Tasks → Waves → Sessions → Results → Merger → Output
                         ↓
                   PostgreSQL
                   (Context DB)
```

### Key Components

1. **Orchestrator** (`src/core/orchestrator.py`)
   - Entry point for all operations
   - Coordinates the full lifecycle
   - Handles graceful shutdown

2. **State Machine** (`src/graph/`)
   - LangGraph-based state management
   - Durable execution with checkpoints
   - Human-in-the-loop support

3. **Task Decomposition** (`src/decomposition/`)
   - NLP parsing of requirements
   - Dependency graph generation
   - Wave assignment for parallel execution

4. **Session Management** (`src/sessions/`)
   - Claude Agent SDK integration
   - Connection pooling
   - Timeout and retry handling

5. **Knowledge Base** (`src/knowledge/`)
   - PostgreSQL with asyncpg
   - Cross-session context sharing
   - Decision logging and retrieval

### Design Decisions

#### Why PostgreSQL over SQLite?
- Concurrent access from multiple sessions
- Better transaction isolation
- Production-ready from day one
- Native async support via asyncpg

#### Why LangGraph over raw async?
- Built-in state persistence
- Automatic recovery from failures
- Visualization and debugging
- Human-in-the-loop patterns

#### Why Claude Agent SDK over raw API?
- Terminal automation built-in
- File operation tools
- Session management
- Proper error handling

### State Management

```python
class KappaState(TypedDict):
    """LangGraph state for Kappa orchestration."""

    # Input
    specification: str
    project_path: str

    # Decomposition
    tasks: list[Task]
    dependency_graph: dict[str, list[str]]
    waves: list[list[str]]

    # Execution
    current_wave: int
    active_sessions: dict[str, SessionInfo]
    completed_tasks: list[str]
    failed_tasks: list[str]

    # Results
    artifacts: list[Artifact]
    conflicts: list[Conflict]
    final_result: Result | None
```

### Error Recovery

1. **Session Failures**: Automatic retry with exponential backoff
2. **Task Failures**: Mark failed, continue with non-dependent tasks
3. **Conflict Failures**: Queue for manual resolution
4. **System Failures**: Resume from last checkpoint

### Scaling Considerations

- **Horizontal**: Run multiple Kappa instances with shared PostgreSQL
- **Vertical**: Increase `KAPPA_MAX_PARALLEL_SESSIONS`
- **Cost**: Monitor API usage, implement rate limiting

## Best Practices

1. Always use async/await for I/O
2. Use Pydantic models for all data structures
3. Log all significant events with loguru
4. Handle errors gracefully with retries
5. Write tests for all new functionality
