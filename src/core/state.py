"""State management types and utilities for Kappa orchestration."""

from datetime import datetime
from enum import Enum
from typing import Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Status of a task in the execution pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SessionStatus(str, Enum):
    """Status of a Claude session."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ProjectStatus(str, Enum):
    """Status of a project execution."""

    PENDING = "pending"
    DECOMPOSING = "decomposing"
    RUNNING = "running"
    RESOLVING_CONFLICTS = "resolving_conflicts"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskInfo(BaseModel):
    """Information about a single task."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    wave: int = Field(ge=0, default=0)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_session_id: str | None = None


class SessionInfo(BaseModel):
    """Information about a Claude session."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str | None = None
    status: SessionStatus = SessionStatus.STARTING
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error: str | None = None
    output: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result of a completed task."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    session_id: str
    success: bool
    output: str | None = None
    error: str | None = None
    files_modified: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)


class Artifact(BaseModel):
    """An artifact produced by task execution."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    artifact_type: str  # "file", "test", "documentation", etc.
    path: str
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Conflict(BaseModel):
    """A conflict detected between session outputs."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str
    session_a_id: str
    session_b_id: str
    conflict_type: str  # "merge", "overwrite", "semantic"
    description: str
    resolution: str | None = None
    resolved_at: datetime | None = None


class KappaState(TypedDict, total=False):
    """
    LangGraph state for Kappa orchestration.

    This TypedDict defines the complete state that flows through
    the LangGraph execution pipeline.
    """

    # Project identification
    project_id: str
    project_name: str

    # Input
    specification: str
    project_path: str
    config: dict[str, Any]

    # Decomposition phase
    tasks: list[dict[str, Any]]  # Serialized TaskInfo objects
    dependency_graph: dict[str, list[str]]  # task_id -> [dependency_ids]
    waves: list[list[str]]  # Wave index -> [task_ids]
    total_waves: int

    # Execution phase
    current_wave: int
    active_sessions: dict[str, dict[str, Any]]  # session_id -> SessionInfo dict
    completed_tasks: list[str]  # task_ids
    failed_tasks: list[str]  # task_ids
    skipped_tasks: list[str]  # task_ids (due to failed dependencies)

    # Results
    task_results: list[dict[str, Any]]  # Serialized TaskResult objects
    artifacts: list[dict[str, Any]]  # Serialized Artifact objects
    conflicts: list[dict[str, Any]]  # Serialized Conflict objects

    # Final output
    status: str  # ProjectStatus value
    final_output: str | None
    error: str | None

    # Metadata
    started_at: str  # ISO format datetime
    completed_at: str | None  # ISO format datetime
    total_duration_seconds: float
    total_tokens_used: int


def create_initial_state(
    specification: str,
    project_path: str,
    project_name: str | None = None,
    config: dict[str, Any] | None = None,
) -> KappaState:
    """Create initial state for a new Kappa execution.

    Args:
        specification: Natural language project specification.
        project_path: Path to the project directory.
        project_name: Optional project name (derived from path if not provided).
        config: Optional configuration overrides.

    Returns:
        Initial KappaState ready for graph execution.

    Example:
        >>> state = create_initial_state(
        ...     "Build a REST API with auth",
        ...     "/path/to/project"
        ... )
        >>> state["status"]
        'pending'
    """
    import os

    return KappaState(
        project_id=str(uuid4()),
        project_name=project_name or os.path.basename(project_path),
        specification=specification,
        project_path=project_path,
        config=config or {},
        tasks=[],
        dependency_graph={},
        waves=[],
        total_waves=0,
        current_wave=0,
        active_sessions={},
        completed_tasks=[],
        failed_tasks=[],
        skipped_tasks=[],
        task_results=[],
        artifacts=[],
        conflicts=[],
        status=ProjectStatus.PENDING.value,
        final_output=None,
        error=None,
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
        total_duration_seconds=0.0,
        total_tokens_used=0,
    )


def get_pending_tasks(state: KappaState) -> list[str]:
    """Get task IDs that are pending execution.

    Args:
        state: Current Kappa state.

    Returns:
        List of pending task IDs.
    """
    completed = set(state.get("completed_tasks", []))
    failed = set(state.get("failed_tasks", []))
    skipped = set(state.get("skipped_tasks", []))
    done = completed | failed | skipped

    return [
        task["id"]
        for task in state.get("tasks", [])
        if task["id"] not in done
    ]


def get_wave_tasks(state: KappaState, wave: int) -> list[str]:
    """Get task IDs for a specific wave.

    Args:
        state: Current Kappa state.
        wave: Wave index.

    Returns:
        List of task IDs in the wave.
    """
    waves = state.get("waves", [])
    if wave < len(waves):
        return waves[wave]
    return []


def is_task_ready(state: KappaState, task_id: str) -> bool:
    """Check if a task's dependencies are satisfied.

    Args:
        state: Current Kappa state.
        task_id: Task ID to check.

    Returns:
        True if all dependencies are completed.
    """
    deps = state.get("dependency_graph", {}).get(task_id, [])
    completed = set(state.get("completed_tasks", []))
    return all(dep in completed for dep in deps)
