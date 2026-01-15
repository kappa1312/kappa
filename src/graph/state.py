"""LangGraph state definitions for Kappa orchestration.

This module defines the comprehensive state structure that flows
through the LangGraph execution pipeline. It includes typed state
with reducer functions for append-only lists.
"""

import operator
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, NotRequired, TypedDict
from uuid import uuid4

# =============================================================================
# ENUMS
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status of the Kappa execution pipeline."""

    PENDING = "pending"
    PARSING = "parsing"
    GENERATING_TASKS = "generating_tasks"
    RESOLVING_DEPENDENCIES = "resolving_dependencies"
    PLANNING_COMPLETE = "planning_complete"
    EXECUTING = "executing"
    MERGING = "merging"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATION_FAILED = "validation_failed"


class WaveStatus(str, Enum):
    """Status of a wave execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


# =============================================================================
# STATE DEFINITION
# =============================================================================


class KappaState(TypedDict, total=False):
    """
    Complete orchestration state for LangGraph.

    This TypedDict defines all state that flows through the Kappa
    execution pipeline. Fields marked with Annotated[..., operator.add]
    are append-only and accumulate values across nodes.

    State Flow:
        1. parse_requirements: Sets requirements from requirements_text
        2. generate_tasks: Creates tasks from requirements
        3. resolve_dependencies: Builds dependency_graph, assigns waves
        4. execute_wave: Runs tasks, updates completed/failed_tasks
        5. merge_outputs: Resolves conflicts in created_files
        6. validate: Runs validation checks
        7. handle_error: Captures errors

    Example:
        >>> state: KappaState = {
        ...     "project_id": "123",
        ...     "project_name": "my-api",
        ...     "requirements_text": "Build REST API...",
        ...     "workspace_path": "/tmp/my-api",
        ...     "status": ExecutionStatus.PENDING.value,
        ... }
    """

    # =========================================================================
    # PROJECT IDENTIFICATION
    # =========================================================================

    project_id: str
    """Unique identifier for this project execution."""

    project_name: str
    """Human-readable project name."""

    # =========================================================================
    # INPUT
    # =========================================================================

    requirements_text: str
    """Natural language project requirements."""

    workspace_path: str
    """Path to the project workspace directory."""

    config: NotRequired[dict[str, Any]]
    """Optional configuration overrides."""

    # =========================================================================
    # PARSED REQUIREMENTS
    # =========================================================================

    requirements: NotRequired[dict[str, Any]]
    """Parsed ProjectRequirements as dict (from parse_requirements node)."""

    # =========================================================================
    # TASK DECOMPOSITION
    # =========================================================================

    tasks: NotRequired[list[dict[str, Any]]]
    """List of TaskSpec objects as dicts."""

    dependency_graph: NotRequired[dict[str, Any]]
    """DependencyGraph data: {'waves': [...], 'total_waves': int, 'edges': {...}}"""

    # =========================================================================
    # EXECUTION TRACKING
    # =========================================================================

    current_wave: NotRequired[int]
    """Current wave being executed (0-indexed)."""

    total_waves: NotRequired[int]
    """Total number of execution waves."""

    completed_tasks: Annotated[list[str], operator.add]
    """Task IDs that completed successfully (append-only)."""

    failed_tasks: Annotated[list[str], operator.add]
    """Task IDs that failed (append-only)."""

    skipped_tasks: Annotated[list[str], operator.add]
    """Task IDs skipped due to failed dependencies (append-only)."""

    active_sessions: NotRequired[dict[str, str]]
    """Currently active sessions: task_id -> session_id."""

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    global_context: NotRequired[dict[str, Any]]
    """Shared context available to all tasks."""

    wave_contexts: NotRequired[dict[int, dict[str, Any]]]
    """Per-wave context: wave_number -> context dict."""

    task_contexts: NotRequired[dict[str, dict[str, Any]]]
    """Per-task context: task_id -> context dict."""

    # =========================================================================
    # RESULTS
    # =========================================================================

    created_files: Annotated[list[str], operator.add]
    """Files created during execution (append-only)."""

    modified_files: Annotated[list[str], operator.add]
    """Files modified during execution (append-only)."""

    task_results: NotRequired[list[dict[str, Any]]]
    """Results from task execution."""

    conflicts: NotRequired[list[dict[str, Any]]]
    """Detected file conflicts."""

    validation_results: NotRequired[dict[str, Any]]
    """Results from validation step."""

    execution_logs: Annotated[list[dict[str, Any]], operator.add]
    """Execution logs from all nodes (append-only)."""

    # =========================================================================
    # STATUS
    # =========================================================================

    status: str
    """Current execution status (ExecutionStatus value)."""

    error: NotRequired[str]
    """Error message if status is failed."""

    error_node: NotRequired[str]
    """Node where error occurred."""

    # =========================================================================
    # TIMING
    # =========================================================================

    started_at: NotRequired[str]
    """ISO format timestamp when execution started."""

    completed_at: NotRequired[str]
    """ISO format timestamp when execution completed."""

    # =========================================================================
    # METRICS
    # =========================================================================

    total_duration_seconds: NotRequired[float]
    """Total execution duration."""

    total_tokens_used: NotRequired[int]
    """Total tokens consumed across all sessions."""

    wave_metrics: NotRequired[dict[int, dict[str, Any]]]
    """Per-wave metrics: wave_number -> metrics dict."""


# =============================================================================
# STATE FACTORY
# =============================================================================


def create_initial_state(
    requirements_text: str,
    workspace_path: str,
    project_name: str | None = None,
    project_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> KappaState:
    """
    Create initial state for a new Kappa execution.

    Args:
        requirements_text: Natural language project requirements.
        workspace_path: Path to the project workspace directory.
        project_name: Optional project name (defaults to workspace basename).
        project_id: Optional project ID (generated if not provided).
        config: Optional configuration overrides.

    Returns:
        Initial KappaState ready for graph execution.

    Example:
        >>> state = create_initial_state(
        ...     "Build Express API with user auth",
        ...     "/tmp/my-api",
        ...     project_name="my-api"
        ... )
        >>> state["status"]
        'pending'
    """
    import os

    return KappaState(
        # Identification
        project_id=project_id or str(uuid4()),
        project_name=project_name or os.path.basename(workspace_path),
        # Input
        requirements_text=requirements_text,
        workspace_path=workspace_path,
        config=config or {},
        # Execution tracking (append-only lists start empty)
        completed_tasks=[],
        failed_tasks=[],
        skipped_tasks=[],
        created_files=[],
        modified_files=[],
        execution_logs=[],
        # Status
        status=ExecutionStatus.PENDING.value,
        # Timing
        started_at=datetime.utcnow().isoformat(),
    )


# =============================================================================
# STATE HELPERS
# =============================================================================


def get_pending_task_ids(state: KappaState) -> list[str]:
    """
    Get task IDs that are pending execution.

    Args:
        state: Current Kappa state.

    Returns:
        List of pending task IDs.
    """
    completed = set(state.get("completed_tasks", []))
    failed = set(state.get("failed_tasks", []))
    skipped = set(state.get("skipped_tasks", []))
    done = completed | failed | skipped

    return [task["id"] for task in state.get("tasks", []) if task["id"] not in done]


def get_wave_task_ids(state: KappaState, wave_number: int) -> list[str]:
    """
    Get task IDs for a specific wave.

    Args:
        state: Current Kappa state.
        wave_number: Wave index (0-based).

    Returns:
        List of task IDs in the wave.
    """
    graph_data = state.get("dependency_graph", {})
    waves = graph_data.get("waves", [])

    if wave_number < len(waves):
        return waves[wave_number]
    return []


def is_task_ready(state: KappaState, task_id: str) -> bool:
    """
    Check if a task's dependencies are satisfied.

    Args:
        state: Current Kappa state.
        task_id: Task ID to check.

    Returns:
        True if all dependencies are completed.
    """
    graph_data = state.get("dependency_graph", {})
    edges = graph_data.get("edges", {})
    deps = edges.get(task_id, [])
    completed = set(state.get("completed_tasks", []))
    return all(dep in completed for dep in deps)


def get_task_by_id(state: KappaState, task_id: str) -> dict[str, Any] | None:
    """
    Get task dict by ID.

    Args:
        state: Current Kappa state.
        task_id: Task identifier.

    Returns:
        Task dict if found, None otherwise.
    """
    for task in state.get("tasks", []):
        if task.get("id") == task_id:
            return task
    return None


def calculate_progress(state: KappaState) -> dict[str, Any]:
    """
    Calculate execution progress metrics.

    Args:
        state: Current Kappa state.

    Returns:
        Dict with progress metrics.
    """
    tasks = state.get("tasks", [])
    completed = state.get("completed_tasks", [])
    failed = state.get("failed_tasks", [])
    skipped = state.get("skipped_tasks", [])

    total = len(tasks)
    done = len(completed) + len(failed) + len(skipped)

    current_wave = state.get("current_wave", 0)
    total_waves = state.get("total_waves", 0)

    return {
        "total_tasks": total,
        "completed_tasks": len(completed),
        "failed_tasks": len(failed),
        "skipped_tasks": len(skipped),
        "pending_tasks": total - done,
        "progress_percent": (done / total * 100) if total > 0 else 0,
        "current_wave": current_wave,
        "total_waves": total_waves,
        "success_rate": (len(completed) / done * 100) if done > 0 else 0,
    }


def create_execution_log(
    node: str,
    action: str,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    Create an execution log entry.

    Args:
        node: Name of the node creating the log.
        action: Action being performed.
        details: Optional additional details.
        error: Optional error message.

    Returns:
        Log entry dict.
    """
    log: dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "node": node,
        "action": action,
    }

    if details:
        log["details"] = details

    if error:
        log["error"] = error
        log["level"] = "error"
    else:
        log["level"] = "info"

    return log


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


def state_to_dict(state: KappaState) -> dict[str, Any]:
    """
    Convert KappaState to a plain dict for serialization.

    Args:
        state: Kappa state to convert.

    Returns:
        Plain dict representation.
    """
    return dict(state)


def state_from_dict(data: dict[str, Any]) -> KappaState:
    """
    Create KappaState from a plain dict.

    Args:
        data: Dict with state data.

    Returns:
        KappaState instance.
    """
    return KappaState(**data)


def get_state_summary(state: KappaState) -> str:
    """
    Generate a human-readable summary of the state.

    Args:
        state: Current Kappa state.

    Returns:
        Summary string.
    """
    progress = calculate_progress(state)

    lines = [
        f"Project: {state.get('project_name', 'Unknown')}",
        f"Status: {state.get('status', 'unknown')}",
        f"Progress: {progress['progress_percent']:.1f}%",
        f"Tasks: {progress['completed_tasks']}/{progress['total_tasks']} completed",
        f"Wave: {progress['current_wave']}/{progress['total_waves']}",
    ]

    if progress["failed_tasks"] > 0:
        lines.append(f"Failed: {progress['failed_tasks']} tasks")

    error = state.get("error")
    if error:
        lines.append(f"Error: {error}")

    return "\n".join(lines)


# =============================================================================
# STATE RECONSTRUCTION
# =============================================================================


async def reconstruct_state_from_db(project_id: str) -> KappaState | None:
    """
    Reconstruct state from database for resumption.

    Args:
        project_id: Project ID to load.

    Returns:
        Reconstructed KappaState or None if not found.
    """
    from src.knowledge.database import get_db_session

    try:
        async with get_db_session() as session:
            from sqlalchemy import select

            from src.knowledge.models import Project

            result = await session.execute(select(Project).where(Project.id == project_id))
            project = result.scalar_one_or_none()

            if not project:
                return None

            # Reconstruct state from project record
            return KappaState(
                project_id=str(project.id),
                project_name=project.name,
                requirements_text=project.specification or "",
                workspace_path=project.workspace_path or "",
                status=project.status or ExecutionStatus.PENDING.value,
                started_at=project.created_at.isoformat() if project.created_at else None,
                completed_at=project.completed_at.isoformat() if project.completed_at else None,
                # These would need to be stored separately or in project metadata
                completed_tasks=[],
                failed_tasks=[],
                skipped_tasks=[],
                created_files=[],
                modified_files=[],
                execution_logs=[],
            )

    except Exception:
        return None


async def save_state_to_db(state: KappaState) -> bool:
    """
    Save state to database for persistence.

    Args:
        state: State to save.

    Returns:
        True if saved successfully.
    """
    from src.knowledge.database import get_db_session

    try:
        async with get_db_session() as session:
            from sqlalchemy import update

            from src.knowledge.models import Project

            await session.execute(
                update(Project)
                .where(Project.id == state.get("project_id"))
                .values(
                    status=state.get("status"),
                    completed_at=(
                        datetime.fromisoformat(state["completed_at"])
                        if state.get("completed_at")
                        else None
                    ),
                )
            )
            await session.commit()
            return True

    except Exception:
        return False
