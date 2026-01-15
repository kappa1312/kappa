"""LangGraph edge routing functions for Kappa pipeline.

This module defines the routing logic that determines how execution
flows between nodes based on state conditions.
"""

from typing import Literal

from loguru import logger

from src.graph.state import ExecutionStatus, KappaState

# =============================================================================
# PRIMARY ROUTING FUNCTIONS
# =============================================================================


def route_after_parsing(
    state: KappaState,
) -> Literal["generate_tasks", "handle_error"]:
    """
    Route after parsing requirements.

    Determines whether to proceed to task generation or handle
    errors encountered during parsing.

    Args:
        state: Current Kappa state.

    Returns:
        "generate_tasks" if parsing succeeded, "handle_error" if failed.
    """
    status = state.get("status", "")
    error = state.get("error")

    if status == ExecutionStatus.FAILED.value or error:
        logger.info("Routing to handle_error due to parsing failure")
        return "handle_error"

    requirements = state.get("requirements")
    if not requirements:
        logger.warning("No requirements parsed, routing to handle_error")
        return "handle_error"

    logger.info("Parsing succeeded, routing to generate_tasks")
    return "generate_tasks"


def route_after_task_generation(
    state: KappaState,
) -> Literal["resolve_dependencies", "handle_error"]:
    """
    Route after task generation.

    Determines whether to proceed to dependency resolution or handle
    errors encountered during task generation.

    Args:
        state: Current Kappa state.

    Returns:
        "resolve_dependencies" if tasks generated, "handle_error" if failed.
    """
    status = state.get("status", "")
    error = state.get("error")

    if status == ExecutionStatus.FAILED.value or error:
        logger.info("Routing to handle_error due to task generation failure")
        return "handle_error"

    tasks = state.get("tasks", [])
    if not tasks:
        logger.warning("No tasks generated, routing to handle_error")
        return "handle_error"

    logger.info(f"Task generation succeeded ({len(tasks)} tasks), routing to resolve_dependencies")
    return "resolve_dependencies"


def route_after_dependency_resolution(
    state: KappaState,
) -> Literal["execute_wave", "handle_error"]:
    """
    Route after dependency resolution.

    Determines whether to start wave execution or handle errors
    from dependency resolution (e.g., cycle detection).

    Args:
        state: Current Kappa state.

    Returns:
        "execute_wave" if ready to execute, "handle_error" if failed.
    """
    status = state.get("status", "")
    error = state.get("error")

    if status == ExecutionStatus.FAILED.value or error:
        logger.info("Routing to handle_error due to dependency resolution failure")
        return "handle_error"

    graph_data = state.get("dependency_graph", {})
    waves = graph_data.get("waves", [])

    if not waves:
        logger.warning("No execution waves, routing to handle_error")
        return "handle_error"

    logger.info(f"Dependency resolution complete ({len(waves)} waves), routing to execute_wave")
    return "execute_wave"


def should_continue_execution(
    state: KappaState,
) -> Literal["execute_wave", "merge_outputs", "handle_error"]:
    """
    Determine next step after wave execution.

    This is the main loop control function that decides whether to:
    - Execute another wave
    - Proceed to merge outputs
    - Handle critical failures

    Args:
        state: Current Kappa state.

    Returns:
        Next node: "execute_wave", "merge_outputs", or "handle_error".
    """
    status = state.get("status", "")

    # Check for fatal error
    if status == ExecutionStatus.FAILED.value:
        logger.info("Routing to handle_error due to execution failure")
        return "handle_error"

    current_wave = state.get("current_wave", 0)
    total_waves = state.get("total_waves", 0)

    # Check for critical failures (>50% of tasks failed)
    completed_tasks = state.get("completed_tasks", [])
    failed_tasks = state.get("failed_tasks", [])
    total_executed = len(completed_tasks) + len(failed_tasks)

    if total_executed > 0:
        failure_rate = len(failed_tasks) / total_executed
        if failure_rate > 0.5:
            logger.warning(f"Critical failure rate ({failure_rate:.1%}), routing to handle_error")
            return "handle_error"

    # More waves to execute?
    if current_wave < total_waves:
        logger.info(f"Routing to execute_wave ({current_wave + 1}/{total_waves})")
        return "execute_wave"

    # All waves done, proceed to merge
    logger.info("All waves complete, routing to merge_outputs")
    return "merge_outputs"


def route_after_merge(
    state: KappaState,
) -> Literal["validate", "handle_error"]:
    """
    Route after output merging.

    Determines whether to proceed to validation or handle unresolved
    critical conflicts.

    Args:
        state: Current Kappa state.

    Returns:
        "validate" to proceed, "handle_error" for critical conflicts.
    """
    status = state.get("status", "")

    if status == ExecutionStatus.FAILED.value:
        logger.info("Routing to handle_error due to merge failure")
        return "handle_error"

    # Check for unresolved critical conflicts
    conflicts = state.get("conflicts", [])
    unresolved_critical = [
        c for c in conflicts if not c.get("resolved") and c.get("conflict_type") == "critical"
    ]

    if unresolved_critical:
        logger.warning(
            f"{len(unresolved_critical)} critical conflicts unresolved, " f"routing to handle_error"
        )
        return "handle_error"

    logger.info("Merge complete, routing to validate")
    return "validate"


def route_after_validation(
    state: KappaState,
) -> Literal["end", "handle_error"]:
    """
    Route after validation.

    Determines whether execution completed successfully or needs
    error handling.

    Args:
        state: Current Kappa state.

    Returns:
        "end" if validation passed, "handle_error" if failed.
    """
    status = state.get("status", "")

    if status == ExecutionStatus.VALIDATION_FAILED.value:
        logger.info("Routing to handle_error due to validation failure")
        return "handle_error"

    if status == ExecutionStatus.FAILED.value:
        logger.info("Routing to handle_error due to general failure")
        return "handle_error"

    logger.info("Validation passed, execution complete")
    return "end"


# =============================================================================
# LEGACY COMPATIBILITY ROUTING
# =============================================================================


def route_after_decomposition(
    state: KappaState,
) -> Literal["execute_wave", "finalize"]:
    """
    Route after decomposition completes (legacy compatibility).

    Args:
        state: Current Kappa state.

    Returns:
        Next node name: "execute_wave" if tasks exist, "finalize" otherwise.
    """
    status = state.get("status", "")

    if status == ExecutionStatus.FAILED.value:
        logger.info("Routing to finalize due to decomposition failure")
        return "finalize"

    tasks = state.get("tasks", [])
    if not tasks:
        logger.info("No tasks generated, routing to finalize")
        return "finalize"

    logger.info(f"Routing to execute_wave with {len(tasks)} tasks")
    return "execute_wave"


def route_after_wave(
    state: KappaState,
) -> Literal["execute_wave", "resolve_conflicts", "finalize"]:
    """
    Route after wave execution completes (legacy compatibility).

    Args:
        state: Current Kappa state.

    Returns:
        Next node name based on execution state.
    """
    status = state.get("status", "")

    # Check for fatal error
    if status == ExecutionStatus.FAILED.value:
        logger.info("Routing to finalize due to execution failure")
        return "finalize"

    current_wave = state.get("current_wave", 0)
    total_waves = state.get("total_waves", 0)

    # Check for critical failures
    completed_tasks = state.get("completed_tasks", [])
    failed_tasks = state.get("failed_tasks", [])
    total_executed = len(completed_tasks) + len(failed_tasks)

    if total_executed > 0 and len(failed_tasks) / total_executed > 0.5:
        logger.warning("More than 50% of tasks failed, routing to finalize")
        return "finalize"

    # More waves to execute?
    if current_wave < total_waves:
        logger.info(f"Routing to execute_wave ({current_wave + 1}/{total_waves})")
        return "execute_wave"

    # All waves done, check for conflicts
    task_results = state.get("task_results", [])

    # Check if multiple sessions modified same files
    modified_files: dict[str, list[str]] = {}
    for result in task_results:
        for file_path in result.get("files_modified", []) + result.get("files_created", []):
            if file_path not in modified_files:
                modified_files[file_path] = []
            modified_files[file_path].append(result.get("task_id", ""))

    potential_conflicts = [f for f, tasks in modified_files.items() if len(tasks) > 1]

    if potential_conflicts:
        logger.info(f"Potential conflicts in {len(potential_conflicts)} files, routing to resolve")
        return "resolve_conflicts"

    logger.info("All waves complete, no conflicts, routing to finalize")
    return "finalize"


def route_after_resolution(
    state: KappaState,
) -> Literal["finalize"]:
    """
    Route after conflict resolution completes (legacy compatibility).

    Args:
        state: Current Kappa state.

    Returns:
        Always routes to finalize.
    """
    conflicts = state.get("conflicts", [])
    unresolved = [c for c in conflicts if not c.get("resolved")]

    if unresolved:
        logger.warning(f"{len(unresolved)} conflicts could not be auto-resolved")
    else:
        logger.info("All conflicts resolved")

    return "finalize"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def should_abort_execution(state: KappaState) -> bool:
    """
    Check if execution should be aborted due to failures.

    Args:
        state: Current Kappa state.

    Returns:
        True if execution should abort.
    """
    status = state.get("status", "")

    if status == ExecutionStatus.FAILED.value:
        return True

    # Check failure rate
    completed_tasks = state.get("completed_tasks", [])
    failed_tasks = state.get("failed_tasks", [])
    total_executed = len(completed_tasks) + len(failed_tasks)

    if total_executed > 0 and len(failed_tasks) / total_executed > 0.5:
        logger.warning("More than 50% of tasks failed, aborting")
        return True

    return False


def get_ready_tasks(state: KappaState) -> list[str]:
    """
    Get task IDs ready for execution (dependencies satisfied).

    Args:
        state: Current Kappa state.

    Returns:
        List of task IDs whose dependencies are satisfied.
    """
    current_wave = state.get("current_wave", 0)
    graph_data = state.get("dependency_graph", {})
    waves = graph_data.get("waves", [])

    if current_wave >= len(waves):
        return []

    wave_tasks = waves[current_wave]
    completed = set(state.get("completed_tasks", []))
    edges = graph_data.get("edges", {})

    ready_tasks = []
    for task_id in wave_tasks:
        deps = edges.get(task_id, [])
        if all(dep in completed for dep in deps):
            ready_tasks.append(task_id)

    return ready_tasks


def get_next_wave_tasks(state: KappaState) -> list[str]:
    """
    Get task IDs for the next wave.

    Args:
        state: Current Kappa state.

    Returns:
        List of task IDs in the next wave.
    """
    current_wave = state.get("current_wave", 0)
    graph_data = state.get("dependency_graph", {})
    waves = graph_data.get("waves", [])

    if current_wave >= len(waves):
        return []

    return waves[current_wave]


def has_unresolved_conflicts(state: KappaState) -> bool:
    """
    Check if there are unresolved conflicts.

    Args:
        state: Current Kappa state.

    Returns:
        True if unresolved conflicts exist.
    """
    conflicts = state.get("conflicts", [])
    return any(not c.get("resolved") for c in conflicts)


def get_execution_progress(state: KappaState) -> dict:
    """
    Get execution progress information.

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

    return {
        "total_tasks": total,
        "completed_tasks": len(completed),
        "failed_tasks": len(failed),
        "skipped_tasks": len(skipped),
        "pending_tasks": total - done,
        "progress_percent": (done / total * 100) if total > 0 else 0,
        "current_wave": state.get("current_wave", 0),
        "total_waves": state.get("total_waves", 0),
    }
