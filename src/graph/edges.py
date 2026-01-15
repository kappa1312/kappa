"""LangGraph edge routing functions for Kappa pipeline."""

from typing import Literal

from loguru import logger

from src.core.state import KappaState, ProjectStatus


def route_after_decomposition(
    state: KappaState,
) -> Literal["execute_wave", "finalize"]:
    """
    Route after decomposition completes.

    Args:
        state: Current Kappa state.

    Returns:
        Next node name: "execute_wave" if tasks exist, "finalize" otherwise.
    """
    status = state.get("status", "")

    if status == ProjectStatus.FAILED.value:
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
    Route after wave execution completes.

    Args:
        state: Current Kappa state.

    Returns:
        Next node name based on execution state.
    """
    status = state.get("status", "")

    # Check for fatal error
    if status == ProjectStatus.FAILED.value:
        logger.info("Routing to finalize due to execution failure")
        return "finalize"

    current_wave = state.get("current_wave", 0)
    total_waves = state.get("total_waves", 0)

    # More waves to execute?
    if current_wave < total_waves:
        logger.info(f"Routing to execute_wave ({current_wave + 1}/{total_waves})")
        return "execute_wave"

    # All waves done, check for conflicts
    task_results = state.get("task_results", [])

    # Check if multiple sessions modified same files
    modified_files: dict[str, list[str]] = {}
    for result in task_results:
        for file_path in result.get("files_modified", []):
            if file_path not in modified_files:
                modified_files[file_path] = []
            modified_files[file_path].append(result.get("session_id", ""))

    potential_conflicts = [f for f, sessions in modified_files.items() if len(sessions) > 1]

    if potential_conflicts:
        logger.info(f"Potential conflicts in {len(potential_conflicts)} files, routing to resolve")
        return "resolve_conflicts"

    logger.info("All waves complete, no conflicts, routing to finalize")
    return "finalize"


def route_after_resolution(
    state: KappaState,
) -> Literal["finalize"]:
    """
    Route after conflict resolution completes.

    Args:
        state: Current Kappa state.

    Returns:
        Always routes to finalize.
    """
    conflicts = state.get("conflicts", [])
    unresolved = [c for c in conflicts if not c.get("resolution")]

    if unresolved:
        logger.warning(f"{len(unresolved)} conflicts could not be auto-resolved")
    else:
        logger.info("All conflicts resolved")

    return "finalize"


def should_continue_execution(state: KappaState) -> bool:
    """
    Check if execution should continue or abort.

    Args:
        state: Current Kappa state.

    Returns:
        True if execution should continue.
    """
    status = state.get("status", "")

    if status == ProjectStatus.FAILED.value:
        return False

    # Check if too many tasks have failed
    failed_tasks = state.get("failed_tasks", [])
    total_tasks = len(state.get("tasks", []))

    if total_tasks > 0 and len(failed_tasks) / total_tasks > 0.5:
        logger.warning("More than 50% of tasks failed, aborting")
        return False

    return True


def get_next_wave_tasks(state: KappaState) -> list[str]:
    """
    Get task IDs ready for execution in the next wave.

    Args:
        state: Current Kappa state.

    Returns:
        List of task IDs whose dependencies are satisfied.
    """
    current_wave = state.get("current_wave", 0)
    waves = state.get("waves", [])

    if current_wave >= len(waves):
        return []

    wave_tasks = waves[current_wave]
    completed = set(state.get("completed_tasks", []))
    dependency_graph = state.get("dependency_graph", {})

    ready_tasks = []
    for task_id in wave_tasks:
        deps = dependency_graph.get(task_id, [])
        if all(dep in completed for dep in deps):
            ready_tasks.append(task_id)

    return ready_tasks
