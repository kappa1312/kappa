"""LangGraph node implementations for Kappa pipeline stages."""

from datetime import datetime
from typing import Any

from loguru import logger

from src.core.state import KappaState, ProjectStatus


async def initialize_node(state: KappaState) -> dict[str, Any]:
    """
    Initialize the Kappa execution pipeline.

    Sets up the project directory, validates configuration,
    and prepares for decomposition.

    Args:
        state: Current Kappa state.

    Returns:
        State updates with initialization results.
    """
    logger.info(f"Initializing project: {state.get('project_name')}")

    project_path = state.get("project_path", "")

    # Ensure project directory exists
    from pathlib import Path

    path = Path(project_path)
    path.mkdir(parents=True, exist_ok=True)

    # Create logs directory
    (path / "logs").mkdir(exist_ok=True)

    return {
        "status": ProjectStatus.DECOMPOSING.value,
        "started_at": datetime.utcnow().isoformat(),
    }


async def decomposition_node(state: KappaState) -> dict[str, Any]:
    """
    Decompose the specification into executable tasks.

    Parses the natural language specification, generates tasks,
    and builds the dependency graph.

    Args:
        state: Current Kappa state.

    Returns:
        State updates with tasks, dependencies, and waves.
    """
    logger.info("Starting specification decomposition")

    spec = state.get("specification", "")

    try:
        from src.decomposition.task_generator import TaskGenerator
        from src.decomposition.dependency_resolver import DependencyResolver

        # Generate tasks from specification
        generator = TaskGenerator()
        tasks = await generator.generate(spec)

        logger.info(f"Generated {len(tasks)} tasks")

        # Build dependency graph and assign waves
        resolver = DependencyResolver()
        dependency_graph = resolver.build_graph(tasks)
        waves = resolver.assign_waves(tasks, dependency_graph)

        logger.info(f"Organized into {len(waves)} execution waves")

        return {
            "tasks": [task.model_dump() for task in tasks],
            "dependency_graph": dependency_graph,
            "waves": waves,
            "total_waves": len(waves),
            "status": ProjectStatus.RUNNING.value,
        }

    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        return {
            "status": ProjectStatus.FAILED.value,
            "error": f"Decomposition failed: {e}",
        }


async def wave_execution_node(state: KappaState) -> dict[str, Any]:
    """
    Execute all tasks in the current wave.

    Spawns parallel Claude sessions for independent tasks
    and waits for completion.

    Args:
        state: Current Kappa state.

    Returns:
        State updates with execution results.
    """
    current_wave = state.get("current_wave", 0)
    waves = state.get("waves", [])

    if current_wave >= len(waves):
        logger.info("All waves completed")
        return {}

    wave_task_ids = waves[current_wave]
    tasks = state.get("tasks", [])
    wave_tasks = [t for t in tasks if t["id"] in wave_task_ids]

    logger.info(f"Executing wave {current_wave + 1}/{len(waves)} with {len(wave_tasks)} tasks")

    try:
        from src.sessions.router import SessionRouter
        from src.core.config import get_settings

        settings = get_settings()
        router = SessionRouter(
            max_sessions=settings.kappa_max_parallel_sessions,
            timeout=settings.kappa_session_timeout,
        )

        results = await router.execute_tasks(wave_tasks, state)

        # Categorize results
        completed = []
        failed = []
        task_results = []

        for result in results:
            task_results.append(result.model_dump())
            if result.success:
                completed.append(result.task_id)
            else:
                failed.append(result.task_id)

        logger.info(
            f"Wave {current_wave + 1} complete: "
            f"{len(completed)} succeeded, {len(failed)} failed"
        )

        return {
            "current_wave": current_wave + 1,
            "completed_tasks": state.get("completed_tasks", []) + completed,
            "failed_tasks": state.get("failed_tasks", []) + failed,
            "task_results": state.get("task_results", []) + task_results,
            "active_sessions": {},  # Clear active sessions
        }

    except Exception as e:
        logger.error(f"Wave execution failed: {e}")
        return {
            "status": ProjectStatus.FAILED.value,
            "error": f"Wave execution failed: {e}",
        }


async def conflict_resolution_node(state: KappaState) -> dict[str, Any]:
    """
    Detect and resolve conflicts between session outputs.

    Analyzes file modifications from multiple sessions
    and applies resolution strategies.

    Args:
        state: Current Kappa state.

    Returns:
        State updates with conflict resolutions.
    """
    logger.info("Checking for conflicts")

    try:
        from src.conflict.detector import ConflictDetector
        from src.conflict.resolver import ConflictResolver

        # Detect conflicts
        detector = ConflictDetector()
        task_results = state.get("task_results", [])
        conflicts = await detector.detect(task_results)

        if not conflicts:
            logger.info("No conflicts detected")
            return {"conflicts": []}

        logger.info(f"Detected {len(conflicts)} conflicts")

        # Attempt resolution
        resolver = ConflictResolver()
        resolved_conflicts = []

        for conflict in conflicts:
            try:
                resolution = await resolver.resolve(conflict)
                conflict["resolution"] = resolution
                conflict["resolved_at"] = datetime.utcnow().isoformat()
                resolved_conflicts.append(conflict)
                logger.info(f"Resolved conflict in: {conflict['file_path']}")
            except Exception as e:
                logger.warning(f"Could not auto-resolve conflict: {e}")
                conflict["resolution"] = None
                resolved_conflicts.append(conflict)

        return {
            "conflicts": resolved_conflicts,
            "status": ProjectStatus.RESOLVING_CONFLICTS.value,
        }

    except Exception as e:
        logger.error(f"Conflict resolution failed: {e}")
        return {
            "error": f"Conflict resolution failed: {e}",
        }


async def finalize_node(state: KappaState) -> dict[str, Any]:
    """
    Finalize the execution and generate output summary.

    Runs final validation, generates reports, and
    marks the project as complete.

    Args:
        state: Current Kappa state.

    Returns:
        State updates with final output.
    """
    logger.info("Finalizing execution")

    completed_tasks = state.get("completed_tasks", [])
    failed_tasks = state.get("failed_tasks", [])
    total_tasks = len(state.get("tasks", []))

    # Calculate metrics
    success_rate = len(completed_tasks) / total_tasks if total_tasks > 0 else 0
    started_at = datetime.fromisoformat(state.get("started_at", datetime.utcnow().isoformat()))
    duration = (datetime.utcnow() - started_at).total_seconds()

    # Generate summary
    summary_lines = [
        f"Project: {state.get('project_name')}",
        f"Status: {'Completed' if not failed_tasks else 'Completed with errors'}",
        f"Tasks: {len(completed_tasks)}/{total_tasks} succeeded",
        f"Duration: {duration:.1f} seconds",
        f"Success rate: {success_rate:.1%}",
    ]

    if failed_tasks:
        summary_lines.append(f"Failed tasks: {', '.join(failed_tasks)}")

    conflicts = state.get("conflicts", [])
    if conflicts:
        unresolved = [c for c in conflicts if not c.get("resolution")]
        summary_lines.append(f"Conflicts: {len(conflicts)} total, {len(unresolved)} unresolved")

    final_output = "\n".join(summary_lines)

    logger.info("Execution finalized")
    logger.info(final_output)

    # Determine final status
    if failed_tasks and len(failed_tasks) == total_tasks:
        final_status = ProjectStatus.FAILED.value
    elif failed_tasks:
        final_status = ProjectStatus.COMPLETED.value  # Partial success
    else:
        final_status = ProjectStatus.COMPLETED.value

    return {
        "status": final_status,
        "final_output": final_output,
        "completed_at": datetime.utcnow().isoformat(),
        "total_duration_seconds": duration,
    }
