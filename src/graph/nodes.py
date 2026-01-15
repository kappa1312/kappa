"""LangGraph node implementations for Kappa pipeline stages.

This module defines the 7 core nodes of the Kappa orchestration graph:
1. parse_requirements - Parse natural language into ProjectRequirements
2. generate_tasks - Generate TaskSpecs from ProjectRequirements
3. resolve_dependencies - Build dependency graph and assign waves
4. execute_wave - Execute all tasks in current wave in parallel
5. merge_outputs - Merge outputs and resolve conflicts
6. validate - Validate all outputs
7. handle_error - Handle execution errors
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.graph.state import (
    ExecutionStatus,
    KappaState,
    create_execution_log,
)


# =============================================================================
# NODE 1: PARSE REQUIREMENTS
# =============================================================================


async def parse_requirements_node(state: KappaState) -> dict[str, Any]:
    """
    Parse natural language requirements into structured ProjectRequirements.

    This node takes the raw requirements_text and uses the RequirementsParser
    to extract structured information including project type, tech stack,
    features, pages, and integrations.

    Args:
        state: Current Kappa state with requirements_text.

    Returns:
        State updates with parsed requirements dict.

    State Changes:
        - requirements: Parsed ProjectRequirements as dict
        - status: ExecutionStatus.PARSING -> GENERATING_TASKS
        - execution_logs: Appended with parsing log
    """
    logger.info("Parsing requirements")

    requirements_text = state.get("requirements_text", "")
    project_name = state.get("project_name", "project")

    if not requirements_text:
        logger.error("No requirements text provided")
        return {
            "status": ExecutionStatus.FAILED.value,
            "error": "No requirements text provided",
            "error_node": "parse_requirements",
            "execution_logs": [
                create_execution_log(
                    "parse_requirements",
                    "error",
                    error="No requirements text provided",
                )
            ],
        }

    try:
        from src.decomposition.parser import RequirementsParser

        parser = RequirementsParser()
        requirements = await parser.parse(requirements_text)

        # Override name if provided
        if project_name and requirements.name == "untitled_project":
            requirements.name = project_name

        logger.info(
            f"Parsed requirements: type={requirements.project_type.value}, "
            f"features={len(requirements.features)}"
        )

        return {
            "requirements": requirements.model_dump(),
            "status": ExecutionStatus.GENERATING_TASKS.value,
            "execution_logs": [
                create_execution_log(
                    "parse_requirements",
                    "completed",
                    details={
                        "project_type": requirements.project_type.value,
                        "features_count": len(requirements.features),
                        "tech_stack": requirements.tech_stack,
                    },
                )
            ],
        }

    except Exception as e:
        logger.error(f"Failed to parse requirements: {e}")
        return {
            "status": ExecutionStatus.FAILED.value,
            "error": f"Failed to parse requirements: {e}",
            "error_node": "parse_requirements",
            "execution_logs": [
                create_execution_log(
                    "parse_requirements",
                    "error",
                    error=str(e),
                )
            ],
        }


# =============================================================================
# NODE 2: GENERATE TASKS
# =============================================================================


async def generate_tasks_node(state: KappaState) -> dict[str, Any]:
    """
    Generate TaskSpecs from parsed ProjectRequirements.

    This node takes the structured requirements and generates atomic,
    executable tasks organized by project type.

    Args:
        state: Current Kappa state with requirements dict.

    Returns:
        State updates with generated tasks.

    State Changes:
        - tasks: List of TaskSpec dicts
        - status: GENERATING_TASKS -> RESOLVING_DEPENDENCIES
        - execution_logs: Appended with generation log
    """
    logger.info("Generating tasks from requirements")

    requirements_data = state.get("requirements")
    if not requirements_data:
        logger.error("No requirements data available")
        return {
            "status": ExecutionStatus.FAILED.value,
            "error": "No requirements data available for task generation",
            "error_node": "generate_tasks",
            "execution_logs": [
                create_execution_log(
                    "generate_tasks",
                    "error",
                    error="No requirements data available",
                )
            ],
        }

    try:
        from src.decomposition.models import ProjectRequirements
        from src.decomposition.task_generator import TaskGenerator

        # Reconstruct ProjectRequirements from dict
        requirements = ProjectRequirements(**requirements_data)

        # Generate tasks
        generator = TaskGenerator()
        tasks = await generator.generate(requirements)

        logger.info(f"Generated {len(tasks)} tasks")

        # Convert to dicts for state
        task_dicts = [task.model_dump() for task in tasks]

        return {
            "tasks": task_dicts,
            "status": ExecutionStatus.RESOLVING_DEPENDENCIES.value,
            "execution_logs": [
                create_execution_log(
                    "generate_tasks",
                    "completed",
                    details={
                        "tasks_count": len(tasks),
                        "categories": list(set(t.category.value for t in tasks)),
                    },
                )
            ],
        }

    except Exception as e:
        logger.error(f"Failed to generate tasks: {e}")
        return {
            "status": ExecutionStatus.FAILED.value,
            "error": f"Failed to generate tasks: {e}",
            "error_node": "generate_tasks",
            "execution_logs": [
                create_execution_log(
                    "generate_tasks",
                    "error",
                    error=str(e),
                )
            ],
        }


# =============================================================================
# NODE 3: RESOLVE DEPENDENCIES
# =============================================================================


async def resolve_dependencies_node(state: KappaState) -> dict[str, Any]:
    """
    Build dependency graph and assign execution waves.

    This node analyzes task dependencies, detects cycles, calculates
    execution waves, and identifies potential file conflicts.

    Args:
        state: Current Kappa state with tasks list.

    Returns:
        State updates with dependency graph and waves.

    State Changes:
        - dependency_graph: DependencyGraph data dict
        - current_wave: 0
        - total_waves: Number of waves
        - status: RESOLVING_DEPENDENCIES -> PLANNING_COMPLETE
        - execution_logs: Appended with resolution log
    """
    logger.info("Resolving task dependencies")

    tasks_data = state.get("tasks", [])
    if not tasks_data:
        logger.warning("No tasks to resolve dependencies for")
        return {
            "dependency_graph": {"waves": [], "total_waves": 0, "edges": {}},
            "current_wave": 0,
            "total_waves": 0,
            "status": ExecutionStatus.PLANNING_COMPLETE.value,
            "execution_logs": [
                create_execution_log(
                    "resolve_dependencies",
                    "completed",
                    details={"tasks_count": 0, "waves_count": 0},
                )
            ],
        }

    try:
        from src.decomposition.models import TaskSpec
        from src.decomposition.dependency_resolver import DependencyResolver

        # Reconstruct TaskSpec objects
        tasks = [TaskSpec(**t) for t in tasks_data]

        # Resolve dependencies
        resolver = DependencyResolver(tasks)
        graph = resolver.resolve()

        # Check for conflicts
        conflicts = resolver.get_conflicts()
        if conflicts:
            logger.warning(f"Detected {len(conflicts)} potential file conflicts")

        logger.info(f"Organized {len(tasks)} tasks into {graph.total_waves} waves")

        # Prepare graph data for state
        graph_data = {
            "waves": graph.waves,
            "total_waves": graph.total_waves,
            "edges": graph.edges,
            "conflicts": [c.model_dump() for c in conflicts],
        }

        return {
            "dependency_graph": graph_data,
            "current_wave": 0,
            "total_waves": graph.total_waves,
            "status": ExecutionStatus.PLANNING_COMPLETE.value,
            "execution_logs": [
                create_execution_log(
                    "resolve_dependencies",
                    "completed",
                    details={
                        "tasks_count": len(tasks),
                        "waves_count": graph.total_waves,
                        "conflicts_count": len(conflicts),
                    },
                )
            ],
        }

    except ValueError as e:
        # Cycle detected
        logger.error(f"Dependency resolution failed: {e}")
        return {
            "status": ExecutionStatus.FAILED.value,
            "error": str(e),
            "error_node": "resolve_dependencies",
            "execution_logs": [
                create_execution_log(
                    "resolve_dependencies",
                    "error",
                    error=str(e),
                )
            ],
        }
    except Exception as e:
        logger.error(f"Failed to resolve dependencies: {e}")
        return {
            "status": ExecutionStatus.FAILED.value,
            "error": f"Failed to resolve dependencies: {e}",
            "error_node": "resolve_dependencies",
            "execution_logs": [
                create_execution_log(
                    "resolve_dependencies",
                    "error",
                    error=str(e),
                )
            ],
        }


# =============================================================================
# NODE 4: EXECUTE WAVE
# =============================================================================


async def execute_wave_node(state: KappaState) -> dict[str, Any]:
    """
    Execute all tasks in the current wave in parallel.

    This node spawns Claude sessions to execute tasks that are ready
    (all dependencies satisfied) in the current wave.

    Args:
        state: Current Kappa state with tasks and wave info.

    Returns:
        State updates with execution results.

    State Changes:
        - completed_tasks: Appended with successful task IDs
        - failed_tasks: Appended with failed task IDs
        - current_wave: Incremented
        - created_files: Appended with new files
        - status: EXECUTING
        - execution_logs: Appended with execution log
    """
    current_wave = state.get("current_wave", 0)
    total_waves = state.get("total_waves", 0)

    logger.info(f"Executing wave {current_wave + 1}/{total_waves}")

    # Get tasks for current wave
    graph_data = state.get("dependency_graph", {})
    waves = graph_data.get("waves", [])

    if current_wave >= len(waves):
        logger.info("No more waves to execute")
        return {
            "status": ExecutionStatus.MERGING.value,
            "execution_logs": [
                create_execution_log(
                    "execute_wave",
                    "skipped",
                    details={"reason": "no_more_waves"},
                )
            ],
        }

    wave_task_ids = waves[current_wave]
    tasks_data = state.get("tasks", [])
    wave_tasks = [t for t in tasks_data if t["id"] in wave_task_ids]

    if not wave_tasks:
        logger.warning(f"No tasks found for wave {current_wave}")
        return {
            "current_wave": current_wave + 1,
            "status": ExecutionStatus.EXECUTING.value,
            "execution_logs": [
                create_execution_log(
                    "execute_wave",
                    "skipped",
                    details={"wave": current_wave, "reason": "no_tasks"},
                )
            ],
        }

    try:
        # Execute tasks in parallel
        completed: list[str] = []
        failed: list[str] = []
        created_files: list[str] = []
        task_results: list[dict[str, Any]] = []

        # Try to use ParallelExecutor if available, otherwise simulate
        try:
            from src.decomposition.executor import ParallelExecutor

            executor = ParallelExecutor()
            results = await executor.execute_wave(wave_task_ids, state)

            for result in results:
                task_results.append(result)
                if result.get("success"):
                    completed.append(result["task_id"])
                    created_files.extend(result.get("files_created", []))
                else:
                    failed.append(result["task_id"])

        except ImportError:
            # Executor not available, simulate success for planning
            logger.warning("ParallelExecutor not available, simulating execution")
            for task in wave_tasks:
                completed.append(task["id"])
                created_files.extend(task.get("files_to_create", []))
                task_results.append({
                    "task_id": task["id"],
                    "success": True,
                    "simulated": True,
                })

        logger.info(
            f"Wave {current_wave + 1} complete: "
            f"{len(completed)} succeeded, {len(failed)} failed"
        )

        return {
            "completed_tasks": completed,
            "failed_tasks": failed,
            "created_files": created_files,
            "current_wave": current_wave + 1,
            "task_results": state.get("task_results", []) + task_results,
            "status": ExecutionStatus.EXECUTING.value,
            "active_sessions": {},  # Clear active sessions
            "execution_logs": [
                create_execution_log(
                    "execute_wave",
                    "completed",
                    details={
                        "wave": current_wave,
                        "completed": len(completed),
                        "failed": len(failed),
                        "files_created": len(created_files),
                    },
                )
            ],
        }

    except Exception as e:
        logger.error(f"Wave execution failed: {e}")
        return {
            "failed_tasks": wave_task_ids,
            "current_wave": current_wave + 1,
            "status": ExecutionStatus.EXECUTING.value,
            "execution_logs": [
                create_execution_log(
                    "execute_wave",
                    "error",
                    details={"wave": current_wave},
                    error=str(e),
                )
            ],
        }


# =============================================================================
# NODE 5: MERGE OUTPUTS
# =============================================================================


async def merge_outputs_node(state: KappaState) -> dict[str, Any]:
    """
    Merge outputs from parallel execution and resolve conflicts.

    This node analyzes files created/modified by multiple tasks and
    applies conflict resolution strategies.

    Args:
        state: Current Kappa state with task results.

    Returns:
        State updates with merge results.

    State Changes:
        - conflicts: List of detected/resolved conflicts
        - status: MERGING -> VALIDATING
        - execution_logs: Appended with merge log
    """
    logger.info("Merging outputs and resolving conflicts")

    task_results = state.get("task_results", [])

    if not task_results:
        logger.info("No task results to merge")
        return {
            "conflicts": [],
            "status": ExecutionStatus.VALIDATING.value,
            "execution_logs": [
                create_execution_log(
                    "merge_outputs",
                    "skipped",
                    details={"reason": "no_results"},
                )
            ],
        }

    try:
        # Detect file conflicts
        file_writers: dict[str, list[str]] = {}
        for result in task_results:
            task_id = result.get("task_id", "")
            for file_path in result.get("files_created", []):
                if file_path not in file_writers:
                    file_writers[file_path] = []
                file_writers[file_path].append(task_id)
            for file_path in result.get("files_modified", []):
                if file_path not in file_writers:
                    file_writers[file_path] = []
                file_writers[file_path].append(task_id)

        # Find conflicts (files touched by multiple tasks)
        conflicts: list[dict[str, Any]] = []
        for file_path, writers in file_writers.items():
            if len(writers) > 1:
                conflicts.append({
                    "file_path": file_path,
                    "task_ids": writers,
                    "conflict_type": "multiple_writers",
                    "resolved": False,
                    "detected_at": datetime.utcnow().isoformat(),
                })

        if conflicts:
            logger.warning(f"Detected {len(conflicts)} file conflicts")

            # Attempt auto-resolution
            try:
                from src.conflict.resolver import ConflictResolver

                resolver = ConflictResolver()
                for conflict in conflicts:
                    try:
                        resolution = await resolver.resolve(conflict)
                        conflict["resolution"] = resolution
                        conflict["resolved"] = True
                        conflict["resolved_at"] = datetime.utcnow().isoformat()
                    except Exception as e:
                        logger.warning(f"Could not auto-resolve conflict: {e}")
                        conflict["resolution_error"] = str(e)

            except ImportError:
                logger.warning("ConflictResolver not available")

        else:
            logger.info("No conflicts detected")

        return {
            "conflicts": conflicts,
            "status": ExecutionStatus.VALIDATING.value,
            "execution_logs": [
                create_execution_log(
                    "merge_outputs",
                    "completed",
                    details={
                        "conflicts_detected": len(conflicts),
                        "conflicts_resolved": sum(1 for c in conflicts if c.get("resolved")),
                    },
                )
            ],
        }

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return {
            "conflicts": [],
            "status": ExecutionStatus.VALIDATING.value,
            "execution_logs": [
                create_execution_log(
                    "merge_outputs",
                    "error",
                    error=str(e),
                )
            ],
        }


# =============================================================================
# NODE 6: VALIDATE
# =============================================================================


async def validate_node(state: KappaState) -> dict[str, Any]:
    """
    Validate all outputs from execution.

    This node runs validation checks including type checking, building,
    tests, and linting on the generated project.

    Args:
        state: Current Kappa state with workspace path.

    Returns:
        State updates with validation results.

    State Changes:
        - validation_results: Dict with validation check results
        - status: VALIDATING -> COMPLETED or VALIDATION_FAILED
        - completed_at: ISO timestamp
        - execution_logs: Appended with validation log
    """
    logger.info("Validating outputs")

    workspace_path = state.get("workspace_path", "")

    if not workspace_path or not Path(workspace_path).exists():
        logger.warning("Workspace path not available for validation")
        return {
            "validation_results": {"skipped": True, "reason": "no_workspace"},
            "status": ExecutionStatus.COMPLETED.value,
            "completed_at": datetime.utcnow().isoformat(),
            "execution_logs": [
                create_execution_log(
                    "validate",
                    "skipped",
                    details={"reason": "no_workspace"},
                )
            ],
        }

    try:
        validation_results: dict[str, Any] = {
            "workspace": workspace_path,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }

        # Try to use ProjectValidator if available
        try:
            from src.decomposition.validator import ProjectValidator

            validator = ProjectValidator(workspace_path)
            results = await validator.validate()
            validation_results["checks"] = results
            validation_results["success"] = results.get("success", False)

        except ImportError:
            # Validator not available, do basic checks
            logger.warning("ProjectValidator not available, doing basic checks")

            workspace = Path(workspace_path)

            # Check if any files were created
            files_exist = list(workspace.rglob("*"))
            validation_results["checks"]["files_created"] = {
                "success": len(files_exist) > 0,
                "count": len(files_exist),
            }
            validation_results["success"] = len(files_exist) > 0

        # Determine final status
        if validation_results.get("success", True):
            final_status = ExecutionStatus.COMPLETED.value
            logger.info("Validation passed")
        else:
            final_status = ExecutionStatus.VALIDATION_FAILED.value
            logger.warning("Validation failed")

        return {
            "validation_results": validation_results,
            "status": final_status,
            "completed_at": datetime.utcnow().isoformat(),
            "execution_logs": [
                create_execution_log(
                    "validate",
                    "completed",
                    details={
                        "success": validation_results.get("success"),
                        "checks": list(validation_results.get("checks", {}).keys()),
                    },
                )
            ],
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            "validation_results": {"error": str(e), "success": False},
            "status": ExecutionStatus.VALIDATION_FAILED.value,
            "completed_at": datetime.utcnow().isoformat(),
            "execution_logs": [
                create_execution_log(
                    "validate",
                    "error",
                    error=str(e),
                )
            ],
        }


# =============================================================================
# NODE 7: HANDLE ERROR
# =============================================================================


async def handle_error_node(state: KappaState) -> dict[str, Any]:
    """
    Handle execution errors and prepare error report.

    This node captures error state, generates error reports, and
    prepares the final failed status.

    Args:
        state: Current Kappa state with error information.

    Returns:
        State updates with error handling results.

    State Changes:
        - status: FAILED
        - completed_at: ISO timestamp
        - execution_logs: Appended with error handling log
    """
    logger.info("Handling execution error")

    error = state.get("error", "Unknown error")
    error_node = state.get("error_node", "unknown")

    # Calculate execution summary
    completed_tasks = state.get("completed_tasks", [])
    failed_tasks = state.get("failed_tasks", [])
    total_tasks = len(state.get("tasks", []))

    # Generate error report
    error_report = {
        "error": error,
        "error_node": error_node,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_tasks": total_tasks,
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "current_wave": state.get("current_wave", 0),
            "total_waves": state.get("total_waves", 0),
        },
    }

    logger.error(
        f"Execution failed at {error_node}: {error}\n"
        f"Completed: {len(completed_tasks)}/{total_tasks} tasks"
    )

    return {
        "status": ExecutionStatus.FAILED.value,
        "completed_at": datetime.utcnow().isoformat(),
        "error": error,
        "execution_logs": [
            create_execution_log(
                "handle_error",
                "completed",
                details=error_report,
                error=error,
            )
        ],
    }


# =============================================================================
# LEGACY COMPATIBILITY NODES
# =============================================================================


async def initialize_node(state: KappaState) -> dict[str, Any]:
    """
    Initialize the Kappa execution pipeline (legacy compatibility).

    Sets up the project directory and prepares for decomposition.

    Args:
        state: Current Kappa state.

    Returns:
        State updates with initialization results.
    """
    logger.info(f"Initializing project: {state.get('project_name')}")

    workspace_path = state.get("workspace_path", state.get("project_path", ""))

    if workspace_path:
        path = Path(workspace_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "logs").mkdir(exist_ok=True)

    return {
        "status": ExecutionStatus.PARSING.value,
        "started_at": datetime.utcnow().isoformat(),
        "execution_logs": [
            create_execution_log(
                "initialize",
                "completed",
                details={"workspace": workspace_path},
            )
        ],
    }


async def decomposition_node(state: KappaState) -> dict[str, Any]:
    """
    Combined decomposition node (legacy compatibility).

    Combines parse_requirements, generate_tasks, and resolve_dependencies
    into a single node for backward compatibility.

    Args:
        state: Current Kappa state.

    Returns:
        State updates with decomposition results.
    """
    logger.info("Starting specification decomposition (legacy)")

    spec = state.get("specification", state.get("requirements_text", ""))

    try:
        from src.decomposition.task_generator import LegacyTaskGenerator
        from src.decomposition.dependency_resolver import DependencyResolver

        generator = LegacyTaskGenerator()
        tasks = await generator.generate(spec)

        logger.info(f"Generated {len(tasks)} tasks")

        resolver = DependencyResolver()
        dependency_graph = resolver.build_graph(tasks)
        waves = resolver.assign_waves(tasks, dependency_graph)

        logger.info(f"Organized into {len(waves)} execution waves")

        return {
            "tasks": [task.model_dump() for task in tasks],
            "dependency_graph": {"edges": dependency_graph, "waves": waves},
            "waves": waves,
            "total_waves": len(waves),
            "current_wave": 0,
            "status": ExecutionStatus.EXECUTING.value,
            "execution_logs": [
                create_execution_log(
                    "decomposition",
                    "completed",
                    details={"tasks": len(tasks), "waves": len(waves)},
                )
            ],
        }

    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        return {
            "status": ExecutionStatus.FAILED.value,
            "error": f"Decomposition failed: {e}",
            "error_node": "decomposition",
            "execution_logs": [
                create_execution_log(
                    "decomposition",
                    "error",
                    error=str(e),
                )
            ],
        }


async def wave_execution_node(state: KappaState) -> dict[str, Any]:
    """
    Legacy wave execution node.

    Delegates to execute_wave_node for actual execution.
    """
    return await execute_wave_node(state)


async def conflict_resolution_node(state: KappaState) -> dict[str, Any]:
    """
    Legacy conflict resolution node.

    Delegates to merge_outputs_node for actual conflict handling.
    """
    return await merge_outputs_node(state)


async def finalize_node(state: KappaState) -> dict[str, Any]:
    """
    Finalize the execution and generate output summary.

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
    started_at_str = state.get("started_at", datetime.utcnow().isoformat())
    started_at = datetime.fromisoformat(started_at_str)
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
        summary_lines.append(f"Failed tasks: {', '.join(failed_tasks[:5])}")

    conflicts = state.get("conflicts", [])
    if conflicts:
        unresolved = [c for c in conflicts if not c.get("resolved")]
        summary_lines.append(
            f"Conflicts: {len(conflicts)} total, {len(unresolved)} unresolved"
        )

    final_output = "\n".join(summary_lines)

    logger.info("Execution finalized")
    logger.info(final_output)

    # Determine final status
    if failed_tasks and len(failed_tasks) == total_tasks:
        final_status = ExecutionStatus.FAILED.value
    else:
        final_status = ExecutionStatus.COMPLETED.value

    return {
        "status": final_status,
        "final_output": final_output,
        "completed_at": datetime.utcnow().isoformat(),
        "total_duration_seconds": duration,
        "execution_logs": [
            create_execution_log(
                "finalize",
                "completed",
                details={
                    "success_rate": success_rate,
                    "duration": duration,
                },
            )
        ],
    }
