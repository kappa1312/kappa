"""Main Kappa orchestrator - coordinates the entire execution pipeline.

This module provides the primary interface for running Kappa projects,
integrating the LangGraph orchestration engine with database persistence
and state management.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from src.core.config import Settings, get_settings
from src.decomposition.executor import (
    ParallelExecutor,
    create_executor,
    TaskExecutionResult,
    WaveExecutionResult,
)
from src.prompts.builder import PromptContext, create_prompt_context


# =============================================================================
# EXCEPTIONS
# =============================================================================


class KappaError(Exception):
    """Base exception for Kappa errors."""

    pass


class DecompositionError(KappaError):
    """Failed to decompose requirements."""

    pass


class SessionError(KappaError):
    """Claude session error."""

    pass


class ConflictError(KappaError):
    """Unresolvable code conflict."""

    pass


class ValidationError(KappaError):
    """Validation failed."""

    pass


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


class Kappa:
    """
    Main Kappa orchestrator class.

    Coordinates the entire pipeline from specification to production code:
    1. Parse and decompose requirements
    2. Build dependency graph
    3. Spawn parallel Claude sessions
    4. Execute tasks in waves
    5. Resolve conflicts
    6. Merge and validate results

    Example:
        >>> kappa = Kappa()
        >>> result = await kappa.execute(
        ...     "Build a REST API with user authentication",
        ...     project_name="my-api"
        ... )
        >>> print(result["status"])
        'completed'

        >>> # Or use the legacy interface
        >>> result = await kappa.run(
        ...     spec="Build a REST API",
        ...     project_path="./my-project"
        ... )
    """

    def __init__(
        self,
        settings: Settings | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        """Initialize Kappa orchestrator.

        Args:
            settings: Optional settings override. Uses default if not provided.
            workspace: Optional workspace directory. Uses KAPPA_WORKING_DIR if not provided.
        """
        self.settings = settings or get_settings()
        self._graph: Any | None = None
        self._persistent_graph: Any | None = None

        # Set workspace directory
        if workspace:
            self.workspace = Path(workspace).resolve()
        else:
            import os
            self.workspace = Path(
                os.getenv("KAPPA_WORKING_DIR", "./workspace")
            ).resolve()

        # Configure logging
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configure loguru based on settings."""
        logger.remove()  # Remove default handler

        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

        # Create logs directory if needed
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        logger.add(
            "logs/kappa_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level=self.settings.kappa_log_level,
            format=log_format,
        )

        if self.settings.kappa_debug:
            logger.add(
                lambda msg: print(msg, end=""),
                level="DEBUG",
                format=log_format,
                colorize=True,
            )
        else:
            logger.add(
                lambda msg: print(msg, end=""),
                level=self.settings.kappa_log_level,
                format=log_format,
                colorize=True,
            )

    # =========================================================================
    # PRIMARY INTERFACE (NEW)
    # =========================================================================

    async def execute(
        self,
        requirements: str,
        project_name: str | None = None,
        project_id: str | None = None,
        config: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """
        Execute a complete project build.

        This is the primary interface for running Kappa. It creates a project
        in the database, runs the LangGraph orchestration, and persists results.

        Args:
            requirements: Natural language project requirements.
            project_name: Optional project name.
            project_id: Optional project ID (generated if not provided).
            config: Optional configuration overrides.
            persist: Whether to persist to database (default True).

        Returns:
            Execution result dict with project info and status.

        Raises:
            DecompositionError: If requirements cannot be parsed.
            SessionError: If all sessions fail.
            ConflictError: If conflicts cannot be resolved.
            ValidationError: If validation fails.

        Example:
            >>> kappa = Kappa()
            >>> result = await kappa.execute(
            ...     "Build Express API with GET /users and POST /users",
            ...     project_name="user-api"
            ... )
            >>> print(result["status"])
            'completed'
        """
        from src.graph.state import ExecutionStatus, create_initial_state

        # Generate IDs
        project_id = project_id or str(uuid4())
        project_name = project_name or f"project-{project_id[:8]}"

        # Create workspace for project
        workspace_path = self.workspace / project_name
        workspace_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Kappa execution for project: {project_name}")
        logger.info(f"Workspace: {workspace_path}")
        logger.info(f"Requirements: {requirements[:100]}...")

        # Create project record in database
        project = None
        if persist:
            try:
                project = await self._create_project_record(
                    project_id=project_id,
                    project_name=project_name,
                    requirements=requirements,
                    workspace_path=str(workspace_path),
                )
            except Exception as e:
                logger.warning(f"Could not create project record: {e}")

        # Create initial state
        state = create_initial_state(
            requirements_text=requirements,
            workspace_path=str(workspace_path),
            project_name=project_name,
            project_id=project_id,
            config=config,
        )

        try:
            # Build and run the graph
            if persist:
                graph = await self._build_persistent_graph()
            else:
                graph = await self._build_graph()

            # Run graph with config for checkpointing
            run_config = {"configurable": {"thread_id": project_id}}
            final_state = await graph.ainvoke(state, config=run_config)

            # Save final state
            if persist and project:
                await self._save_project_result(project_id, final_state)

            logger.info(f"Execution completed with status: {final_state.get('status')}")

            return self._format_result(
                project_id=project_id,
                project_name=project_name,
                workspace_path=str(workspace_path),
                state=final_state,
            )

        except Exception as e:
            logger.error(f"Execution failed: {e}")

            # Update project record with failure
            if persist:
                await self._mark_project_failed(project_id, str(e))

            return {
                "project_id": project_id,
                "project_name": project_name,
                "workspace_path": str(workspace_path),
                "status": ExecutionStatus.FAILED.value,
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat(),
            }

    async def preview(
        self,
        requirements: str,
    ) -> dict[str, Any]:
        """
        Preview the decomposition without executing.

        Useful for reviewing the task breakdown before running.

        Args:
            requirements: Natural language project requirements.

        Returns:
            Dict with parsed requirements, tasks, and dependency graph.

        Example:
            >>> kappa = Kappa()
            >>> preview = await kappa.preview("Build a REST API")
            >>> print(f"Found {len(preview['tasks'])} tasks")
        """
        from src.decomposition.models import ProjectRequirements
        from src.decomposition.parser import RequirementsParser
        from src.decomposition.task_generator import TaskGenerator
        from src.decomposition.dependency_resolver import DependencyResolver

        logger.info("Generating preview for requirements")

        # Parse requirements
        parser = RequirementsParser()
        parsed_requirements = await parser.parse(requirements)

        # Generate tasks
        generator = TaskGenerator()
        tasks = await generator.generate(parsed_requirements)

        # Build dependency graph
        resolver = DependencyResolver(tasks)
        graph = resolver.resolve()
        conflicts = resolver.detect_conflicts()

        return {
            "requirements": parsed_requirements.model_dump(),
            "tasks": [t.model_dump() for t in tasks],
            "dependency_graph": {
                "waves": graph.waves,
                "total_waves": len(graph.waves),
                "edges": graph.edges,
            },
            "potential_conflicts": [c.model_dump() for c in conflicts],
        }

    async def resume(
        self,
        project_id: str,
    ) -> dict[str, Any]:
        """
        Resume a paused or failed project execution.

        Args:
            project_id: ID of the project to resume.

        Returns:
            Execution result dict.

        Raises:
            KappaError: If project not found or cannot be resumed.
        """
        from src.graph.state import reconstruct_state_from_db

        logger.info(f"Resuming project: {project_id}")

        # Reconstruct state from database
        state = await reconstruct_state_from_db(project_id)
        if not state:
            raise KappaError(f"Project not found: {project_id}")

        # Check if resumable
        status = state.get("status", "")
        if status in ("completed", "failed"):
            logger.warning(f"Project already in terminal state: {status}")

        # Build persistent graph and resume
        graph = await self._build_persistent_graph()
        run_config = {"configurable": {"thread_id": project_id}}

        final_state = await graph.ainvoke(state, config=run_config)

        # Save result
        await self._save_project_result(project_id, final_state)

        return self._format_result(
            project_id=project_id,
            project_name=state.get("project_name", ""),
            workspace_path=state.get("workspace_path", ""),
            state=final_state,
        )

    # =========================================================================
    # PARALLEL EXECUTION (NEW)
    # =========================================================================

    async def execute_parallel(
        self,
        requirements: str,
        project_name: str | None = None,
        max_concurrent: int | None = None,
        timeout: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a project using direct parallel execution.

        This method bypasses LangGraph and executes tasks directly using
        the ParallelExecutor for maximum performance.

        Args:
            requirements: Natural language project requirements.
            project_name: Optional project name.
            max_concurrent: Maximum concurrent sessions (default from settings).
            timeout: Timeout per task in seconds (default from settings).
            dry_run: If True, simulate execution without running Claude.

        Returns:
            Execution result dict.

        Example:
            >>> kappa = Kappa()
            >>> result = await kappa.execute_parallel(
            ...     "Build Express API with GET /users and POST /users",
            ...     max_concurrent=10
            ... )
        """
        from src.decomposition.parser import RequirementsParser
        from src.decomposition.task_generator import TaskGenerator
        from src.decomposition.dependency_resolver import DependencyResolver
        from src.knowledge.context_manager import SharedContext

        project_id = str(uuid4())
        project_name = project_name or f"project-{project_id[:8]}"
        workspace_path = self.workspace / project_name
        workspace_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting parallel execution for project: {project_name}")
        logger.info(f"Workspace: {workspace_path}")

        # Parse requirements
        parser = RequirementsParser()
        parsed_requirements = await parser.parse(requirements)

        # Generate tasks
        generator = TaskGenerator()
        tasks = await generator.generate(parsed_requirements)

        # Build dependency graph
        resolver = DependencyResolver(tasks)
        graph = resolver.resolve()

        logger.info(f"Generated {len(tasks)} tasks in {graph.total_waves} waves")

        # Create executor
        max_concurrent = max_concurrent or self.settings.kappa_max_parallel_sessions
        timeout = timeout or self.settings.kappa_session_timeout

        if dry_run:
            executor = create_executor("dry_run")
        else:
            executor = create_executor(
                "parallel",
                max_concurrent=max_concurrent,
                timeout=timeout,
                workspace=str(workspace_path),
            )

        # Create shared context
        context = create_prompt_context(
            project_name=project_name,
            workspace=str(workspace_path),
            requirements=parsed_requirements,
        )

        # Build state for executor
        state = {
            "project_id": project_id,
            "project_name": project_name,
            "workspace_path": str(workspace_path),
            "tasks": [t.model_dump() for t in tasks],
            "dependency_graph": {
                "waves": graph.waves,
                "total_waves": graph.total_waves,
            },
            "global_context": {
                "tech_stack": parsed_requirements.tech_stack,
                "project_type": parsed_requirements.project_type.value,
            },
            "completed_tasks": [],
            "failed_tasks": [],
        }

        # Execute graph
        started_at = datetime.utcnow()
        wave_results = await executor.execute_graph(graph, state, context)

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        # Aggregate results
        total_completed = sum(len(w.completed_tasks) for w in wave_results)
        total_failed = sum(len(w.failed_tasks) for w in wave_results)
        all_files_created = []
        all_files_modified = []

        for wave in wave_results:
            all_files_created.extend(wave.all_files_created)
            all_files_modified.extend(wave.all_files_modified)

        success = total_failed == 0 or total_completed > total_failed

        logger.info(
            f"Parallel execution complete: {total_completed} tasks succeeded, "
            f"{total_failed} failed in {duration:.1f}s"
        )

        return {
            "project_id": project_id,
            "project_name": project_name,
            "workspace_path": str(workspace_path),
            "status": "completed" if success else "failed",
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": duration,
            "total_tasks": len(tasks),
            "completed_tasks": total_completed,
            "failed_tasks": total_failed,
            "waves_executed": len(wave_results),
            "files_created": list(set(all_files_created)),
            "files_modified": list(set(all_files_modified)),
            "wave_results": [w.to_dict() for w in wave_results],
        }

    async def execute_wave_parallel(
        self,
        tasks: list[dict[str, Any]],
        state: dict[str, Any],
        wave_number: int = 0,
        max_concurrent: int | None = None,
    ) -> WaveExecutionResult:
        """
        Execute a single wave of tasks in parallel.

        Args:
            tasks: List of task dictionaries to execute.
            state: Current execution state.
            wave_number: Wave number for tracking.
            max_concurrent: Maximum concurrent sessions.

        Returns:
            WaveExecutionResult with all task results.

        Example:
            >>> result = await kappa.execute_wave_parallel(
            ...     tasks=[{"id": "task-1", ...}, {"id": "task-2", ...}],
            ...     state=current_state,
            ...     wave_number=0
            ... )
        """
        max_concurrent = max_concurrent or self.settings.kappa_max_parallel_sessions

        executor = create_executor(
            "parallel",
            max_concurrent=max_concurrent,
            timeout=self.settings.kappa_session_timeout,
            workspace=str(self.workspace),
        )

        task_ids = [t.get("id", str(uuid4())) for t in tasks]

        # Ensure tasks are in state
        state.setdefault("tasks", []).extend(tasks)

        return await executor.execute_wave(
            task_ids=task_ids,
            state=state,
            wave_number=wave_number,
        )

    # =========================================================================
    # LEGACY INTERFACE (BACKWARD COMPATIBILITY)
    # =========================================================================

    async def run(
        self,
        spec: str,
        project_path: str | Path,
        project_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full Kappa pipeline (legacy interface).

        This method is maintained for backward compatibility.
        Consider using execute() for new code.

        Args:
            spec: Natural language project specification.
            project_path: Path to the project directory.
            project_name: Optional project name.
            config: Optional configuration overrides.

        Returns:
            Final state dict with results.

        Raises:
            DecompositionError: If specification cannot be parsed.
            SessionError: If all sessions fail.
            ConflictError: If conflicts cannot be resolved.
        """
        from src.graph.state import ExecutionStatus, create_initial_state

        project_path = Path(project_path).resolve()
        project_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Kappa execution for: {project_path}")
        logger.info(f"Specification: {spec[:100]}...")

        # Create initial state using new state module
        state = create_initial_state(
            requirements_text=spec,
            workspace_path=str(project_path),
            project_name=project_name,
            config=config,
        )

        try:
            # Build and run the LangGraph
            graph = await self._build_graph()
            final_state = await graph.ainvoke(state)

            logger.info(f"Execution completed with status: {final_state.get('status')}")
            return dict(final_state)

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            state["status"] = ExecutionStatus.FAILED.value
            state["error"] = str(e)
            state["completed_at"] = datetime.utcnow().isoformat()
            return dict(state)

    async def decompose(self, spec: str) -> list[dict[str, Any]]:
        """
        Decompose a specification into tasks without execution.

        Useful for previewing the task breakdown before running.

        Args:
            spec: Natural language project specification.

        Returns:
            List of task dictionaries.

        Example:
            >>> kappa = Kappa()
            >>> tasks = await kappa.decompose("Build a calculator CLI")
            >>> len(tasks)
            5
        """
        preview = await self.preview(spec)
        return preview.get("tasks", [])

    async def run_wave(
        self,
        state: dict[str, Any],
        wave: int,
    ) -> list[dict[str, Any]]:
        """
        Execute all tasks in a single wave.

        Args:
            state: Current Kappa state.
            wave: Wave index to execute.

        Returns:
            List of TaskResult dictionaries.
        """
        from src.sessions.router import SessionRouter

        router = SessionRouter(
            max_sessions=self.settings.kappa_max_parallel_sessions,
            timeout=self.settings.kappa_session_timeout,
        )

        # Get wave tasks from dependency graph
        graph_data = state.get("dependency_graph", {})
        waves = graph_data.get("waves", [])

        if wave >= len(waves):
            logger.warning(f"Wave {wave} not found")
            return []

        wave_task_ids = waves[wave]
        tasks = [t for t in state.get("tasks", []) if t.get("id") in wave_task_ids]

        logger.info(f"Executing wave {wave} with {len(tasks)} tasks")

        results = await router.execute_tasks(tasks, state)
        return [r.model_dump() if hasattr(r, "model_dump") else r for r in results]

    async def resolve_conflicts(
        self,
        conflicts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Attempt to resolve detected conflicts.

        Args:
            conflicts: List of conflict dictionaries.

        Returns:
            List of resolved conflict dictionaries.

        Raises:
            ConflictError: If conflicts cannot be resolved automatically.
        """
        from src.conflict.resolver import ConflictResolver

        resolver = ConflictResolver()
        resolved = []

        for conflict in conflicts:
            try:
                resolution = await resolver.resolve(conflict)
                conflict["resolution"] = resolution
                conflict["resolved"] = True
                conflict["resolved_at"] = datetime.utcnow().isoformat()
                resolved.append(conflict)
            except Exception as e:
                logger.warning(f"Could not resolve conflict: {e}")
                raise ConflictError(
                    f"Failed to resolve conflict in {conflict.get('file_path', 'unknown')}: {e}"
                )

        return resolved

    # =========================================================================
    # STATUS AND CONTROL
    # =========================================================================

    async def status(self, project_id: str) -> dict[str, Any] | None:
        """
        Get the current status of a running or completed project.

        Args:
            project_id: Project UUID.

        Returns:
            Current state if found, None otherwise.
        """
        from src.knowledge.database import get_db_session

        try:
            async with get_db_session() as session:
                from src.knowledge.models import Project
                from sqlalchemy import select

                stmt = select(Project).where(Project.id == project_id)
                result = await session.execute(stmt)
                project = result.scalar_one_or_none()

                if project:
                    return {
                        "project_id": str(project.id),
                        "project_name": project.name,
                        "specification": project.specification,
                        "status": project.status,
                        "workspace_path": project.workspace_path,
                        "created_at": project.created_at.isoformat() if project.created_at else None,
                        "completed_at": project.completed_at.isoformat() if project.completed_at else None,
                    }
        except Exception as e:
            logger.warning(f"Could not get status: {e}")

        return None

    async def cancel(self, project_id: str) -> bool:
        """
        Cancel a running project execution.

        Args:
            project_id: Project UUID to cancel.

        Returns:
            True if cancelled successfully.
        """
        logger.info(f"Cancelling project: {project_id}")

        try:
            from src.knowledge.database import get_db_session
            from src.graph.state import ExecutionStatus

            async with get_db_session() as session:
                from src.knowledge.models import Project
                from sqlalchemy import update

                await session.execute(
                    update(Project)
                    .where(Project.id == project_id)
                    .values(
                        status=ExecutionStatus.FAILED.value,
                        completed_at=datetime.utcnow(),
                    )
                )
                await session.commit()

            return True

        except Exception as e:
            logger.error(f"Failed to cancel project: {e}")
            return False

    async def list_projects(
        self,
        limit: int = 20,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List recent projects.

        Args:
            limit: Maximum number of projects to return.
            status_filter: Optional status to filter by.

        Returns:
            List of project summaries.
        """
        from src.knowledge.database import get_db_session

        try:
            async with get_db_session() as session:
                from src.knowledge.models import Project
                from sqlalchemy import select

                stmt = select(Project).order_by(Project.created_at.desc()).limit(limit)

                if status_filter:
                    stmt = stmt.where(Project.status == status_filter)

                result = await session.execute(stmt)
                projects = result.scalars().all()

                return [
                    {
                        "project_id": str(p.id),
                        "project_name": p.name,
                        "status": p.status,
                        "created_at": p.created_at.isoformat() if p.created_at else None,
                    }
                    for p in projects
                ]

        except Exception as e:
            logger.warning(f"Could not list projects: {e}")
            return []

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    async def _build_graph(self) -> Any:
        """Build the LangGraph execution graph.

        Returns:
            Compiled LangGraph application.
        """
        if self._graph is not None:
            return self._graph

        from src.graph.builder import create_orchestration_graph

        self._graph = create_orchestration_graph()
        return self._graph

    async def _build_persistent_graph(self) -> Any:
        """Build the LangGraph with PostgreSQL persistence.

        Returns:
            Compiled LangGraph application with checkpointer.
        """
        if self._persistent_graph is not None:
            return self._persistent_graph

        from src.graph.builder import build_kappa_graph_with_persistence

        self._persistent_graph = await build_kappa_graph_with_persistence(self.settings)
        return self._persistent_graph

    async def _create_project_record(
        self,
        project_id: str,
        project_name: str,
        requirements: str,
        workspace_path: str,
    ) -> Any:
        """Create a project record in the database.

        Args:
            project_id: Unique project ID.
            project_name: Human-readable project name.
            requirements: Natural language requirements.
            workspace_path: Path to project workspace.

        Returns:
            Created Project model instance.
        """
        from src.knowledge.database import get_db_session
        from src.graph.state import ExecutionStatus

        async with get_db_session() as session:
            from src.knowledge.models import Project

            project = Project(
                id=project_id,
                name=project_name,
                specification=requirements,
                workspace_path=workspace_path,
                status=ExecutionStatus.PENDING.value,
                created_at=datetime.utcnow(),
            )
            session.add(project)
            await session.commit()
            await session.refresh(project)

            logger.info(f"Created project record: {project_id}")
            return project

    async def _save_project_result(
        self,
        project_id: str,
        state: dict[str, Any],
    ) -> None:
        """Save final state to database.

        Args:
            project_id: Project ID.
            state: Final state dict.
        """
        from src.knowledge.database import get_db_session

        try:
            async with get_db_session() as session:
                from src.knowledge.models import Project
                from sqlalchemy import update

                completed_at = None
                if state.get("completed_at"):
                    completed_at = datetime.fromisoformat(state["completed_at"])

                await session.execute(
                    update(Project)
                    .where(Project.id == project_id)
                    .values(
                        status=state.get("status"),
                        completed_at=completed_at,
                    )
                )
                await session.commit()

                logger.info(f"Saved project result: {project_id}")

        except Exception as e:
            logger.warning(f"Could not save project result: {e}")

    async def _mark_project_failed(
        self,
        project_id: str,
        error: str,
    ) -> None:
        """Mark a project as failed in the database.

        Args:
            project_id: Project ID.
            error: Error message.
        """
        from src.knowledge.database import get_db_session
        from src.graph.state import ExecutionStatus

        try:
            async with get_db_session() as session:
                from src.knowledge.models import Project
                from sqlalchemy import update

                await session.execute(
                    update(Project)
                    .where(Project.id == project_id)
                    .values(
                        status=ExecutionStatus.FAILED.value,
                        completed_at=datetime.utcnow(),
                    )
                )
                await session.commit()

        except Exception as e:
            logger.warning(f"Could not mark project failed: {e}")

    def _format_result(
        self,
        project_id: str,
        project_name: str,
        workspace_path: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Format execution result for return.

        Args:
            project_id: Project ID.
            project_name: Project name.
            workspace_path: Workspace path.
            state: Final state dict.

        Returns:
            Formatted result dict.
        """
        from src.graph.state import calculate_progress

        progress = calculate_progress(state)

        return {
            "project_id": project_id,
            "project_name": project_name,
            "workspace_path": workspace_path,
            "status": state.get("status"),
            "error": state.get("error"),
            "started_at": state.get("started_at"),
            "completed_at": state.get("completed_at"),
            "progress": progress,
            "completed_tasks": state.get("completed_tasks", []),
            "failed_tasks": state.get("failed_tasks", []),
            "created_files": state.get("created_files", []),
            "conflicts": state.get("conflicts", []),
            "validation_results": state.get("validation_results"),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Kappa(workspace={self.workspace}, "
            f"max_sessions={self.settings.kappa_max_parallel_sessions}, "
            f"debug={self.settings.kappa_debug})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def run_kappa(
    spec: str,
    project_path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Convenience function to run Kappa (legacy interface).

    Args:
        spec: Natural language project specification.
        project_path: Path to project directory.
        **kwargs: Additional arguments passed to Kappa.run().

    Returns:
        Final state dict.

    Example:
        >>> import anyio
        >>> result = anyio.run(run_kappa, "Build a CLI tool", "./my-tool")
    """
    kappa = Kappa()
    return await kappa.run(spec, project_path, **kwargs)


async def execute_project(
    requirements: str,
    project_name: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Convenience function to execute a Kappa project.

    Args:
        requirements: Natural language project requirements.
        project_name: Optional project name.
        **kwargs: Additional arguments passed to Kappa.execute().

    Returns:
        Execution result dict.

    Example:
        >>> import anyio
        >>> result = anyio.run(
        ...     execute_project,
        ...     "Build Express API with user auth",
        ...     "user-api"
        ... )
    """
    kappa = Kappa()
    return await kappa.execute(requirements, project_name, **kwargs)


async def preview_project(requirements: str) -> dict[str, Any]:
    """
    Preview project decomposition without executing.

    Args:
        requirements: Natural language project requirements.

    Returns:
        Preview dict with tasks and dependency graph.

    Example:
        >>> import anyio
        >>> preview = anyio.run(preview_project, "Build a REST API")
        >>> print(f"Found {len(preview['tasks'])} tasks")
    """
    kappa = Kappa()
    return await kappa.preview(requirements)
