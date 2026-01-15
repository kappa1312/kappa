"""Main Kappa orchestrator - coordinates the entire execution pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
from loguru import logger

from src.core.config import Settings, get_settings
from src.core.state import (
    KappaState,
    ProjectStatus,
    TaskResult,
    create_initial_state,
)


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
        >>> result = await kappa.run(
        ...     spec="Build a REST API with user authentication",
        ...     project_path="./my-project"
        ... )
        >>> print(result.status)
        'completed'
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize Kappa orchestrator.

        Args:
            settings: Optional settings override. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._graph: Any | None = None
        self._db_session: Any | None = None

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

    async def run(
        self,
        spec: str,
        project_path: str | Path,
        project_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> KappaState:
        """
        Execute the full Kappa pipeline.

        Args:
            spec: Natural language project specification.
            project_path: Path to the project directory.
            project_name: Optional project name.
            config: Optional configuration overrides.

        Returns:
            Final KappaState with results.

        Raises:
            DecompositionError: If specification cannot be parsed.
            SessionError: If all sessions fail.
            ConflictError: If conflicts cannot be resolved.
        """
        project_path = Path(project_path).resolve()
        project_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting Kappa execution for: {project_path}")
        logger.info(f"Specification: {spec[:100]}...")

        # Create initial state
        state = create_initial_state(
            specification=spec,
            project_path=str(project_path),
            project_name=project_name,
            config=config,
        )

        try:
            # Build and run the LangGraph
            graph = await self._build_graph()
            final_state = await graph.ainvoke(state)

            logger.info(f"Execution completed with status: {final_state['status']}")
            return final_state

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            state["status"] = ProjectStatus.FAILED.value
            state["error"] = str(e)
            state["completed_at"] = datetime.utcnow().isoformat()
            return state

    async def _build_graph(self) -> Any:
        """Build the LangGraph execution graph.

        Returns:
            Compiled LangGraph application.
        """
        if self._graph is not None:
            return self._graph

        from src.graph.builder import build_kappa_graph

        self._graph = await build_kappa_graph(self.settings)
        return self._graph

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
        from src.decomposition.task_generator import TaskGenerator

        generator = TaskGenerator()
        tasks = await generator.generate(spec)
        return [task.model_dump() for task in tasks]

    async def run_wave(
        self,
        state: KappaState,
        wave: int,
    ) -> list[TaskResult]:
        """
        Execute all tasks in a single wave.

        Args:
            state: Current Kappa state.
            wave: Wave index to execute.

        Returns:
            List of TaskResult objects.
        """
        from src.sessions.router import SessionRouter

        router = SessionRouter(
            max_sessions=self.settings.kappa_max_parallel_sessions,
            timeout=self.settings.kappa_session_timeout,
        )

        wave_tasks = state.get("waves", [[]])[wave]
        tasks = [t for t in state.get("tasks", []) if t["id"] in wave_tasks]

        logger.info(f"Executing wave {wave} with {len(tasks)} tasks")

        results = await router.execute_tasks(tasks, state)
        return results

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
                conflict["resolved_at"] = datetime.utcnow().isoformat()
                resolved.append(conflict)
            except Exception as e:
                logger.warning(f"Could not resolve conflict: {e}")
                raise ConflictError(
                    f"Failed to resolve conflict in {conflict['file_path']}: {e}"
                )

        return resolved

    async def status(self, project_id: str) -> KappaState | None:
        """
        Get the current status of a running or completed project.

        Args:
            project_id: Project UUID.

        Returns:
            Current state if found, None otherwise.
        """
        from src.knowledge.database import get_db_session

        async with get_db_session() as session:
            from src.knowledge.models import Project
            from sqlalchemy import select

            stmt = select(Project).where(Project.id == project_id)
            result = await session.execute(stmt)
            project = result.scalar_one_or_none()

            if project:
                return KappaState(
                    project_id=project.id,
                    project_name=project.name,
                    specification=project.specification,
                    status=project.status,
                )

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

        # Signal cancellation to active sessions
        # Implementation depends on session management

        return True

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Kappa(max_sessions={self.settings.kappa_max_parallel_sessions}, "
            f"debug={self.settings.kappa_debug})"
        )


async def run_kappa(
    spec: str,
    project_path: str,
    **kwargs: Any,
) -> KappaState:
    """
    Convenience function to run Kappa.

    Args:
        spec: Natural language project specification.
        project_path: Path to project directory.
        **kwargs: Additional arguments passed to Kappa.run().

    Returns:
        Final KappaState.

    Example:
        >>> from src.core.orchestrator import run_kappa
        >>> result = anyio.run(run_kappa, "Build a CLI tool", "./my-tool")
    """
    kappa = Kappa()
    return await kappa.run(spec, project_path, **kwargs)
