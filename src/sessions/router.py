"""Session router - manages parallel Claude session execution."""

from typing import Any

import anyio
from loguru import logger

from src.core.state import KappaState, TaskResult
from src.sessions.base import SessionConfig
from src.sessions.terminal import TerminalSession


class SessionRouter:
    """
    Route and manage parallel Claude sessions.

    Handles spawning multiple sessions, load balancing,
    and collecting results from parallel execution.

    Example:
        >>> router = SessionRouter(max_sessions=5)
        >>> results = await router.execute_tasks(tasks, state)
        >>> len(results)
        5
    """

    def __init__(
        self,
        max_sessions: int = 5,
        timeout: int = 3600,
        session_config: SessionConfig | None = None,
    ) -> None:
        """Initialize the session router.

        Args:
            max_sessions: Maximum concurrent sessions.
            timeout: Session timeout in seconds.
            session_config: Default session configuration.
        """
        self.max_sessions = max_sessions
        self.timeout = timeout
        self.session_config = session_config or SessionConfig(
            timeout_seconds=timeout,
        )
        self._active_sessions: dict[str, TerminalSession] = {}
        self._semaphore = anyio.Semaphore(max_sessions)

    async def execute_tasks(
        self,
        tasks: list[dict[str, Any]],
        state: KappaState,
    ) -> list[TaskResult]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of task dictionaries to execute.
            state: Current Kappa state for context.

        Returns:
            List of TaskResult objects.

        Example:
            >>> router = SessionRouter(max_sessions=3)
            >>> tasks = [{"id": "1", "name": "Task 1", ...}, ...]
            >>> results = await router.execute_tasks(tasks, state)
        """
        logger.info(f"Executing {len(tasks)} tasks with max {self.max_sessions} sessions")

        results: list[TaskResult] = []

        async def execute_one(task: dict[str, Any]) -> None:
            async with self._semaphore:
                result = await self._execute_task(task, state)
                results.append(result)

        async with anyio.create_task_group() as tg:
            for task in tasks:
                tg.start_soon(execute_one, task)

        logger.info(f"Completed {len(results)} tasks")
        return results

    async def _execute_task(
        self,
        task: dict[str, Any],
        state: KappaState,
    ) -> TaskResult:
        """Execute a single task in a session.

        Args:
            task: Task dictionary.
            state: Current Kappa state.

        Returns:
            TaskResult from execution.
        """
        task_id = task.get("id", "unknown")
        task_name = task.get("name", "Unnamed task")

        logger.info(f"Starting task: {task_name} ({task_id})")

        # Create session configuration
        config = SessionConfig(
            timeout_seconds=self.timeout,
            working_directory=state.get("project_path", "."),
            system_prompt=self._build_system_prompt(task, state),
        )

        # Create and run session
        session = TerminalSession(
            config=config,
            task_id=task_id,
        )

        try:
            await session.start()
            self._active_sessions[session.id] = session

            # Build task prompt
            prompt = self._build_task_prompt(task, state)

            # Execute
            result = await session.execute(prompt)

            logger.info(
                f"Task {task_name} completed: "
                f"{'success' if result.success else 'failed'}"
            )

            return result

        except Exception as e:
            logger.error(f"Task {task_name} failed with exception: {e}")
            return TaskResult(
                task_id=task_id,
                session_id=session.id,
                success=False,
                error=str(e),
            )

        finally:
            await session.close()
            self._active_sessions.pop(session.id, None)

    def _build_system_prompt(
        self,
        task: dict[str, Any],
        state: KappaState,
    ) -> str:
        """Build system prompt for the session.

        Args:
            task: Task dictionary.
            state: Current Kappa state.

        Returns:
            System prompt string.
        """
        project_name = state.get("project_name", "Unknown Project")

        return f"""You are working on the {project_name} project as part of an autonomous development team.

Your current task is: {task.get('name', 'Unknown')}

Guidelines:
- Follow the project's code style and conventions
- Write clean, well-documented code
- Include appropriate error handling
- Add type hints for all functions
- Only modify files directly related to your task
- Do not modify other files being worked on by parallel tasks

Focus on completing your specific task efficiently and correctly."""

    def _build_task_prompt(
        self,
        task: dict[str, Any],
        state: KappaState,
    ) -> str:
        """Build the task execution prompt.

        Args:
            task: Task dictionary.
            state: Current Kappa state.

        Returns:
            Task prompt string.
        """
        lines = [
            f"# Task: {task.get('name', 'Unknown')}",
            "",
            "## Description",
            task.get("description", "No description provided"),
            "",
        ]

        # Add target files if specified
        file_targets = task.get("file_targets", [])
        if file_targets:
            lines.extend([
                "## Target Files",
                *[f"- {f}" for f in file_targets],
                "",
            ])

        # Add dependencies context
        dependencies = task.get("dependencies", [])
        if dependencies:
            lines.extend([
                "## Dependencies",
                f"This task depends on: {', '.join(dependencies)}",
                "",
            ])

        # Add execution instructions
        lines.extend([
            "## Instructions",
            "1. Read any existing relevant files first",
            "2. Implement the required functionality",
            "3. Ensure code follows project conventions",
            "4. Test your changes if possible",
            "",
            "Begin implementation now.",
        ])

        return "\n".join(lines)

    async def cancel_all(self) -> None:
        """Cancel all active sessions."""
        logger.info(f"Cancelling {len(self._active_sessions)} active sessions")

        for session in list(self._active_sessions.values()):
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Error closing session {session.id}: {e}")

        self._active_sessions.clear()

    @property
    def active_count(self) -> int:
        """Get count of active sessions."""
        return len(self._active_sessions)

    def get_session(self, session_id: str) -> TerminalSession | None:
        """Get an active session by ID.

        Args:
            session_id: Session UUID.

        Returns:
            Session if found, None otherwise.
        """
        return self._active_sessions.get(session_id)
