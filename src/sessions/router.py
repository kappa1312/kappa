"""
Session Router for Kappa OS.

Determines the optimal session type for each task and manages
parallel Claude session execution with load balancing.
"""

from enum import Enum
from typing import Any

import anyio
from loguru import logger

from src.core.state import KappaState, TaskResult
from src.decomposition.models import SessionType, TaskSpec
from src.sessions.base import SessionConfig
from src.sessions.terminal import TerminalSession, TerminalSessionManager

# =============================================================================
# ROUTING STRATEGY
# =============================================================================


class RoutingStrategy(str, Enum):
    """Routing strategies for session selection."""

    AUTO = "auto"  # Automatically determine best session type
    TERMINAL_ONLY = "terminal_only"  # Force all tasks to terminal
    ROUND_ROBIN = "round_robin"  # Distribute across session types
    LOAD_BALANCED = "load_balanced"  # Route based on current load
    PRIORITY_BASED = "priority_based"  # Route high priority to faster sessions


# =============================================================================
# TASK ROUTER
# =============================================================================


class TaskRouter:
    """
    Routes tasks to appropriate session types.

    Currently supports:
    - TERMINAL: Claude Code CLI (default, 80% of tasks)
    - WEB: Browser automation (future)
    - NATIVE: Native app automation (future)
    - EXTENSION: Chrome extension (future)

    Example:
        >>> router = TaskRouter()
        >>> session_type = router.route(task)
        >>> print(session_type)  # SessionType.TERMINAL
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.AUTO):
        """
        Initialize task router.

        Args:
            strategy: Routing strategy to use.
        """
        self.strategy = strategy
        self._session_counts: dict[SessionType, int] = {
            SessionType.TERMINAL: 0,
            SessionType.WEB: 0,
            SessionType.NATIVE: 0,
            SessionType.EXTENSION: 0,
        }
        self._routing_stats: dict[str, int] = {}

    def route(self, task: TaskSpec | dict[str, Any]) -> SessionType:
        """
        Determine the best session type for a task.

        Args:
            task: TaskSpec or task dict to route.

        Returns:
            Recommended SessionType.
        """
        # Handle dict input
        if isinstance(task, dict):
            task = TaskSpec(**task)

        # If task already specifies session type, use it
        if task.session_type:
            session_type = SessionType(task.session_type)
            self._record_routing(task.id, session_type)
            return session_type

        # Strategy-based routing
        if self.strategy == RoutingStrategy.TERMINAL_ONLY:
            return SessionType.TERMINAL

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_route()

        if self.strategy == RoutingStrategy.LOAD_BALANCED:
            return self._load_balanced_route()

        # Auto routing based on task analysis
        session_type = self._auto_route(task)
        self._record_routing(task.id, session_type)
        return session_type

    def _auto_route(self, task: TaskSpec) -> SessionType:
        """Automatically determine session type based on task content."""

        description_lower = task.description.lower()
        title_lower = task.title.lower()
        combined = f"{title_lower} {description_lower}"

        # Check for browser/web automation needs
        web_keywords = [
            "screenshot",
            "browser",
            "selenium",
            "playwright",
            "e2e test",
            "end-to-end",
            "visual test",
            "cypress",
            "puppeteer",
            "web scraping",
            "crawl",
            "browser automation",
        ]
        if any(kw in combined for kw in web_keywords):
            logger.debug(f"Task {task.id} routed to WEB (keyword match)")
            return SessionType.WEB

        # Check for native app automation needs
        native_keywords = [
            "xcode",
            "android studio",
            "figma",
            "sketch",
            "photoshop",
            "illustrator",
            "desktop app",
            "native app",
            "electron",
            "tauri",
            "gui automation",
        ]
        if any(kw in combined for kw in native_keywords):
            logger.debug(f"Task {task.id} routed to NATIVE (keyword match)")
            return SessionType.NATIVE

        # Check for Chrome extension needs
        extension_keywords = [
            "chrome extension",
            "browser extension",
            "manifest.json",
            "content script",
            "background script",
            "firefox addon",
            "safari extension",
        ]
        if any(kw in combined for kw in extension_keywords):
            logger.debug(f"Task {task.id} routed to EXTENSION (keyword match)")
            return SessionType.EXTENSION

        # Default to terminal (handles 80%+ of tasks)
        return SessionType.TERMINAL

    def _round_robin_route(self) -> SessionType:
        """Route using round-robin across available session types."""
        # For now, just return terminal since others aren't implemented
        return SessionType.TERMINAL

    def _load_balanced_route(self) -> SessionType:
        """Route based on current session load."""
        # Find session type with lowest count
        min_count = min(self._session_counts.values())
        for session_type, count in self._session_counts.items():
            if count == min_count:
                return session_type
        return SessionType.TERMINAL

    def _record_routing(self, task_id: str, session_type: SessionType) -> None:
        """Record routing decision for stats."""
        self._session_counts[session_type] += 1
        self._routing_stats[task_id] = session_type.value

    def route_batch(self, tasks: list[TaskSpec | dict[str, Any]]) -> dict[str, SessionType]:
        """Route multiple tasks at once."""
        return {
            (t.id if isinstance(t, TaskSpec) else t.get("id", "unknown")): self.route(t)
            for t in tasks
        }

    def get_routing_stats(self) -> dict[str, Any]:
        """Get statistics on routing decisions."""
        return {
            "session_counts": {k.value: v for k, v in self._session_counts.items()},
            "total_routed": sum(self._session_counts.values()),
            "strategy": self.strategy.value,
        }

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        for key in self._session_counts:
            self._session_counts[key] = 0
        self._routing_stats.clear()


# =============================================================================
# SESSION ROUTER (LEGACY + ENHANCED)
# =============================================================================


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
        use_manager: bool = True,
    ) -> None:
        """
        Initialize the session router.

        Args:
            max_sessions: Maximum concurrent sessions.
            timeout: Session timeout in seconds.
            session_config: Default session configuration.
            use_manager: Use TerminalSessionManager (new) vs TerminalSession (legacy).
        """
        self.max_sessions = max_sessions
        self.timeout = timeout
        self.session_config = session_config or SessionConfig(
            timeout_seconds=timeout,
        )
        self.use_manager = use_manager
        self._active_sessions: dict[str, TerminalSession] = {}
        self._semaphore = anyio.Semaphore(max_sessions)
        self._task_router = TaskRouter()

        # Initialize session manager if using new approach
        if use_manager:
            self._manager = TerminalSessionManager(
                max_concurrent=max_sessions, default_config=self.session_config
            )
        else:
            self._manager = None

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

    async def execute_tasks_with_manager(
        self, tasks: list[TaskSpec | dict[str, Any]], state: dict[str, Any], workspace: str
    ) -> list[dict[str, Any]]:
        """
        Execute tasks using the TerminalSessionManager.

        Args:
            tasks: List of TaskSpec or task dicts.
            state: Current execution state.
            workspace: Working directory.

        Returns:
            List of result dictionaries.
        """
        if not self._manager:
            self._manager = TerminalSessionManager(
                max_concurrent=self.max_sessions, default_config=self.session_config
            )

        results = []

        async def execute_one(task: TaskSpec | dict[str, Any]) -> dict[str, Any]:
            # Convert dict to TaskSpec if needed
            if isinstance(task, dict):
                task_spec = TaskSpec(**task)
            else:
                task_spec = task

            # Build prompt
            prompt = self._build_task_prompt_enhanced(task_spec, state)

            # Create and execute session
            session_id = await self._manager.create_session(
                task_id=task_spec.id,
                prompt=prompt,
                workspace=workspace,
                context=state.get("global_context", {}),
                config=self.session_config,
            )

            # Wait for completion
            result = await self._manager.wait_for_completion(
                session_id=session_id, timeout=self.timeout
            )

            return {
                "task_id": task_spec.id,
                "session_id": session_id,
                "success": result.is_success(),
                "output": result.stdout,
                "error": result.error_message,
                "files_created": result.files_created,
                "files_modified": result.files_modified,
                "duration_seconds": result.duration_seconds,
                "return_code": result.return_code,
            }

        # Execute all tasks in parallel
        async with anyio.create_task_group() as tg:
            for task in tasks:

                async def wrapper(t: TaskSpec | dict[str, Any] = task) -> None:
                    result = await execute_one(t)
                    results.append(result)

                tg.start_soon(wrapper)

        return results

    async def _execute_task(
        self,
        task: dict[str, Any],
        state: KappaState,
    ) -> TaskResult:
        """Execute a single task in a session."""
        task_id = task.get("id", "unknown")
        task_name = task.get("name", task.get("title", "Unnamed task"))

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
                f"Task {task_name} completed: " f"{'success' if result.success else 'failed'}"
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
        """Build system prompt for the session."""
        project_name = state.get("project_name", "Unknown Project")

        return f"""You are working on the {project_name} project as part of an autonomous development team.

Your current task is: {task.get('name', task.get('title', 'Unknown'))}

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
        """Build the task execution prompt."""
        lines = [
            f"# Task: {task.get('name', task.get('title', 'Unknown'))}",
            "",
            "## Description",
            task.get("description", "No description provided"),
            "",
        ]

        # Add target files if specified
        file_targets = task.get("file_targets", task.get("files_to_create", []))
        if file_targets:
            lines.extend(
                [
                    "## Target Files",
                    *[f"- {f}" for f in file_targets],
                    "",
                ]
            )

        # Add dependencies context
        dependencies = task.get("dependencies", [])
        if dependencies:
            lines.extend(
                [
                    "## Dependencies",
                    f"This task depends on: {', '.join(dependencies)}",
                    "",
                ]
            )

        # Add execution instructions
        lines.extend(
            [
                "## Instructions",
                "1. Read any existing relevant files first",
                "2. Implement the required functionality",
                "3. Ensure code follows project conventions",
                "4. Test your changes if possible",
                "",
                "Begin implementation now.",
            ]
        )

        return "\n".join(lines)

    def _build_task_prompt_enhanced(self, task: TaskSpec, state: dict[str, Any]) -> str:
        """Build enhanced task prompt with more context."""
        lines = [
            f"# Task: {task.title}",
            "",
            "## Description",
            task.description,
            "",
        ]

        # Files to create
        if task.files_to_create:
            lines.extend(
                [
                    "## Files to Create",
                    *[f"- `{f}`" for f in task.files_to_create],
                    "",
                ]
            )

        # Files to modify
        if task.files_to_modify:
            lines.extend(
                [
                    "## Files to Modify",
                    *[f"- `{f}`" for f in task.files_to_modify],
                    "",
                ]
            )

        # Dependencies
        if task.dependencies:
            lines.extend(
                [
                    "## Dependencies",
                    f"This task depends on: {', '.join(task.dependencies)}",
                    "",
                ]
            )

        # Validation commands
        if task.validation_commands:
            lines.extend(
                [
                    "## Validation",
                    "After completing, run these commands to validate:",
                    *[f"- `{cmd}`" for cmd in task.validation_commands],
                    "",
                ]
            )

        # Add global context if available
        global_context = state.get("global_context", {})
        if global_context:
            if "tech_stack" in global_context:
                lines.extend(
                    [
                        "## Tech Stack",
                        *[f"- {k}: {v}" for k, v in global_context["tech_stack"].items()],
                        "",
                    ]
                )

        lines.extend(
            [
                "## Instructions",
                "1. Read any existing relevant files first",
                "2. Implement the required functionality",
                "3. Follow project code conventions",
                "4. Include proper error handling",
                "5. Add type hints to all functions",
                "",
                "Execute this task completely. Create production-quality code.",
            ]
        )

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

        # Also cancel manager sessions if using manager
        if self._manager:
            await self._manager.kill_all()

    @property
    def active_count(self) -> int:
        """Get count of active sessions."""
        count = len(self._active_sessions)
        if self._manager:
            count += len(self._manager.active_sessions)
        return count

    def get_session(self, session_id: str) -> TerminalSession | None:
        """Get an active session by ID."""
        return self._active_sessions.get(session_id)

    def get_task_router(self) -> TaskRouter:
        """Get the task router instance."""
        return self._task_router

    def get_manager(self) -> TerminalSessionManager | None:
        """Get the session manager instance."""
        return self._manager
