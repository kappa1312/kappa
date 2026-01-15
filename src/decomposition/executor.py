"""
Parallel task executor for Kappa.

This module provides the infrastructure for executing tasks in parallel
using Claude sessions, with proper concurrency control, context sharing,
and result tracking.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any

from loguru import logger

from src.decomposition.models import DependencyGraph, TaskSpec
from src.prompts.builder import PromptContext, get_prompt_builder
from src.sessions.base import SessionConfig, SessionResult

# =============================================================================
# RESULT MODELS
# =============================================================================


class TaskExecutionResult:
    """Result of a single task execution."""

    def __init__(
        self,
        task_id: str,
        success: bool,
        output: str | None = None,
        error: str | None = None,
        files_created: list[str] | None = None,
        files_modified: list[str] | None = None,
        session_id: str | None = None,
        duration_seconds: float = 0.0,
        tokens_used: int = 0,
        wave_number: int | None = None,
        exports: dict[str, list[str]] | None = None,
        types_exported: list[str] | None = None,
    ):
        self.task_id = task_id
        self.success = success
        self.output = output
        self.error = error
        self.files_created = files_created or []
        self.files_modified = files_modified or []
        self.session_id = session_id
        self.duration_seconds = duration_seconds
        self.tokens_used = tokens_used
        self.wave_number = wave_number
        self.exports = exports or {}
        self.types_exported = types_exported or []
        self.completed_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
            "wave_number": self.wave_number,
            "exports": self.exports,
            "types_exported": self.types_exported,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_session_result(
        cls,
        task_id: str,
        result: SessionResult,
        wave_number: int | None = None,
    ) -> "TaskExecutionResult":
        """Create from SessionResult."""
        return cls(
            task_id=task_id,
            success=result.is_success(),
            output=result.stdout,
            error=result.error_message,
            files_created=result.files_created,
            files_modified=result.files_modified,
            session_id=result.session_id,
            duration_seconds=result.duration_seconds or 0.0,
            tokens_used=result.token_usage.get("total", 0),
            wave_number=wave_number,
        )


class WaveExecutionResult:
    """Result of executing a full wave of tasks."""

    def __init__(
        self,
        wave_number: int,
        results: list[TaskExecutionResult],
    ):
        self.wave_number = wave_number
        self.results = results
        self.completed_at = datetime.utcnow().isoformat()

    @property
    def completed_tasks(self) -> list[str]:
        """Get IDs of successfully completed tasks."""
        return [r.task_id for r in self.results if r.success]

    @property
    def failed_tasks(self) -> list[str]:
        """Get IDs of failed tasks."""
        return [r.task_id for r in self.results if not r.success]

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if not self.results:
            return 0.0
        return len(self.completed_tasks) / len(self.results)

    @property
    def total_duration(self) -> float:
        """Get total execution duration (max of all tasks)."""
        if not self.results:
            return 0.0
        return max(r.duration_seconds for r in self.results)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(r.tokens_used for r in self.results)

    @property
    def all_files_created(self) -> list[str]:
        """Get all files created in this wave."""
        files = []
        for r in self.results:
            files.extend(r.files_created)
        return list(set(files))

    @property
    def all_files_modified(self) -> list[str]:
        """Get all files modified in this wave."""
        files = []
        for r in self.results:
            files.extend(r.files_modified)
        return list(set(files))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wave_number": self.wave_number,
            "results": [r.to_dict() for r in self.results],
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.success_rate,
            "total_duration": self.total_duration,
            "total_tokens": self.total_tokens,
            "all_files_created": self.all_files_created,
            "all_files_modified": self.all_files_modified,
            "completed_at": self.completed_at,
        }


# =============================================================================
# EXECUTION CALLBACK TYPE
# =============================================================================


ExecutionCallback = Callable[[str, TaskExecutionResult], None]


# =============================================================================
# PARALLEL EXECUTOR
# =============================================================================


class ParallelExecutor:
    """
    Execute tasks in parallel using Claude sessions.

    Manages concurrency, session routing, and result collection for
    parallel task execution within waves.

    Attributes:
        max_concurrent: Maximum number of concurrent executions.
        timeout: Timeout per task in seconds.

    Example:
        >>> executor = ParallelExecutor(max_concurrent=5)
        >>> results = await executor.execute_wave(
        ...     task_ids=["task-1", "task-2", "task-3"],
        ...     state=kappa_state
        ... )
        >>> print(f"Completed: {len(results.completed_tasks)}")
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        timeout: int = 600,
        workspace: str = ".",
        use_terminal_manager: bool = True,
    ):
        """
        Initialize parallel executor.

        Args:
            max_concurrent: Maximum concurrent tasks (default 5).
            timeout: Timeout per task in seconds (default 600).
            workspace: Default workspace directory.
            use_terminal_manager: Whether to use new TerminalSessionManager.
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.workspace = workspace
        self.use_terminal_manager = use_terminal_manager
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_sessions: dict[str, str] = {}  # task_id -> session_id
        self._prompt_builder = get_prompt_builder()
        self._callbacks: list[ExecutionCallback] = []

        # Session manager (lazy init)
        self._session_manager = None

    def add_callback(self, callback: ExecutionCallback) -> None:
        """Add a callback for task completion events."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: ExecutionCallback) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def _emit_callback(self, task_id: str, result: TaskExecutionResult) -> None:
        """Emit callback to all registered listeners."""
        for callback in self._callbacks:
            try:
                callback(task_id, result)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    async def _get_session_manager(self):
        """Get or create session manager."""
        if self._session_manager is None:
            try:
                from src.sessions.terminal import TerminalSessionManager

                self._session_manager = TerminalSessionManager(
                    max_concurrent=self.max_concurrent,
                    default_config=SessionConfig(
                        timeout_seconds=self.timeout,
                        working_directory=self.workspace,
                    ),
                )
            except ImportError:
                logger.warning("TerminalSessionManager not available")
        return self._session_manager

    async def execute_wave(
        self,
        task_ids: list[str],
        state: dict[str, Any],
        wave_number: int = 0,
        context: PromptContext | None = None,
    ) -> WaveExecutionResult:
        """
        Execute all tasks in a wave in parallel.

        Args:
            task_ids: List of task IDs to execute.
            state: Current Kappa state with tasks and context.
            wave_number: Wave number for logging.
            context: Optional PromptContext for shared context.

        Returns:
            WaveExecutionResult with all task results.
        """
        logger.info(f"Executing wave {wave_number} with {len(task_ids)} tasks")

        # Get task specs from state
        tasks = self._get_tasks_by_ids(task_ids, state)

        if not tasks:
            logger.warning(f"No tasks found for IDs: {task_ids}")
            return WaveExecutionResult(wave_number=wave_number, results=[])

        # Create context if not provided
        if context is None:
            context = PromptContext(
                project_name=state.get("project_name", "Unknown"),
                workspace=state.get("workspace_path", self.workspace),
                tech_stack=state.get("global_context", {}).get("tech_stack", {}),
            )

        # Execute all tasks in parallel with semaphore
        coroutines = [
            self._execute_with_semaphore(task, state, wave_number, context, task_ids)
            for task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        task_results = []
        for task, result in zip(tasks, results, strict=False):
            if isinstance(result, Exception):
                error_result = TaskExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=str(result),
                    wave_number=wave_number,
                )
                task_results.append(error_result)
                self._emit_callback(task.id, error_result)
            else:
                task_results.append(result)
                self._emit_callback(task.id, result)

        wave_result = WaveExecutionResult(
            wave_number=wave_number,
            results=task_results,
        )

        logger.info(
            f"Wave {wave_number} complete: "
            f"{len(wave_result.completed_tasks)} succeeded, "
            f"{len(wave_result.failed_tasks)} failed"
        )

        return wave_result

    async def execute_task(
        self,
        task: TaskSpec | dict[str, Any],
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> TaskExecutionResult:
        """
        Execute a single task.

        Args:
            task: TaskSpec or task dict to execute.
            state: Current Kappa state.
            context: Optional PromptContext.

        Returns:
            TaskExecutionResult.
        """
        # Convert dict to TaskSpec if needed
        if isinstance(task, dict):
            task = TaskSpec(**task)

        if context is None:
            context = PromptContext(
                project_name=state.get("project_name", "Unknown"),
                workspace=state.get("workspace_path", self.workspace),
            )

        return await self._execute_task_internal(task, state, context)

    async def execute_graph(
        self,
        graph: DependencyGraph,
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> list[WaveExecutionResult]:
        """
        Execute all tasks in a dependency graph wave by wave.

        Args:
            graph: DependencyGraph with tasks and waves.
            state: Current Kappa state.
            context: Optional PromptContext.

        Returns:
            List of WaveExecutionResult for each wave.
        """
        results = []

        if context is None:
            context = PromptContext(
                project_name=state.get("project_name", "Unknown"),
                workspace=state.get("workspace_path", self.workspace),
            )

        for wave_number in range(graph.total_waves):
            wave_task_ids = graph.waves[wave_number]
            logger.info(f"Starting wave {wave_number} with {len(wave_task_ids)} tasks")

            wave_result = await self.execute_wave(
                task_ids=wave_task_ids,
                state=state,
                wave_number=wave_number,
                context=context,
            )

            results.append(wave_result)

            # Update context with wave outputs
            for task_result in wave_result.results:
                if task_result.success:
                    context.add_task_output(task_result.task_id, task_result.to_dict())

            context.add_wave_output(wave_number, wave_result.to_dict())

            # Update state with completed tasks
            state.setdefault("completed_tasks", []).extend(wave_result.completed_tasks)
            state.setdefault("failed_tasks", []).extend(wave_result.failed_tasks)

            # Check if we should abort
            if wave_result.success_rate < 0.5:
                logger.error(f"Wave {wave_number} had >50% failure rate, aborting execution")
                break

        return results

    async def _execute_with_semaphore(
        self,
        task: TaskSpec,
        state: dict[str, Any],
        wave_number: int,
        context: PromptContext,
        parallel_task_ids: list[str],
    ) -> TaskExecutionResult:
        """Execute task with semaphore for concurrency control."""
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    self._execute_task_internal(
                        task, state, context, wave_number, parallel_task_ids
                    ),
                    timeout=self.timeout,
                )
            except TimeoutError:
                logger.error(f"Task {task.id} timed out after {self.timeout}s")
                return TaskExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=f"Task timed out after {self.timeout} seconds",
                    wave_number=wave_number,
                )

    async def _execute_task_internal(
        self,
        task: TaskSpec,
        state: dict[str, Any],
        context: PromptContext,
        wave_number: int | None = None,
        parallel_task_ids: list[str] | None = None,
    ) -> TaskExecutionResult:
        """
        Internal task execution logic.

        This method handles the actual execution using Claude sessions.
        """
        start_time = datetime.utcnow()
        logger.info(f"Executing task: {task.id} - {task.title}")

        try:
            # Build prompt with context
            prompt = self._build_task_prompt(
                task, context, wave_number or 0, parallel_task_ids or []
            )

            # Execute with session manager
            if self.use_terminal_manager:
                result = await self._execute_with_manager(task, prompt, state)
            else:
                result = await self._execute_with_router(task, state)

            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration
            result.wave_number = wave_number

            return result

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            duration = (datetime.utcnow() - start_time).total_seconds()
            return TaskExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
                duration_seconds=duration,
                wave_number=wave_number,
            )

    def _build_task_prompt(
        self,
        task: TaskSpec,
        context: PromptContext,
        wave_number: int,
        parallel_task_ids: list[str],
    ) -> str:
        """Build prompt for task execution."""
        return self._prompt_builder.build_parallel_task_prompt(
            task=task,
            context=context,
            wave_number=wave_number,
            parallel_tasks=parallel_task_ids,
        )

    async def _execute_with_manager(
        self,
        task: TaskSpec,
        prompt: str,
        state: dict[str, Any],
    ) -> TaskExecutionResult:
        """Execute task using TerminalSessionManager."""
        manager = await self._get_session_manager()

        if manager is None:
            return await self._simulate_execution(task, {})

        workspace = state.get("workspace_path", self.workspace)

        try:
            # Create session
            session_id = await manager.create_session(
                task_id=task.id,
                prompt=prompt,
                workspace=workspace,
                context=state.get("global_context", {}),
            )

            self._active_sessions[task.id] = session_id

            # Wait for completion
            result = await manager.wait_for_completion(
                session_id=session_id,
                timeout=self.timeout,
            )

            # Convert to TaskExecutionResult
            return TaskExecutionResult.from_session_result(task.id, result)

        except Exception as e:
            logger.error(f"Session execution failed for task {task.id}: {e}")
            return TaskExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
            )
        finally:
            self._active_sessions.pop(task.id, None)

    async def _execute_with_router(
        self,
        task: TaskSpec,
        state: dict[str, Any],
    ) -> TaskExecutionResult:
        """Execute task using SessionRouter (legacy)."""
        try:
            from src.sessions.router import SessionRouter

            router = SessionRouter(
                max_sessions=self.max_concurrent,
                timeout=self.timeout,
            )

            # Execute single task
            results = await router.execute_tasks([task.model_dump()], state)

            if results and len(results) > 0:
                result = results[0]

                # Handle Pydantic model or dict
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                else:
                    result_dict = result

                return TaskExecutionResult(
                    task_id=task.id,
                    success=result_dict.get("success", False),
                    output=result_dict.get("output"),
                    error=result_dict.get("error"),
                    files_created=result_dict.get("files_created", []),
                    files_modified=result_dict.get("files_modified", []),
                    session_id=result_dict.get("session_id"),
                    tokens_used=result_dict.get("token_usage", {}).get("total", 0),
                )

            return TaskExecutionResult(
                task_id=task.id,
                success=False,
                error="No result from session",
            )

        except ImportError:
            # SessionRouter not available, simulate execution
            logger.warning("SessionRouter not available, simulating execution")
            return await self._simulate_execution(task, {})

    async def _simulate_execution(
        self,
        task: TaskSpec,
        context: dict[str, Any],
    ) -> TaskExecutionResult:
        """
        Simulate task execution for testing.

        Used when actual session execution is not available.
        """
        logger.info(f"Simulating execution for task: {task.id}")

        # Simulate some work
        await asyncio.sleep(0.1)

        return TaskExecutionResult(
            task_id=task.id,
            success=True,
            output=f"Simulated output for {task.title}",
            files_created=task.files_to_create,
            files_modified=task.files_to_modify,
        )

    def _get_tasks_by_ids(
        self,
        task_ids: list[str],
        state: dict[str, Any],
    ) -> list[TaskSpec]:
        """Get TaskSpec objects from state by IDs."""
        tasks = []
        task_id_set = set(task_ids)

        for task_dict in state.get("tasks", []):
            if task_dict.get("id") in task_id_set:
                try:
                    tasks.append(TaskSpec(**task_dict))
                except Exception as e:
                    logger.warning(f"Could not create TaskSpec: {e}")

        return tasks

    @property
    def active_sessions(self) -> dict[str, str]:
        """Get currently active sessions."""
        return self._active_sessions.copy()

    async def cancel_all(self) -> None:
        """Cancel all active sessions."""
        logger.info("Cancelling all active sessions")

        if self._session_manager:
            await self._session_manager.kill_all()

        self._active_sessions.clear()


# =============================================================================
# SEQUENTIAL EXECUTOR
# =============================================================================


class SequentialExecutor:
    """
    Execute tasks sequentially (one at a time).

    Useful for debugging or when parallel execution is not desired.
    """

    def __init__(self, timeout: int = 600, workspace: str = "."):
        """
        Initialize sequential executor.

        Args:
            timeout: Timeout per task in seconds.
            workspace: Default workspace directory.
        """
        self.timeout = timeout
        self._parallel = ParallelExecutor(
            max_concurrent=1,
            timeout=timeout,
            workspace=workspace,
        )

    async def execute_wave(
        self,
        task_ids: list[str],
        state: dict[str, Any],
        wave_number: int = 0,
        context: PromptContext | None = None,
    ) -> WaveExecutionResult:
        """Execute tasks sequentially."""
        return await self._parallel.execute_wave(task_ids, state, wave_number, context)

    async def execute_task(
        self,
        task: TaskSpec | dict[str, Any],
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> TaskExecutionResult:
        """Execute a single task."""
        return await self._parallel.execute_task(task, state, context)

    async def execute_graph(
        self,
        graph: DependencyGraph,
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> list[WaveExecutionResult]:
        """Execute all tasks in a dependency graph."""
        return await self._parallel.execute_graph(graph, state, context)


# =============================================================================
# RETRY EXECUTOR
# =============================================================================


class RetryExecutor:
    """
    Executor wrapper that retries failed tasks.

    Wraps another executor and retries failed tasks up to a maximum
    number of attempts.
    """

    def __init__(
        self,
        executor: ParallelExecutor | SequentialExecutor | None = None,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        """
        Initialize retry executor.

        Args:
            executor: Underlying executor (default ParallelExecutor).
            max_retries: Maximum retry attempts per task.
            retry_delay: Delay between retries in seconds.
        """
        self.executor = executor or ParallelExecutor()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def execute_wave(
        self,
        task_ids: list[str],
        state: dict[str, Any],
        wave_number: int = 0,
        context: PromptContext | None = None,
    ) -> WaveExecutionResult:
        """Execute wave with retries for failed tasks."""
        result = await self.executor.execute_wave(task_ids, state, wave_number, context)

        # Retry failed tasks
        failed_ids = result.failed_tasks
        retry_count = 0

        while failed_ids and retry_count < self.max_retries:
            retry_count += 1
            logger.info(f"Retrying {len(failed_ids)} failed tasks (attempt {retry_count})")

            await asyncio.sleep(self.retry_delay)

            retry_result = await self.executor.execute_wave(failed_ids, state, wave_number, context)

            # Update results with retry outcomes
            for retry_task_result in retry_result.results:
                # Find and replace the original failed result
                for i, orig_result in enumerate(result.results):
                    if orig_result.task_id == retry_task_result.task_id:
                        result.results[i] = retry_task_result
                        break

            # Update failed IDs for next retry
            failed_ids = [r.task_id for r in retry_result.results if not r.success]

        return result

    async def execute_task(
        self,
        task: TaskSpec | dict[str, Any],
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> TaskExecutionResult:
        """Execute single task with retries."""
        result = await self.executor.execute_task(task, state, context)

        retry_count = 0
        while not result.success and retry_count < self.max_retries:
            retry_count += 1
            logger.info(f"Retrying task (attempt {retry_count})")

            await asyncio.sleep(self.retry_delay)
            result = await self.executor.execute_task(task, state, context)

        return result

    async def execute_graph(
        self,
        graph: DependencyGraph,
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> list[WaveExecutionResult]:
        """Execute graph with retries per wave."""
        results = []

        for wave_number in range(graph.total_waves):
            wave_result = await self.execute_wave(
                task_ids=graph.waves[wave_number],
                state=state,
                wave_number=wave_number,
                context=context,
            )
            results.append(wave_result)

            # Update state
            state.setdefault("completed_tasks", []).extend(wave_result.completed_tasks)
            state.setdefault("failed_tasks", []).extend(wave_result.failed_tasks)

        return results


# =============================================================================
# DRY RUN EXECUTOR
# =============================================================================


class DryRunExecutor:
    """
    Executor that simulates execution without actually running tasks.

    Useful for testing task decomposition and dependency resolution
    without executing Claude sessions.
    """

    def __init__(
        self,
        delay_per_task: float = 0.1,
        success_rate: float = 1.0,
    ):
        """
        Initialize dry run executor.

        Args:
            delay_per_task: Simulated delay per task in seconds.
            success_rate: Probability of task success (0.0 to 1.0).
        """
        self.delay_per_task = delay_per_task
        self.success_rate = success_rate

    async def execute_wave(
        self,
        task_ids: list[str],
        state: dict[str, Any],
        wave_number: int = 0,
        context: PromptContext | None = None,
    ) -> WaveExecutionResult:
        """Simulate wave execution."""
        import random

        tasks = []
        for task_dict in state.get("tasks", []):
            if task_dict.get("id") in task_ids:
                tasks.append(task_dict)

        results = []
        for task in tasks:
            await asyncio.sleep(self.delay_per_task)

            success = random.random() < self.success_rate

            results.append(
                TaskExecutionResult(
                    task_id=task.get("id", "unknown"),
                    success=success,
                    output=(
                        f"Dry run output for {task.get('title', 'unknown')}" if success else None
                    ),
                    error="Simulated failure" if not success else None,
                    files_created=task.get("files_to_create", []),
                    files_modified=task.get("files_to_modify", []),
                    wave_number=wave_number,
                )
            )

        return WaveExecutionResult(
            wave_number=wave_number,
            results=results,
        )

    async def execute_task(
        self,
        task: TaskSpec | dict[str, Any],
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> TaskExecutionResult:
        """Simulate single task execution."""
        import random

        if isinstance(task, dict):
            task = TaskSpec(**task)

        await asyncio.sleep(self.delay_per_task)
        success = random.random() < self.success_rate

        return TaskExecutionResult(
            task_id=task.id,
            success=success,
            output=f"Dry run output for {task.title}" if success else None,
            error="Simulated failure" if not success else None,
            files_created=task.files_to_create,
            files_modified=task.files_to_modify,
        )

    async def execute_graph(
        self,
        graph: DependencyGraph,
        state: dict[str, Any],
        context: PromptContext | None = None,
    ) -> list[WaveExecutionResult]:
        """Simulate graph execution."""
        results = []

        # Ensure state has completed_tasks and failed_tasks lists
        if "completed_tasks" not in state:
            state["completed_tasks"] = []
        if "failed_tasks" not in state:
            state["failed_tasks"] = []

        for wave_number in range(graph.total_waves):
            wave_result = await self.execute_wave(
                task_ids=graph.waves[wave_number],
                state=state,
                wave_number=wave_number,
                context=context,
            )
            results.append(wave_result)

            # Update state with completed/failed tasks
            state["completed_tasks"].extend(wave_result.completed_tasks)
            state["failed_tasks"].extend(wave_result.failed_tasks)

            # Update context if provided
            if context is not None:
                for result in wave_result.results:
                    if result.success:
                        context.completed_task_outputs[result.task_id] = {
                            "output": result.output,
                            "files_created": result.files_created,
                            "files_modified": result.files_modified,
                            "wave_number": wave_number,
                        }

            # Stop if too many failures
            if wave_result.success_rate < 0.5:
                break

        return results


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_executor(
    mode: str = "parallel",
    max_concurrent: int = 5,
    timeout: int = 600,
    max_retries: int = 0,
    workspace: str = ".",
    use_terminal_manager: bool = True,
) -> ParallelExecutor | SequentialExecutor | RetryExecutor | DryRunExecutor:
    """
    Create an executor with the specified configuration.

    Args:
        mode: Execution mode ("parallel", "sequential", "retry", "dry_run").
        max_concurrent: Maximum concurrent tasks for parallel mode.
        timeout: Timeout per task in seconds.
        max_retries: Maximum retries (only for retry mode).
        workspace: Default workspace directory.
        use_terminal_manager: Whether to use TerminalSessionManager.

    Returns:
        Configured executor instance.

    Example:
        >>> executor = create_executor("parallel", max_concurrent=10)
        >>> results = await executor.execute_wave(tasks, state)
    """
    if mode == "sequential":
        return SequentialExecutor(timeout=timeout, workspace=workspace)
    elif mode == "retry":
        base_executor = ParallelExecutor(
            max_concurrent=max_concurrent,
            timeout=timeout,
            workspace=workspace,
            use_terminal_manager=use_terminal_manager,
        )
        return RetryExecutor(executor=base_executor, max_retries=max_retries)
    elif mode == "dry_run":
        return DryRunExecutor()
    else:
        return ParallelExecutor(
            max_concurrent=max_concurrent,
            timeout=timeout,
            workspace=workspace,
            use_terminal_manager=use_terminal_manager,
        )
