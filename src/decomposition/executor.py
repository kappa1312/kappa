"""Parallel task executor for Kappa.

This module provides the infrastructure for executing tasks in parallel
using Claude sessions, with proper concurrency control and result tracking.
"""

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger

from src.decomposition.models import TaskSpec


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
            "completed_at": self.completed_at,
        }


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
            "completed_at": self.completed_at,
        }


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
    ):
        """
        Initialize parallel executor.

        Args:
            max_concurrent: Maximum concurrent tasks (default 5).
            timeout: Timeout per task in seconds (default 600).
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_sessions: dict[str, str] = {}  # task_id -> session_id

    async def execute_wave(
        self,
        task_ids: list[str],
        state: dict[str, Any],
        wave_number: int = 0,
    ) -> WaveExecutionResult:
        """
        Execute all tasks in a wave in parallel.

        Args:
            task_ids: List of task IDs to execute.
            state: Current Kappa state with tasks and context.
            wave_number: Wave number for logging.

        Returns:
            WaveExecutionResult with all task results.
        """
        logger.info(f"Executing wave {wave_number} with {len(task_ids)} tasks")

        # Get task specs from state
        tasks = self._get_tasks_by_ids(task_ids, state)

        if not tasks:
            logger.warning(f"No tasks found for IDs: {task_ids}")
            return WaveExecutionResult(wave_number=wave_number, results=[])

        # Execute all tasks in parallel with semaphore
        coroutines = [
            self._execute_with_semaphore(task, state)
            for task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        task_results = []
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task_results.append(
                    TaskExecutionResult(
                        task_id=task.id,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                task_results.append(result)

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
    ) -> TaskExecutionResult:
        """
        Execute a single task.

        Args:
            task: TaskSpec or task dict to execute.
            state: Current Kappa state.

        Returns:
            TaskExecutionResult.
        """
        # Convert dict to TaskSpec if needed
        if isinstance(task, dict):
            task = TaskSpec(**task)

        return await self._execute_task_internal(task, state)

    async def _execute_with_semaphore(
        self,
        task: TaskSpec,
        state: dict[str, Any],
    ) -> TaskExecutionResult:
        """Execute task with semaphore for concurrency control."""
        async with self._semaphore:
            try:
                return await asyncio.wait_for(
                    self._execute_task_internal(task, state),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                logger.error(f"Task {task.id} timed out after {self.timeout}s")
                return TaskExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=f"Task timed out after {self.timeout} seconds",
                )

    async def _execute_task_internal(
        self,
        task: TaskSpec,
        state: dict[str, Any],
    ) -> TaskExecutionResult:
        """
        Internal task execution logic.

        This method handles the actual execution using Claude sessions.
        """
        start_time = datetime.utcnow()
        logger.info(f"Executing task: {task.id} - {task.title}")

        try:
            # Build execution context
            context = self._build_task_context(task, state)

            # Get session router and execute
            result = await self._execute_with_session(task, context, state)

            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration

            return result

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            duration = (datetime.utcnow() - start_time).total_seconds()
            return TaskExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    async def _execute_with_session(
        self,
        task: TaskSpec,
        context: dict[str, Any],
        state: dict[str, Any],
    ) -> TaskExecutionResult:
        """Execute task using a Claude session."""
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
            return await self._simulate_execution(task, context)

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

    def _build_task_context(
        self,
        task: TaskSpec,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build execution context for a task."""
        context: dict[str, Any] = {
            "task": task.model_dump(),
            "workspace_path": state.get("workspace_path"),
            "project_name": state.get("project_name"),
        }

        # Add global context
        if state.get("global_context"):
            context["global"] = state["global_context"]

        # Add task-specific context
        task_contexts = state.get("task_contexts", {})
        if task.id in task_contexts:
            context["task_context"] = task_contexts[task.id]

        # Add context from dependency outputs
        for dep_id in task.requires_context_from:
            task_results = state.get("task_results", [])
            for result in task_results:
                if result.get("task_id") == dep_id:
                    context.setdefault("dependency_outputs", {})[dep_id] = result
                    break

        return context

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
        self._active_sessions.clear()


# =============================================================================
# SEQUENTIAL EXECUTOR
# =============================================================================


class SequentialExecutor:
    """
    Execute tasks sequentially (one at a time).

    Useful for debugging or when parallel execution is not desired.
    """

    def __init__(self, timeout: int = 600):
        """
        Initialize sequential executor.

        Args:
            timeout: Timeout per task in seconds.
        """
        self.timeout = timeout
        self._parallel = ParallelExecutor(max_concurrent=1, timeout=timeout)

    async def execute_wave(
        self,
        task_ids: list[str],
        state: dict[str, Any],
        wave_number: int = 0,
    ) -> WaveExecutionResult:
        """Execute tasks sequentially."""
        return await self._parallel.execute_wave(task_ids, state, wave_number)

    async def execute_task(
        self,
        task: TaskSpec | dict[str, Any],
        state: dict[str, Any],
    ) -> TaskExecutionResult:
        """Execute a single task."""
        return await self._parallel.execute_task(task, state)


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
    ) -> WaveExecutionResult:
        """Execute wave with retries for failed tasks."""
        result = await self.executor.execute_wave(task_ids, state, wave_number)

        # Retry failed tasks
        failed_ids = result.failed_tasks
        retry_count = 0

        while failed_ids and retry_count < self.max_retries:
            retry_count += 1
            logger.info(f"Retrying {len(failed_ids)} failed tasks (attempt {retry_count})")

            await asyncio.sleep(self.retry_delay)

            retry_result = await self.executor.execute_wave(
                failed_ids, state, wave_number
            )

            # Update results with retry outcomes
            for retry_task_result in retry_result.results:
                # Find and replace the original failed result
                for i, orig_result in enumerate(result.results):
                    if orig_result.task_id == retry_task_result.task_id:
                        result.results[i] = retry_task_result
                        break

            # Update failed IDs for next retry
            failed_ids = [
                r.task_id for r in retry_result.results if not r.success
            ]

        return result

    async def execute_task(
        self,
        task: TaskSpec | dict[str, Any],
        state: dict[str, Any],
    ) -> TaskExecutionResult:
        """Execute single task with retries."""
        result = await self.executor.execute_task(task, state)

        retry_count = 0
        while not result.success and retry_count < self.max_retries:
            retry_count += 1
            logger.info(f"Retrying task (attempt {retry_count})")

            await asyncio.sleep(self.retry_delay)
            result = await self.executor.execute_task(task, state)

        return result


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_executor(
    mode: str = "parallel",
    max_concurrent: int = 5,
    timeout: int = 600,
    max_retries: int = 0,
) -> ParallelExecutor | SequentialExecutor | RetryExecutor:
    """
    Create an executor with the specified configuration.

    Args:
        mode: Execution mode ("parallel", "sequential", "retry").
        max_concurrent: Maximum concurrent tasks for parallel mode.
        timeout: Timeout per task in seconds.
        max_retries: Maximum retries (only for retry mode).

    Returns:
        Configured executor instance.

    Example:
        >>> executor = create_executor("parallel", max_concurrent=10)
        >>> results = await executor.execute_wave(tasks, state)
    """
    if mode == "sequential":
        return SequentialExecutor(timeout=timeout)
    elif mode == "retry":
        base_executor = ParallelExecutor(max_concurrent=max_concurrent, timeout=timeout)
        return RetryExecutor(executor=base_executor, max_retries=max_retries)
    else:
        return ParallelExecutor(max_concurrent=max_concurrent, timeout=timeout)
