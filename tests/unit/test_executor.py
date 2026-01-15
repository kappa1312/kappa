"""Unit tests for the parallel executor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.decomposition.executor import (
    TaskExecutionResult,
    WaveExecutionResult,
    ParallelExecutor,
    SequentialExecutor,
    RetryExecutor,
    DryRunExecutor,
    create_executor,
)
from src.decomposition.models import TaskSpec, DependencyGraph, TaskCategory, Complexity
from src.prompts.builder import PromptContext


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_task_spec() -> TaskSpec:
    """Create a sample TaskSpec for testing."""
    return TaskSpec(
        id="task-1",
        title="Create user model",
        description="Create the User model with id, name, email fields",
        category=TaskCategory.DATA_MODEL,
        complexity=Complexity.LOW,
        dependencies=[],
        files_to_create=["src/models/user.py"],
        files_to_modify=[],
        priority=1,
        estimated_duration_minutes=30,
    )


@pytest.fixture
def sample_tasks() -> list[TaskSpec]:
    """Create a list of sample tasks."""
    return [
        TaskSpec(
            id="task-1",
            title="Create user model",
            description="Create User model",
            category=TaskCategory.DATA_MODEL,
            complexity=Complexity.LOW,
            dependencies=[],
            files_to_create=["src/models/user.py"],
            files_to_modify=[],
        ),
        TaskSpec(
            id="task-2",
            title="Create user service",
            description="Create UserService",
            category=TaskCategory.BUSINESS_LOGIC,
            complexity=Complexity.MEDIUM,
            dependencies=["task-1"],
            files_to_create=["src/services/user.py"],
            files_to_modify=[],
        ),
        TaskSpec(
            id="task-3",
            title="Create user API",
            description="Create user API endpoints",
            category=TaskCategory.API,
            complexity=Complexity.MEDIUM,
            dependencies=["task-2"],
            files_to_create=["src/api/user.py"],
            files_to_modify=[],
        ),
    ]


@pytest.fixture
def sample_state(sample_tasks: list[TaskSpec]) -> dict:
    """Create sample execution state."""
    return {
        "project_id": "test-project",
        "project_name": "test",
        "workspace_path": "/tmp/test",
        "tasks": [t.model_dump() for t in sample_tasks],
        "dependency_graph": {
            "waves": [["task-1"], ["task-2"], ["task-3"]],
            "total_waves": 3,
        },
        "global_context": {
            "tech_stack": {"language": "python"},
        },
        "completed_tasks": [],
        "failed_tasks": [],
    }


@pytest.fixture
def sample_dependency_graph(sample_tasks: list[TaskSpec]) -> DependencyGraph:
    """Create a sample dependency graph."""
    graph = DependencyGraph()
    for task in sample_tasks:
        graph.add_task(task)
    graph.waves = [["task-1"], ["task-2"], ["task-3"]]
    return graph


@pytest.fixture
def prompt_context() -> PromptContext:
    """Create sample prompt context."""
    return PromptContext(
        project_name="test-project",
        workspace="/tmp/test",
        tech_stack={"language": "python"},
    )


# =============================================================================
# TASK EXECUTION RESULT TESTS
# =============================================================================


class TestTaskExecutionResult:
    """Tests for TaskExecutionResult."""

    def test_create_success_result(self) -> None:
        """Test creating a successful result."""
        result = TaskExecutionResult(
            task_id="task-1",
            success=True,
            output="Task completed",
            files_created=["src/models/user.py"],
        )

        assert result.task_id == "task-1"
        assert result.success is True
        assert result.output == "Task completed"
        assert result.error is None
        assert "user.py" in result.files_created[0]

    def test_create_failure_result(self) -> None:
        """Test creating a failure result."""
        result = TaskExecutionResult(
            task_id="task-1",
            success=False,
            error="Something went wrong",
        )

        assert result.task_id == "task-1"
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = TaskExecutionResult(
            task_id="task-1",
            success=True,
            output="Done",
            files_created=["file.py"],
            wave_number=0,
        )

        d = result.to_dict()

        assert d["task_id"] == "task-1"
        assert d["success"] is True
        assert d["output"] == "Done"
        assert d["wave_number"] == 0
        assert "completed_at" in d


# =============================================================================
# WAVE EXECUTION RESULT TESTS
# =============================================================================


class TestWaveExecutionResult:
    """Tests for WaveExecutionResult."""

    def test_create_wave_result(self) -> None:
        """Test creating a wave result."""
        results = [
            TaskExecutionResult(task_id="task-1", success=True),
            TaskExecutionResult(task_id="task-2", success=True),
            TaskExecutionResult(task_id="task-3", success=False, error="Failed"),
        ]

        wave = WaveExecutionResult(wave_number=0, results=results)

        assert wave.wave_number == 0
        assert len(wave.results) == 3

    def test_completed_tasks(self) -> None:
        """Test getting completed task IDs."""
        results = [
            TaskExecutionResult(task_id="task-1", success=True),
            TaskExecutionResult(task_id="task-2", success=True),
            TaskExecutionResult(task_id="task-3", success=False),
        ]

        wave = WaveExecutionResult(wave_number=0, results=results)

        assert wave.completed_tasks == ["task-1", "task-2"]

    def test_failed_tasks(self) -> None:
        """Test getting failed task IDs."""
        results = [
            TaskExecutionResult(task_id="task-1", success=True),
            TaskExecutionResult(task_id="task-2", success=False),
            TaskExecutionResult(task_id="task-3", success=False),
        ]

        wave = WaveExecutionResult(wave_number=0, results=results)

        assert wave.failed_tasks == ["task-2", "task-3"]

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        results = [
            TaskExecutionResult(task_id="task-1", success=True),
            TaskExecutionResult(task_id="task-2", success=True),
            TaskExecutionResult(task_id="task-3", success=False),
            TaskExecutionResult(task_id="task-4", success=False),
        ]

        wave = WaveExecutionResult(wave_number=0, results=results)

        assert wave.success_rate == 0.5

    def test_empty_wave_success_rate(self) -> None:
        """Test success rate for empty wave."""
        wave = WaveExecutionResult(wave_number=0, results=[])

        assert wave.success_rate == 0.0

    def test_total_duration(self) -> None:
        """Test total duration calculation."""
        results = [
            TaskExecutionResult(task_id="task-1", success=True, duration_seconds=10.0),
            TaskExecutionResult(task_id="task-2", success=True, duration_seconds=20.0),
            TaskExecutionResult(task_id="task-3", success=True, duration_seconds=15.0),
        ]

        wave = WaveExecutionResult(wave_number=0, results=results)

        assert wave.total_duration == 20.0  # Max of all durations

    def test_all_files_created(self) -> None:
        """Test aggregating files created."""
        results = [
            TaskExecutionResult(task_id="task-1", success=True, files_created=["a.py", "b.py"]),
            TaskExecutionResult(task_id="task-2", success=True, files_created=["c.py"]),
            TaskExecutionResult(task_id="task-3", success=True, files_created=["a.py"]),  # Duplicate
        ]

        wave = WaveExecutionResult(wave_number=0, results=results)

        assert sorted(wave.all_files_created) == ["a.py", "b.py", "c.py"]


# =============================================================================
# DRY RUN EXECUTOR TESTS
# =============================================================================


class TestDryRunExecutor:
    """Tests for DryRunExecutor."""

    @pytest.mark.asyncio
    async def test_execute_task(self, sample_task_spec: TaskSpec, sample_state: dict) -> None:
        """Test dry run task execution."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=1.0)

        result = await executor.execute_task(sample_task_spec, sample_state)

        assert result.task_id == "task-1"
        assert result.success is True
        assert result.output is not None
        assert "user model" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_wave(self, sample_state: dict) -> None:
        """Test dry run wave execution."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=1.0)

        result = await executor.execute_wave(
            task_ids=["task-1"],
            state=sample_state,
            wave_number=0,
        )

        assert result.wave_number == 0
        assert len(result.results) == 1
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_execute_with_failures(self, sample_state: dict) -> None:
        """Test dry run with simulated failures."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=0.0)

        result = await executor.execute_wave(
            task_ids=["task-1"],
            state=sample_state,
            wave_number=0,
        )

        assert result.success_rate == 0.0
        assert len(result.failed_tasks) == 1


# =============================================================================
# PARALLEL EXECUTOR TESTS
# =============================================================================


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    def test_init(self) -> None:
        """Test executor initialization."""
        executor = ParallelExecutor(max_concurrent=10, timeout=300)

        assert executor.max_concurrent == 10
        assert executor.timeout == 300

    def test_add_callback(self) -> None:
        """Test adding callbacks."""
        executor = ParallelExecutor()
        callback = MagicMock()

        executor.add_callback(callback)

        assert callback in executor._callbacks

    def test_remove_callback(self) -> None:
        """Test removing callbacks."""
        executor = ParallelExecutor()
        callback = MagicMock()

        executor.add_callback(callback)
        executor.remove_callback(callback)

        assert callback not in executor._callbacks

    @pytest.mark.asyncio
    async def test_execute_wave_empty(self, sample_state: dict) -> None:
        """Test executing an empty wave."""
        executor = ParallelExecutor()

        # Empty task list
        sample_state["tasks"] = []

        result = await executor.execute_wave(
            task_ids=["nonexistent"],
            state=sample_state,
            wave_number=0,
        )

        assert result.wave_number == 0
        assert len(result.results) == 0

    @pytest.mark.asyncio
    async def test_execute_task_simulation(
        self,
        sample_task_spec: TaskSpec,
        sample_state: dict,
        prompt_context: PromptContext,
    ) -> None:
        """Test task execution in simulation mode."""
        executor = ParallelExecutor(use_terminal_manager=False)

        # Mock the router to simulate execution
        with patch.object(executor, "_execute_with_router") as mock_execute:
            mock_execute.return_value = TaskExecutionResult(
                task_id="task-1",
                success=True,
                output="Simulated",
            )

            result = await executor.execute_task(
                sample_task_spec,
                sample_state,
                prompt_context,
            )

            assert result.task_id == "task-1"

    @pytest.mark.asyncio
    async def test_get_tasks_by_ids(self, sample_state: dict) -> None:
        """Test retrieving tasks by IDs."""
        executor = ParallelExecutor()

        tasks = executor._get_tasks_by_ids(["task-1", "task-2"], sample_state)

        assert len(tasks) == 2
        assert tasks[0].id == "task-1"
        assert tasks[1].id == "task-2"


# =============================================================================
# SEQUENTIAL EXECUTOR TESTS
# =============================================================================


class TestSequentialExecutor:
    """Tests for SequentialExecutor."""

    def test_init(self) -> None:
        """Test sequential executor initialization."""
        executor = SequentialExecutor(timeout=300)

        assert executor.timeout == 300
        assert executor._parallel.max_concurrent == 1

    @pytest.mark.asyncio
    async def test_execute_wave(self, sample_state: dict) -> None:
        """Test sequential wave execution."""
        executor = SequentialExecutor()

        # Mock internal execution
        with patch.object(executor._parallel, "_simulate_execution") as mock:
            mock.return_value = TaskExecutionResult(
                task_id="task-1",
                success=True,
            )

            result = await executor.execute_wave(
                task_ids=["task-1"],
                state=sample_state,
                wave_number=0,
            )

            assert result.wave_number == 0


# =============================================================================
# RETRY EXECUTOR TESTS
# =============================================================================


class TestRetryExecutor:
    """Tests for RetryExecutor."""

    def test_init(self) -> None:
        """Test retry executor initialization."""
        executor = RetryExecutor(max_retries=3, retry_delay=0.5)

        assert executor.max_retries == 3
        assert executor.retry_delay == 0.5

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, sample_state: dict) -> None:
        """Test that failed tasks are retried."""
        base_executor = DryRunExecutor(delay_per_task=0.01)

        # First call fails, second succeeds
        call_count = 0

        async def mock_execute_wave(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return WaveExecutionResult(
                    wave_number=0,
                    results=[TaskExecutionResult(task_id="task-1", success=False, error="Failed")],
                )
            return WaveExecutionResult(
                wave_number=0,
                results=[TaskExecutionResult(task_id="task-1", success=True)],
            )

        base_executor.execute_wave = mock_execute_wave

        executor = RetryExecutor(executor=base_executor, max_retries=2, retry_delay=0.01)

        result = await executor.execute_wave(
            task_ids=["task-1"],
            state=sample_state,
            wave_number=0,
        )

        assert call_count == 2
        assert result.results[0].success is True


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestCreateExecutor:
    """Tests for create_executor factory function."""

    def test_create_parallel(self) -> None:
        """Test creating parallel executor."""
        executor = create_executor("parallel", max_concurrent=10)

        assert isinstance(executor, ParallelExecutor)
        assert executor.max_concurrent == 10

    def test_create_sequential(self) -> None:
        """Test creating sequential executor."""
        executor = create_executor("sequential", timeout=300)

        assert isinstance(executor, SequentialExecutor)
        assert executor.timeout == 300

    def test_create_retry(self) -> None:
        """Test creating retry executor."""
        executor = create_executor("retry", max_retries=5)

        assert isinstance(executor, RetryExecutor)
        assert executor.max_retries == 5

    def test_create_dry_run(self) -> None:
        """Test creating dry run executor."""
        executor = create_executor("dry_run")

        assert isinstance(executor, DryRunExecutor)

    def test_create_default(self) -> None:
        """Test default executor creation."""
        executor = create_executor()

        assert isinstance(executor, ParallelExecutor)


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================


class TestExecutorIntegration:
    """Integration-style tests for executor."""

    @pytest.mark.asyncio
    async def test_full_wave_execution_dry_run(
        self,
        sample_state: dict,
        prompt_context: PromptContext,
    ) -> None:
        """Test full wave execution with dry run."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=1.0)

        result = await executor.execute_wave(
            task_ids=["task-1", "task-2"],
            state=sample_state,
            wave_number=0,
            context=prompt_context,
        )

        assert result.wave_number == 0
        assert len(result.completed_tasks) == 2
        assert result.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_multiple_waves_dry_run(
        self,
        sample_tasks: list[TaskSpec],
        sample_state: dict,
        sample_dependency_graph: DependencyGraph,
    ) -> None:
        """Test executing multiple waves."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=1.0)

        results = await executor.execute_graph(
            graph=sample_dependency_graph,
            state=sample_state,
        )

        assert len(results) == 3  # 3 waves
        assert all(r.success_rate == 1.0 for r in results)
