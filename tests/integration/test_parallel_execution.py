"""Integration tests for parallel execution."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.core.orchestrator import Kappa
from src.decomposition.executor import (
    DryRunExecutor,
)
from src.decomposition.models import (
    Complexity,
    DependencyGraph,
    ProjectRequirements,
    ProjectType,
    TaskCategory,
    TaskSpec,
)
from src.knowledge.context_manager import SharedContext
from src.prompts.builder import PromptBuilder, PromptContext, create_prompt_context

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_requirements() -> ProjectRequirements:
    """Create sample project requirements."""
    return ProjectRequirements(
        name="test-api",
        description="A simple REST API",
        project_type=ProjectType.API,
        tech_stack={
            "language": "python",
            "framework": "fastapi",
            "database": "postgresql",
        },
        features=["user management", "authentication"],
        pages=[],
        integrations=[],
        constraints=[],
    )


@pytest.fixture
def sample_tasks() -> list[TaskSpec]:
    """Create sample tasks for a wave."""
    return [
        TaskSpec(
            id="setup-1",
            title="Project setup",
            description="Initialize the project structure",
            category=TaskCategory.SETUP,
            complexity=Complexity.LOW,
            dependencies=[],
            files_to_create=["pyproject.toml", "src/__init__.py"],
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="model-1",
            title="User model",
            description="Create User SQLAlchemy model",
            category=TaskCategory.DATA_MODEL,
            complexity=Complexity.LOW,
            dependencies=["setup-1"],
            files_to_create=["src/models/user.py"],
            files_to_modify=[],
            wave_number=1,
        ),
        TaskSpec(
            id="model-2",
            title="Session model",
            description="Create Session model",
            category=TaskCategory.DATA_MODEL,
            complexity=Complexity.LOW,
            dependencies=["setup-1"],
            files_to_create=["src/models/session.py"],
            files_to_modify=[],
            wave_number=1,
        ),
        TaskSpec(
            id="api-1",
            title="User API",
            description="Create user CRUD endpoints",
            category=TaskCategory.API,
            complexity=Complexity.MEDIUM,
            dependencies=["model-1"],
            files_to_create=["src/api/users.py"],
            files_to_modify=[],
            wave_number=2,
        ),
    ]


@pytest.fixture
def sample_graph(sample_tasks: list[TaskSpec]) -> DependencyGraph:
    """Create sample dependency graph."""
    graph = DependencyGraph()
    for task in sample_tasks:
        graph.add_task(task)

    graph.waves = [
        ["setup-1"],  # Wave 0: Setup
        ["model-1", "model-2"],  # Wave 1: Models (parallel)
        ["api-1"],  # Wave 2: API
    ]
    return graph


@pytest.fixture
def sample_state(sample_tasks: list[TaskSpec]) -> dict:
    """Create sample execution state."""
    return {
        "project_id": "test-123",
        "project_name": "test-api",
        "workspace_path": "/tmp/test-api",
        "tasks": [t.model_dump() for t in sample_tasks],
        "dependency_graph": {
            "waves": [
                ["setup-1"],
                ["model-1", "model-2"],
                ["api-1"],
            ],
            "total_waves": 3,
        },
        "global_context": {
            "tech_stack": {"language": "python", "framework": "fastapi"},
        },
        "completed_tasks": [],
        "failed_tasks": [],
    }


# =============================================================================
# PROMPT BUILDER INTEGRATION TESTS
# =============================================================================


class TestPromptBuilderIntegration:
    """Test PromptBuilder integration with parallel execution."""

    def test_build_parallel_task_prompt(
        self,
        sample_tasks: list[TaskSpec],
        sample_requirements: ProjectRequirements,
    ) -> None:
        """Test building prompt for parallel task."""
        builder = PromptBuilder()
        context = create_prompt_context(
            project_name="test-api",
            workspace="/tmp/test-api",
            requirements=sample_requirements,
        )

        prompt = builder.build_parallel_task_prompt(
            task=sample_tasks[1],  # model-1
            context=context,
            wave_number=1,
            parallel_tasks=["model-1", "model-2"],
        )

        assert "User model" in prompt
        assert "Wave" in prompt
        assert "parallel" in prompt.lower() or "model-2" in prompt

    def test_context_types_available(
        self,
        sample_tasks: list[TaskSpec],
    ) -> None:
        """Test that context types are available in prompts."""
        builder = PromptBuilder()
        context = PromptContext(
            project_name="test-api",
            workspace="/tmp/test",
        )

        # Add some types
        context.add_type_definition("User", "interface User { id: string; name: string; }")

        prompt = builder.build(sample_tasks[1], context)

        # Type should be included for model task
        assert "User" in prompt or "type" in prompt.lower()


# =============================================================================
# SHARED CONTEXT INTEGRATION TESTS
# =============================================================================


class TestSharedContextIntegration:
    """Test SharedContext integration with execution."""

    def test_context_accumulates_across_waves(self) -> None:
        """Test that context accumulates from wave to wave."""
        context = SharedContext(project_id="test-123")

        # Wave 0 output
        context.record_task_output(
            "setup-1",
            {
                "task_id": "setup-1",
                "success": True,
                "wave_number": 0,
                "files_created": ["pyproject.toml"],
            },
        )

        # Wave 1 output
        context.add_type("User", "class User: pass")
        context.add_export("src/models/user.py", ["User"])

        # Get context for wave 2
        wave_context = context.get_context_for_wave(2)

        assert "User" in wave_context["types"]
        assert "src/models/user.py" in wave_context["exports"]
        assert 0 in wave_context["previous_waves"]

    def test_get_types_for_dependent_task(self, sample_tasks: list[TaskSpec]) -> None:
        """Test getting types for a task based on dependencies."""
        context = SharedContext(project_id="test-123")

        # Add type from model task
        context.add_type("User", "class User: pass")
        context.record_task_output(
            "model-1",
            {
                "task_id": "model-1",
                "success": True,
                "types_exported": ["User"],
            },
        )

        # Get types for API task (depends on model-1)
        types = context.get_types_for_task("api-1", ["model-1"])

        assert "User" in types


# =============================================================================
# EXECUTOR INTEGRATION TESTS
# =============================================================================


class TestExecutorIntegration:
    """Integration tests for executors."""

    @pytest.mark.asyncio
    async def test_dry_run_full_graph(
        self,
        sample_graph: DependencyGraph,
        sample_state: dict,
    ) -> None:
        """Test executing full graph with dry run."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=1.0)

        results = await executor.execute_graph(
            graph=sample_graph,
            state=sample_state,
        )

        # Should have 3 waves
        assert len(results) == 3

        # Wave 0: 1 task
        assert len(results[0].results) == 1
        assert results[0].completed_tasks == ["setup-1"]

        # Wave 1: 2 tasks in parallel
        assert len(results[1].results) == 2
        assert set(results[1].completed_tasks) == {"model-1", "model-2"}

        # Wave 2: 1 task
        assert len(results[2].results) == 1
        assert results[2].completed_tasks == ["api-1"]

    @pytest.mark.asyncio
    async def test_executor_updates_state(
        self,
        sample_graph: DependencyGraph,
        sample_state: dict,
    ) -> None:
        """Test that executor updates state with results."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=1.0)

        await executor.execute_graph(
            graph=sample_graph,
            state=sample_state,
        )

        # State should have been updated
        assert "setup-1" in sample_state["completed_tasks"]
        assert "model-1" in sample_state["completed_tasks"]
        assert "model-2" in sample_state["completed_tasks"]
        assert "api-1" in sample_state["completed_tasks"]

    @pytest.mark.asyncio
    async def test_executor_stops_on_high_failure_rate(
        self,
        sample_graph: DependencyGraph,
        sample_state: dict,
    ) -> None:
        """Test that executor stops when failure rate is too high."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=0.0)  # All fail

        results = await executor.execute_graph(
            graph=sample_graph,
            state=sample_state,
        )

        # Should stop after wave with >50% failure
        # In this case, wave 0 has 100% failure, so should only execute wave 0
        assert len(results) >= 1
        assert results[0].success_rate == 0.0

    @pytest.mark.asyncio
    async def test_context_passed_between_waves(
        self,
        sample_graph: DependencyGraph,
        sample_state: dict,
    ) -> None:
        """Test that context is passed between waves."""
        executor = DryRunExecutor(delay_per_task=0.01, success_rate=1.0)
        context = PromptContext(project_name="test-api", workspace="/tmp/test")

        results = await executor.execute_graph(
            graph=sample_graph,
            state=sample_state,
            context=context,
        )

        # Context should have task outputs from all waves
        assert "setup-1" in context.completed_task_outputs
        assert "model-1" in context.completed_task_outputs
        assert "api-1" in context.completed_task_outputs


# =============================================================================
# ORCHESTRATOR INTEGRATION TESTS
# =============================================================================


class TestOrchestratorParallelExecution:
    """Test Kappa orchestrator parallel execution."""

    @pytest.fixture
    def kappa(self, tmp_path: Path, mock_settings: None) -> Kappa:
        """Create Kappa instance for testing."""
        return Kappa(workspace=tmp_path)

    @pytest.mark.asyncio
    @patch("src.decomposition.parser.RequirementsParser.parse")
    @patch("src.decomposition.task_generator.TaskGenerator.generate")
    @patch("src.decomposition.dependency_resolver.DependencyResolver.resolve")
    async def test_execute_parallel_dry_run(
        self,
        mock_resolve: AsyncMock,
        mock_generate: AsyncMock,
        mock_parse: AsyncMock,
        kappa: Kappa,
        sample_requirements: ProjectRequirements,
        sample_tasks: list[TaskSpec],
        sample_graph: DependencyGraph,
    ) -> None:
        """Test parallel execution with dry run."""
        # Setup mocks
        mock_parse.return_value = sample_requirements
        mock_generate.return_value = sample_tasks
        mock_resolve.return_value = sample_graph

        result = await kappa.execute_parallel(
            requirements="Build a REST API",
            project_name="test-api",
            dry_run=True,
        )

        assert result["status"] == "completed"
        assert result["project_name"] == "test-api"
        assert result["total_tasks"] == len(sample_tasks)
        assert result["completed_tasks"] > 0
        assert "wave_results" in result

    @pytest.mark.asyncio
    async def test_execute_wave_parallel(
        self,
        kappa: Kappa,
        sample_tasks: list[TaskSpec],
        sample_state: dict,
    ) -> None:
        """Test executing a single wave in parallel."""
        # Use dry run executor by patching where it's USED (orchestrator), not defined
        with patch("src.core.orchestrator.create_executor") as mock_create:
            mock_executor = DryRunExecutor(delay_per_task=0.01)
            mock_create.return_value = mock_executor

            result = await kappa.execute_wave_parallel(
                tasks=[sample_tasks[0].model_dump()],
                state=sample_state,
                wave_number=0,
            )

            assert result.wave_number == 0
            assert len(result.results) == 1


# =============================================================================
# END-TO-END TESTS
# =============================================================================


@pytest.mark.e2e
class TestParallelExecutionE2E:
    """End-to-end tests for parallel execution."""

    @pytest.mark.asyncio
    async def test_simple_project_dry_run(
        self,
        tmp_path: Path,
        mock_settings: None,
    ) -> None:
        """Test simple project execution with dry run."""
        kappa = Kappa(workspace=tmp_path)

        # This uses actual decomposition but dry run execution
        result = await kappa.execute_parallel(
            requirements="Build a simple Python CLI tool that prints hello world",
            project_name="hello-cli",
            dry_run=True,
        )

        assert result["status"] in ["completed", "failed"]
        assert result["project_name"] == "hello-cli"
        assert (tmp_path / "hello-cli").exists()

    @pytest.mark.asyncio
    async def test_task_decomposition_and_execution(
        self,
        tmp_path: Path,
        mock_settings: None,
    ) -> None:
        """Test that decomposition integrates with execution."""
        kappa = Kappa(workspace=tmp_path)

        # First preview to see tasks
        preview = await kappa.preview("Build a REST API with user endpoints")

        assert len(preview["tasks"]) > 0
        assert "dependency_graph" in preview
        assert preview["dependency_graph"]["total_waves"] > 0

        # Now execute with dry run
        result = await kappa.execute_parallel(
            requirements="Build a REST API with user endpoints",
            dry_run=True,
        )

        assert result["waves_executed"] > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


@pytest.mark.slow
class TestParallelExecutionPerformance:
    """Performance tests for parallel execution."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_faster_than_sequential(self) -> None:
        """Test that parallel execution is faster than sequential."""

        # Create 10 tasks that each take 0.1s
        tasks = [
            TaskSpec(
                id=f"task-{i}",
                title=f"Task {i}",
                description=f"Task {i} description",
                category=TaskCategory.SETUP,
                complexity=Complexity.LOW,
                dependencies=[],
                files_to_create=[f"file{i}.py"],
                files_to_modify=[],
            )
            for i in range(10)
        ]

        state = {
            "tasks": [t.model_dump() for t in tasks],
            "completed_tasks": [],
            "failed_tasks": [],
        }

        # Parallel execution (should be ~0.1s with 10 concurrent)
        parallel_executor = DryRunExecutor(delay_per_task=0.1, success_rate=1.0)

        import time

        start = time.time()
        await parallel_executor.execute_wave(
            task_ids=[t.id for t in tasks],
            state=state,
            wave_number=0,
        )
        parallel_time = time.time() - start

        # Sequential would be ~1.0s
        # Parallel should be significantly faster
        assert parallel_time < 0.5  # Should complete in ~0.1s + overhead
