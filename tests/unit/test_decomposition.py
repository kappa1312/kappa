"""Unit tests for decomposition module."""

import pytest

from src.decomposition.dependency_resolver import DependencyResolver
from src.decomposition.models import Complexity, Task, TaskCategory
from src.decomposition.parser import SpecificationParser
from src.decomposition.task_generator import TaskGenerator


class TestSpecificationParser:
    """Tests for SpecificationParser."""

    def test_parse_simple_spec(self) -> None:
        """Test parsing a simple specification."""
        parser = SpecificationParser()
        spec = "Build a REST API with user authentication"

        requirements = parser.parse(spec)

        assert len(requirements) > 0
        assert any("api" in r.text.lower() for r in requirements)

    def test_parse_multi_feature_spec(self) -> None:
        """Test parsing a specification with multiple features."""
        parser = SpecificationParser()
        spec = """
        Build an e-commerce platform with:
        - User registration and login
        - Product catalog
        - Shopping cart
        - Checkout with Stripe
        """

        requirements = parser.parse(spec)

        assert len(requirements) >= 4

    def test_category_inference(self) -> None:
        """Test that categories are correctly inferred."""
        parser = SpecificationParser()

        api_spec = "Create REST API endpoints"
        api_reqs = parser.parse(api_spec)
        assert any(r.category == TaskCategory.API for r in api_reqs)

        test_spec = "Write unit tests"
        test_reqs = parser.parse(test_spec)
        assert any(r.category == TaskCategory.TESTING for r in test_reqs)

    def test_entity_extraction(self) -> None:
        """Test entity extraction from specification."""
        parser = SpecificationParser()
        spec = "Create User model and Product model with CRUD operations"

        requirements = parser.parse(spec)
        all_entities = []
        for req in requirements:
            all_entities.extend(req.entities)

        assert "user" in all_entities or "product" in all_entities


class TestDependencyResolver:
    """Tests for DependencyResolver."""

    def test_build_graph(self, sample_tasks: list) -> None:
        """Test building dependency graph."""
        resolver = DependencyResolver()
        graph = resolver.build_graph(sample_tasks)

        assert "task-1" in graph
        assert "task-5" in graph
        # task-5 directly depends on task-2, task-3, task-4 (not task-1)
        assert "task-2" in graph["task-5"]
        assert "task-3" in graph["task-5"]
        assert "task-4" in graph["task-5"]

    def test_assign_waves(self, sample_tasks: list) -> None:
        """Test wave assignment."""
        resolver = DependencyResolver()
        graph = resolver.build_graph(sample_tasks)
        waves = resolver.assign_waves(sample_tasks, graph)

        assert len(waves) > 0
        # Task-1 should be in first wave (no dependencies)
        assert "task-1" in waves[0]
        # Task-5 should be in a later wave (has dependencies)
        task_5_wave = next(i for i, w in enumerate(waves) if "task-5" in w)
        assert task_5_wave > 0

    def test_detect_cycles(self) -> None:
        """Test cycle detection."""
        resolver = DependencyResolver()

        # Create cyclic tasks
        cyclic_tasks = [
            Task(id="a", name="A", description="A", dependencies=["c"]),
            Task(id="b", name="B", description="B", dependencies=["a"]),
            Task(id="c", name="C", description="C", dependencies=["b"]),
        ]

        graph = resolver.build_graph(cyclic_tasks)
        cycles = resolver.detect_cycles(graph)

        assert cycles is not None
        assert len(cycles) > 0

    def test_topological_sort(self, sample_tasks: list) -> None:
        """Test topological sorting."""
        resolver = DependencyResolver()
        graph = resolver.build_graph(sample_tasks)

        sorted_tasks = resolver.topological_sort(sample_tasks, graph)

        # task-1 should come before task-2 (task-2 depends on task-1)
        assert sorted_tasks.index("task-1") < sorted_tasks.index("task-2")

    def test_critical_path(self, sample_tasks: list) -> None:
        """Test critical path calculation."""
        resolver = DependencyResolver()
        graph = resolver.build_graph(sample_tasks)

        critical_path = resolver.get_critical_path(sample_tasks, graph)

        assert len(critical_path) > 0
        assert "task-1" in critical_path


class TestTaskGenerator:
    """Tests for TaskGenerator."""

    @pytest.mark.asyncio
    async def test_generate_tasks(self, sample_specification: str) -> None:
        """Test task generation from specification."""
        generator = TaskGenerator()
        tasks = await generator.generate(sample_specification)

        assert len(tasks) > 0
        # Should include setup task
        assert any(t.category == TaskCategory.SETUP for t in tasks)

    @pytest.mark.asyncio
    async def test_generate_includes_models(self) -> None:
        """Test that model tasks are generated."""
        generator = TaskGenerator()
        spec = "Build a blog with User and Post models"

        tasks = await generator.generate(spec)

        model_tasks = [t for t in tasks if t.category == TaskCategory.DATA_MODEL]
        assert len(model_tasks) > 0

    @pytest.mark.asyncio
    async def test_generate_waves_assigned(self) -> None:
        """Test that waves are assigned to generated tasks."""
        generator = TaskGenerator()
        spec = "Build a simple CRUD API"

        tasks = await generator.generate(spec)

        # Should have tasks in multiple waves
        waves = set(t.wave for t in tasks)
        assert len(waves) > 1


class TestTaskModel:
    """Tests for Task model."""

    def test_task_is_ready(self) -> None:
        """Test is_ready method."""
        task = Task(
            id="test",
            name="Test",
            description="Test task",
            dependencies=["dep-1", "dep-2"],
        )

        assert not task.is_ready(set())
        assert not task.is_ready({"dep-1"})
        assert task.is_ready({"dep-1", "dep-2"})
        assert task.is_ready({"dep-1", "dep-2", "dep-3"})

    def test_task_to_prompt(self) -> None:
        """Test prompt generation."""
        task = Task(
            id="test",
            name="Create User Model",
            description="Define the User SQLAlchemy model",
            file_targets=["src/models/user.py"],
            tags=["model", "user"],
        )

        prompt = task.to_prompt()

        assert "Create User Model" in prompt
        assert "src/models/user.py" in prompt
        assert "model" in prompt
