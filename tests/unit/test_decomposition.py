"""Unit tests for decomposition module."""

import pytest

from src.decomposition.dependency_resolver import DependencyResolver
from src.decomposition.models import (
    DependencyGraph,
    FileConflict,
    ProjectRequirements,
    ProjectType,
    SessionType,
    Task,
    TaskCategory,
    TaskSpec,
)
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
        # TaskSpec uses wave_number, legacy Task uses wave
        waves = set(getattr(t, "wave_number", getattr(t, "wave", 0)) for t in tasks)
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


class TestTaskSpec:
    """Tests for TaskSpec model."""

    def test_task_spec_creation(self) -> None:
        """Test creating a TaskSpec."""
        task = TaskSpec(
            id="task-1",
            title="Create User Model",
            description="Define the User model",
            session_type=SessionType.TERMINAL,
            dependencies=[],
            files_to_create=["src/models/user.py"],
            files_to_modify=[],
        )

        assert task.id == "task-1"
        assert task.title == "Create User Model"
        assert task.session_type == SessionType.TERMINAL
        assert 0 <= task.priority <= 10  # Priority is in valid range
        assert task.estimated_duration_minutes == 30

    def test_task_spec_with_dependencies(self) -> None:
        """Test TaskSpec with dependencies."""
        task = TaskSpec(
            id="task-2",
            title="Create Routes",
            description="Create API routes",
            session_type=SessionType.TERMINAL,
            dependencies=["task-1"],
            files_to_create=["src/routes/user.py"],
            files_to_modify=[],
            requires_context_from=["task-1"],
        )

        assert "task-1" in task.dependencies
        assert "task-1" in task.requires_context_from

    def test_task_spec_validation(self) -> None:
        """Test TaskSpec validation."""
        task = TaskSpec(
            id="task-1",
            title="Test",
            description="Test task",
            session_type=SessionType.TERMINAL,
            dependencies=[],
            files_to_create=[],
            files_to_modify=[],
            priority=5,
            estimated_duration_minutes=60,
        )

        assert 0 <= task.priority <= 10
        assert task.estimated_duration_minutes > 0

    def test_task_spec_serialization(self) -> None:
        """Test TaskSpec serialization."""
        task = TaskSpec(
            id="task-1",
            title="Test Task",
            description="Test description",
            session_type=SessionType.WEB,
            dependencies=["dep-1"],
            files_to_create=["file.py"],
            files_to_modify=["other.py"],
        )

        data = task.model_dump()

        assert data["id"] == "task-1"
        assert data["title"] == "Test Task"
        assert data["session_type"] == "web"


class TestProjectRequirements:
    """Tests for ProjectRequirements model."""

    def test_project_requirements_creation(self) -> None:
        """Test creating ProjectRequirements."""
        req = ProjectRequirements(
            name="test-project",
            description="A test project",
            project_type=ProjectType.API,
            tech_stack={"framework": "express", "language": "typescript"},
            features=["auth", "crud"],
            pages=[],
            integrations=[],
            constraints=[],
        )

        assert req.name == "test-project"
        assert req.project_type == ProjectType.API
        assert "framework" in req.tech_stack

    def test_project_requirements_defaults(self) -> None:
        """Test ProjectRequirements defaults."""
        req = ProjectRequirements(
            name="test",
            description="Test",
            project_type=ProjectType.WEBSITE,
            tech_stack={},
            features=[],
            pages=[],
            integrations=[],
            constraints=[],
        )

        assert req.priority == "normal"
        assert req.timeline is None

    def test_project_requirements_serialization(self) -> None:
        """Test ProjectRequirements serialization."""
        req = ProjectRequirements(
            name="test",
            description="Test",
            project_type=ProjectType.DASHBOARD,
            tech_stack={"framework": "react"},
            features=["analytics"],
            pages=["overview", "settings"],
            integrations=["stripe"],
            constraints=["mobile-first"],
        )

        data = req.model_dump()

        assert data["project_type"] == "dashboard"
        assert len(data["pages"]) == 2


class TestDependencyGraph:
    """Tests for DependencyGraph model."""

    def test_dependency_graph_creation(self) -> None:
        """Test creating a DependencyGraph."""
        graph = DependencyGraph(
            nodes={},
            edges={},
            waves=[],
        )

        assert graph.total_waves == 0
        assert len(graph.nodes) == 0

    def test_add_task_to_graph(self) -> None:
        """Test adding a task to the graph."""
        graph = DependencyGraph(nodes={}, edges={}, waves=[])

        task = TaskSpec(
            id="task-1",
            title="Test",
            description="Test",
            session_type=SessionType.TERMINAL,
            dependencies=[],
            files_to_create=[],
            files_to_modify=[],
        )

        graph.add_task(task)

        assert "task-1" in graph.nodes
        assert graph.get_task("task-1") == task

    def test_get_dependents(self) -> None:
        """Test getting task dependents."""
        graph = DependencyGraph(
            nodes={},
            edges={"task-1": [], "task-2": ["task-1"], "task-3": ["task-1"]},
            waves=[],
        )

        dependents = graph.get_dependents("task-1")

        assert "task-2" in dependents
        assert "task-3" in dependents

    def test_is_ready(self) -> None:
        """Test checking if task is ready."""
        graph = DependencyGraph(
            nodes={},
            edges={"task-1": [], "task-2": ["task-1"]},
            waves=[],
        )

        # task-1 has no dependencies, should be ready
        assert graph.is_ready("task-1", set())

        # task-2 depends on task-1, should not be ready
        assert not graph.is_ready("task-2", set())

        # task-2 should be ready after task-1 completes
        assert graph.is_ready("task-2", {"task-1"})

    def test_total_waves_property(self) -> None:
        """Test total_waves property."""
        graph = DependencyGraph(
            nodes={},
            edges={},
            waves=[["task-1"], ["task-2", "task-3"], ["task-4"]],
        )

        assert graph.total_waves == 3


class TestFileConflict:
    """Tests for FileConflict model."""

    def test_file_conflict_creation(self) -> None:
        """Test creating a FileConflict."""
        conflict = FileConflict(
            file_path="src/models/user.py",
            task_ids=["task-1", "task-2"],
            wave_number=1,
            conflict_type="write",
        )

        assert conflict.file_path == "src/models/user.py"
        assert len(conflict.task_ids) == 2
        assert conflict.wave_number == 1

    def test_file_conflict_serialization(self) -> None:
        """Test FileConflict serialization."""
        conflict = FileConflict(
            file_path="test.py",
            task_ids=["a", "b"],
            wave_number=0,
            conflict_type="modify",
            description="Both tasks modify test.py",
        )

        data = conflict.model_dump()

        assert data["file_path"] == "test.py"
        assert data["description"] == "Both tasks modify test.py"


class TestProjectType:
    """Tests for ProjectType enum."""

    def test_project_types(self) -> None:
        """Test all project types exist."""
        assert ProjectType.WEBSITE.value == "website"
        assert ProjectType.API.value == "api"
        assert ProjectType.DASHBOARD.value == "dashboard"
        assert ProjectType.CLI_TOOL.value == "cli_tool"
        assert ProjectType.LIBRARY.value == "library"
        assert ProjectType.MOBILE_APP.value == "mobile_app"


class TestSessionType:
    """Tests for SessionType enum."""

    def test_session_types(self) -> None:
        """Test all session types exist."""
        assert SessionType.TERMINAL.value == "terminal"
        assert SessionType.WEB.value == "web"
        assert SessionType.NATIVE.value == "native"
        assert SessionType.EXTENSION.value == "extension"


class TestDependencyResolverWithTaskSpec:
    """Tests for DependencyResolver with TaskSpec objects."""

    def test_resolve_with_task_specs(self) -> None:
        """Test resolving dependencies with TaskSpec objects."""
        tasks = [
            TaskSpec(
                id="task-1",
                title="Setup",
                description="Setup project",
                session_type=SessionType.TERMINAL,
                dependencies=[],
                files_to_create=["package.json"],
                files_to_modify=[],
            ),
            TaskSpec(
                id="task-2",
                title="Models",
                description="Create models",
                session_type=SessionType.TERMINAL,
                dependencies=["task-1"],
                files_to_create=["src/models.py"],
                files_to_modify=[],
            ),
            TaskSpec(
                id="task-3",
                title="Routes",
                description="Create routes",
                session_type=SessionType.TERMINAL,
                dependencies=["task-1"],
                files_to_create=["src/routes.py"],
                files_to_modify=[],
            ),
            TaskSpec(
                id="task-4",
                title="Tests",
                description="Create tests",
                session_type=SessionType.TERMINAL,
                dependencies=["task-2", "task-3"],
                files_to_create=["tests/test_app.py"],
                files_to_modify=[],
            ),
        ]

        resolver = DependencyResolver(tasks)
        graph = resolver.resolve()

        assert isinstance(graph, DependencyGraph)
        assert graph.total_waves >= 2
        assert "task-1" in graph.waves[0]

    def test_detect_file_conflicts(self) -> None:
        """Test detecting file conflicts."""
        tasks = [
            TaskSpec(
                id="task-1",
                title="Task A",
                description="Task A",
                session_type=SessionType.TERMINAL,
                dependencies=[],
                files_to_create=["shared.py"],
                files_to_modify=[],
                wave_number=0,
            ),
            TaskSpec(
                id="task-2",
                title="Task B",
                description="Task B",
                session_type=SessionType.TERMINAL,
                dependencies=[],
                files_to_create=["shared.py"],  # Same file!
                files_to_modify=[],
                wave_number=0,
            ),
        ]

        resolver = DependencyResolver(tasks)
        resolver.resolve()
        conflicts = resolver.detect_conflicts()

        assert len(conflicts) > 0
        assert conflicts[0].file_path == "shared.py"

    def test_no_cycles_in_valid_graph(self) -> None:
        """Test that valid graphs have no cycles."""
        tasks = [
            TaskSpec(
                id="task-1",
                title="A",
                description="A",
                session_type=SessionType.TERMINAL,
                dependencies=[],
                files_to_create=[],
                files_to_modify=[],
            ),
            TaskSpec(
                id="task-2",
                title="B",
                description="B",
                session_type=SessionType.TERMINAL,
                dependencies=["task-1"],
                files_to_create=[],
                files_to_modify=[],
            ),
        ]

        resolver = DependencyResolver(tasks)
        # Should not raise
        graph = resolver.resolve()
        assert graph is not None
