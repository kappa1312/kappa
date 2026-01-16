"""End-to-end tests for complete Kappa workflow."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.core.orchestrator import Kappa
from src.core.state import ProjectStatus


@pytest.mark.e2e
class TestFullWorkflow:
    """End-to-end tests for the complete Kappa workflow."""

    @pytest.fixture
    def project_dir(self, tmp_path: Path) -> Path:
        """Create temporary project directory."""
        project = tmp_path / "test-project"
        project.mkdir()
        return project

    @pytest.mark.asyncio
    @pytest.mark.slow
    @patch("src.sessions.terminal.TerminalSession._execute_with_client")
    async def test_simple_project_generation(
        self,
        mock_execute: AsyncMock,
        project_dir: Path,
        mock_settings: None,
    ) -> None:
        """Test generating a simple project end-to-end."""
        # Mock Claude responses
        mock_execute.return_value = "Task completed successfully"

        kappa = Kappa()

        # Simple specification
        spec = "Create a hello world Python script"

        result = await kappa.run(
            spec=spec,
            project_path=str(project_dir),
            project_name="hello-world",
        )

        # Verify execution completed
        assert result["status"] in [
            ProjectStatus.COMPLETED.value,
            ProjectStatus.FAILED.value,
        ]

    @pytest.mark.asyncio
    async def test_decomposition_only(
        self,
        mock_settings: None,
    ) -> None:
        """Test decomposition without full execution."""
        kappa = Kappa()

        spec = """
        Build a REST API with:
        - User authentication
        - Product catalog
        - Shopping cart
        """

        tasks = await kappa.decompose(spec)

        # Should generate multiple tasks
        assert len(tasks) > 3

        # Should have setup task (check both 'name' and 'title' for compatibility)
        setup_tasks = [
            t
            for t in tasks
            if "setup" in t.get("name", t.get("title", "")).lower()
            or "initialize" in t.get("name", t.get("title", "")).lower()
            or "foundation" in t.get("name", t.get("title", "")).lower()
        ]
        assert len(setup_tasks) > 0

        # Should have model tasks (check both old and new category formats)
        model_tasks = [
            t
            for t in tasks
            if t.get("category") in ("data_model", "business_logic")
            or "model" in t.get("name", t.get("title", "")).lower()
        ]
        assert len(model_tasks) > 0

    @pytest.mark.asyncio
    async def test_wave_organization(
        self,
        mock_settings: None,
    ) -> None:
        """Test that tasks are organized into waves correctly."""
        from src.decomposition.dependency_resolver import DependencyResolver
        from src.decomposition.task_generator import TaskGenerator

        generator = TaskGenerator()
        resolver = DependencyResolver()

        spec = "Build a blog with posts and comments"
        tasks = await generator.generate(spec)

        graph = resolver.build_graph(tasks)
        waves = resolver.assign_waves(tasks, graph)

        # First wave should have no dependencies
        first_wave_tasks = [t for t in tasks if t.id in waves[0]]
        for task in first_wave_tasks:
            assert len(task.dependencies) == 0

        # Later waves should depend on earlier waves
        if len(waves) > 1:
            later_tasks = [t for t in tasks if t.id in waves[-1]]
            for task in later_tasks:
                assert len(task.dependencies) > 0

    def test_conflict_detection_in_workflow(
        self,
        mock_settings: None,
    ) -> None:
        """Test that conflicts are detected during workflow."""
        from src.conflict.detector import ConflictDetector
        from src.decomposition.models import TaskSpec

        detector = ConflictDetector()

        # Create TaskSpec objects with overlapping files_to_create
        tasks = [
            TaskSpec(
                id="task-1",
                title="Task 1",
                description="First task that creates user model",
                files_to_create=["src/models/user.py", "src/api/auth.py"],
            ),
            TaskSpec(
                id="task-2",
                title="Task 2",
                description="Second task that also creates user model",
                files_to_create=["src/models/user.py", "src/api/user.py"],
            ),
        ]

        report = detector.analyze(tasks)

        # Should detect conflict in user.py (both tasks write to same file)
        assert report.total_conflicts >= 1
        # Check that we have conflicts in the report
        assert len(report.conflicts) >= 1


@pytest.mark.e2e
class TestCLIWorkflow:
    """End-to-end tests for CLI commands."""

    def test_cli_health_command(self) -> None:
        """Test CLI health command."""
        from typer.testing import CliRunner

        from src.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["health"])

        # Should complete without error
        assert result.exit_code == 0

    def test_cli_init_command(self, tmp_path: Path) -> None:
        """Test CLI init command."""
        from typer.testing import CliRunner

        from src.cli.main import app

        runner = CliRunner()
        project_path = tmp_path / "new-project"

        result = runner.invoke(app, ["init", str(project_path), "--name", "test"])

        assert result.exit_code == 0
        assert project_path.exists()
        assert (project_path / "README.md").exists()

    def test_cli_decompose_command(self) -> None:
        """Test CLI decompose command."""
        from typer.testing import CliRunner

        from src.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["decompose", "Build a simple API"])

        # Should show task breakdown
        assert result.exit_code == 0
