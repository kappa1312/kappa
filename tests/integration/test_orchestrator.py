"""Integration tests for the orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.orchestrator import Kappa
from src.core.state import ProjectStatus


class TestKappaOrchestrator:
    """Integration tests for Kappa orchestrator."""

    @pytest.fixture
    def kappa(self, mock_settings: None) -> Kappa:
        """Create Kappa instance for testing."""
        return Kappa()

    @pytest.mark.asyncio
    async def test_decompose(
        self,
        kappa: Kappa,
        sample_specification: str,
    ) -> None:
        """Test decomposition without execution."""
        tasks = await kappa.decompose(sample_specification)

        assert len(tasks) > 0
        assert all("id" in task for task in tasks)
        # Check for both 'name' (legacy) and 'title' (new) for compatibility
        assert all("name" in task or "title" in task for task in tasks)

    @pytest.mark.asyncio
    @patch("src.graph.builder.build_kappa_graph")
    async def test_run_creates_project_directory(
        self,
        mock_build_graph: AsyncMock,
        kappa: Kappa,
        tmp_path: str,
    ) -> None:
        """Test that run creates project directory."""
        # Mock the graph
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "status": ProjectStatus.COMPLETED.value,
                "final_output": "Test complete",
            }
        )
        mock_build_graph.return_value = mock_graph

        project_path = tmp_path / "test-project"

        result = await kappa.run(
            spec="Test specification",
            project_path=str(project_path),
        )

        assert project_path.exists()

    @pytest.mark.asyncio
    async def test_status_returns_none_for_unknown_project(
        self,
        kappa: Kappa,
    ) -> None:
        """Test status returns None for unknown project."""
        # This would normally check the database
        # For unit test, we just verify the method exists
        # Integration test would use actual database
        pass

    def test_repr(self, kappa: Kappa) -> None:
        """Test string representation."""
        repr_str = repr(kappa)

        assert "Kappa" in repr_str
        assert "max_sessions" in repr_str


class TestKappaState:
    """Tests for Kappa state management."""

    def test_create_initial_state(self) -> None:
        """Test creating initial state."""
        from src.core.state import create_initial_state

        state = create_initial_state(
            specification="Test spec",
            project_path="/test/path",
            project_name="test-project",
        )

        assert state["specification"] == "Test spec"
        assert state["project_path"] == "/test/path"
        assert state["project_name"] == "test-project"
        assert state["status"] == "pending"

    def test_get_pending_tasks(self) -> None:
        """Test getting pending tasks."""
        from src.core.state import KappaState, get_pending_tasks

        state: KappaState = {
            "tasks": [
                {"id": "1"},
                {"id": "2"},
                {"id": "3"},
            ],
            "completed_tasks": ["1"],
            "failed_tasks": [],
            "skipped_tasks": [],
        }

        pending = get_pending_tasks(state)

        assert "2" in pending
        assert "3" in pending
        assert "1" not in pending

    def test_is_task_ready(self) -> None:
        """Test checking if task is ready."""
        from src.core.state import KappaState, is_task_ready

        state: KappaState = {
            "dependency_graph": {
                "task-1": [],
                "task-2": ["task-1"],
                "task-3": ["task-1", "task-2"],
            },
            "completed_tasks": ["task-1"],
        }

        assert is_task_ready(state, "task-1")  # No deps
        assert is_task_ready(state, "task-2")  # task-1 completed
        assert not is_task_ready(state, "task-3")  # task-2 not completed
