"""Integration tests for LangGraph orchestration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.graph.state import (
    ExecutionStatus,
    KappaState,
    WaveStatus,
    calculate_progress,
    create_initial_state,
    get_pending_task_ids,
    get_state_summary,
    get_wave_task_ids,
    is_task_ready,
)
from src.graph.edges import (
    get_execution_progress,
    get_next_wave_tasks,
    get_ready_tasks,
    has_unresolved_conflicts,
    route_after_decomposition,
    route_after_dependency_resolution,
    route_after_merge,
    route_after_parsing,
    route_after_task_generation,
    route_after_validation,
    route_after_wave,
    should_abort_execution,
    should_continue_execution,
)
from src.graph.builder import (
    create_orchestration_graph,
    build_legacy_graph,
    get_graph_info,
    visualize_graph,
)


class TestKappaState:
    """Tests for KappaState and state utilities."""

    def test_create_initial_state(self):
        """Test creating initial state."""
        state = create_initial_state(
            requirements_text="Build a REST API",
            workspace_path="/tmp/test-project",
            project_name="test-api",
        )

        assert state["project_name"] == "test-api"
        assert state["requirements_text"] == "Build a REST API"
        assert state["workspace_path"] == "/tmp/test-project"
        assert state["status"] == ExecutionStatus.PENDING.value
        assert state["completed_tasks"] == []
        assert state["failed_tasks"] == []

    def test_create_initial_state_with_config(self):
        """Test creating state with custom config."""
        state = create_initial_state(
            requirements_text="Build API",
            workspace_path="/tmp/test",
            config={"max_parallel": 10},
        )

        assert state["config"] == {"max_parallel": 10}

    def test_get_pending_task_ids(self):
        """Test getting pending task IDs."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "tasks": [
                {"id": "task-1"},
                {"id": "task-2"},
                {"id": "task-3"},
            ],
            "completed_tasks": ["task-1"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        pending = get_pending_task_ids(state)

        assert "task-1" not in pending
        assert "task-2" in pending
        assert "task-3" in pending

    def test_is_task_ready(self):
        """Test checking if task is ready."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "dependency_graph": {
                "edges": {
                    "task-1": [],
                    "task-2": ["task-1"],
                    "task-3": ["task-1", "task-2"],
                }
            },
            "completed_tasks": ["task-1"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        # task-1 is complete, task-2 should be ready
        assert is_task_ready(state, "task-2")

        # task-3 depends on task-1 and task-2, task-2 not complete
        assert not is_task_ready(state, "task-3")

    def test_get_wave_task_ids(self):
        """Test getting task IDs for a wave."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "dependency_graph": {
                "waves": [["task-1"], ["task-2", "task-3"], ["task-4"]],
            },
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        wave_0 = get_wave_task_ids(state, 0)
        wave_1 = get_wave_task_ids(state, 1)
        wave_2 = get_wave_task_ids(state, 2)
        wave_3 = get_wave_task_ids(state, 3)

        assert wave_0 == ["task-1"]
        assert wave_1 == ["task-2", "task-3"]
        assert wave_2 == ["task-4"]
        assert wave_3 == []  # Out of bounds

    def test_calculate_progress(self):
        """Test progress calculation."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "tasks": [{"id": f"task-{i}"} for i in range(10)],
            "completed_tasks": ["task-0", "task-1", "task-2"],
            "failed_tasks": ["task-3"],
            "skipped_tasks": ["task-4"],
            "current_wave": 2,
            "total_waves": 5,
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        progress = calculate_progress(state)

        assert progress["total_tasks"] == 10
        assert progress["completed_tasks"] == 3
        assert progress["failed_tasks"] == 1
        assert progress["skipped_tasks"] == 1
        assert progress["pending_tasks"] == 5
        assert progress["progress_percent"] == 50.0
        assert progress["current_wave"] == 2
        assert progress["total_waves"] == 5

    def test_get_state_summary(self):
        """Test state summary generation."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test-project",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "tasks": [{"id": f"task-{i}"} for i in range(4)],
            "completed_tasks": ["task-0", "task-1"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "current_wave": 1,
            "total_waves": 3,
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        summary = get_state_summary(state)

        assert "test-project" in summary
        assert "executing" in summary.lower()
        assert "50" in summary  # 50% progress


class TestEdgeRouting:
    """Tests for edge routing functions."""

    def test_route_after_parsing_success(self):
        """Test routing after successful parsing."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.PARSING.value,
            "requirements": {"name": "test"},
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_parsing(state)
        assert result == "generate_tasks"

    def test_route_after_parsing_error(self):
        """Test routing after parsing error."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.FAILED.value,
            "error": "Parsing failed",
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_parsing(state)
        assert result == "handle_error"

    def test_route_after_task_generation_success(self):
        """Test routing after successful task generation."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.GENERATING_TASKS.value,
            "tasks": [{"id": "task-1"}],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_task_generation(state)
        assert result == "resolve_dependencies"

    def test_route_after_task_generation_no_tasks(self):
        """Test routing when no tasks generated."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.GENERATING_TASKS.value,
            "tasks": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_task_generation(state)
        assert result == "handle_error"

    def test_should_continue_execution_more_waves(self):
        """Test continue execution with more waves."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "current_wave": 1,
            "total_waves": 3,
            "completed_tasks": ["task-1"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = should_continue_execution(state)
        assert result == "execute_wave"

    def test_should_continue_execution_all_done(self):
        """Test continue execution when all waves done."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "current_wave": 3,
            "total_waves": 3,
            "completed_tasks": ["task-1", "task-2"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = should_continue_execution(state)
        assert result == "merge_outputs"

    def test_should_continue_execution_high_failure_rate(self):
        """Test continue execution with high failure rate."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "current_wave": 1,
            "total_waves": 3,
            "completed_tasks": ["task-1"],
            "failed_tasks": ["task-2", "task-3", "task-4"],  # 75% failure
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = should_continue_execution(state)
        assert result == "handle_error"

    def test_route_after_merge_success(self):
        """Test routing after successful merge."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.MERGING.value,
            "conflicts": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_merge(state)
        assert result == "validate"

    def test_route_after_merge_critical_conflicts(self):
        """Test routing with critical unresolved conflicts."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.MERGING.value,
            "conflicts": [
                {"resolved": False, "conflict_type": "critical"},
            ],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_merge(state)
        assert result == "handle_error"

    def test_route_after_validation_success(self):
        """Test routing after successful validation."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.COMPLETED.value,
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_validation(state)
        assert result == "end"

    def test_route_after_validation_failure(self):
        """Test routing after validation failure."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.VALIDATION_FAILED.value,
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_validation(state)
        assert result == "handle_error"


class TestEdgeHelpers:
    """Tests for edge helper functions."""

    def test_should_abort_execution_failed_status(self):
        """Test abort check with failed status."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.FAILED.value,
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        assert should_abort_execution(state) is True

    def test_should_abort_execution_high_failure(self):
        """Test abort check with high failure rate."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "completed_tasks": ["t1"],
            "failed_tasks": ["t2", "t3", "t4"],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        assert should_abort_execution(state) is True

    def test_get_ready_tasks(self):
        """Test getting ready tasks."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "current_wave": 1,
            "dependency_graph": {
                "waves": [["task-1"], ["task-2", "task-3"]],
                "edges": {
                    "task-2": ["task-1"],
                    "task-3": ["task-1"],
                },
            },
            "completed_tasks": ["task-1"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        ready = get_ready_tasks(state)

        assert "task-2" in ready
        assert "task-3" in ready

    def test_get_next_wave_tasks(self):
        """Test getting next wave tasks."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "current_wave": 1,
            "dependency_graph": {
                "waves": [["task-1"], ["task-2", "task-3"], ["task-4"]],
            },
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        next_tasks = get_next_wave_tasks(state)

        assert next_tasks == ["task-2", "task-3"]

    def test_has_unresolved_conflicts(self):
        """Test checking for unresolved conflicts."""
        state_no_conflicts: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.MERGING.value,
            "conflicts": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        state_resolved: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.MERGING.value,
            "conflicts": [{"resolved": True}],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        state_unresolved: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.MERGING.value,
            "conflicts": [{"resolved": False}],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        assert has_unresolved_conflicts(state_no_conflicts) is False
        assert has_unresolved_conflicts(state_resolved) is False
        assert has_unresolved_conflicts(state_unresolved) is True

    def test_get_execution_progress(self):
        """Test getting execution progress."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "tasks": [{"id": f"task-{i}"} for i in range(10)],
            "completed_tasks": ["task-0", "task-1", "task-2", "task-3"],
            "failed_tasks": ["task-4"],
            "skipped_tasks": ["task-5"],
            "current_wave": 2,
            "total_waves": 4,
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        progress = get_execution_progress(state)

        assert progress["total_tasks"] == 10
        assert progress["completed_tasks"] == 4
        assert progress["failed_tasks"] == 1
        assert progress["skipped_tasks"] == 1
        assert progress["pending_tasks"] == 4
        assert progress["progress_percent"] == 60.0


class TestGraphBuilder:
    """Tests for graph building."""

    def test_create_orchestration_graph(self):
        """Test creating the orchestration graph."""
        graph = create_orchestration_graph()

        assert graph is not None
        # Graph should be compiled
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_build_legacy_graph(self):
        """Test building the legacy graph."""
        graph = build_legacy_graph()

        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_get_graph_info(self):
        """Test getting graph information."""
        info = get_graph_info()

        assert info["name"] == "Kappa Orchestration Graph"
        assert info["version"] == "2.0"
        assert len(info["nodes"]) == 7
        assert len(info["edges"]) > 0

        # Check node structure
        node_names = [n["name"] for n in info["nodes"]]
        assert "parse_requirements" in node_names
        assert "generate_tasks" in node_names
        assert "resolve_dependencies" in node_names
        assert "execute_wave" in node_names
        assert "merge_outputs" in node_names
        assert "validate" in node_names
        assert "handle_error" in node_names

    def test_visualize_graph(self):
        """Test graph visualization."""
        mermaid = visualize_graph()

        assert "```mermaid" in mermaid
        assert "parse" in mermaid.lower()
        assert "validate" in mermaid.lower()
        assert "END" in mermaid


class TestLegacyEdgeRouting:
    """Tests for legacy edge routing functions."""

    def test_route_after_decomposition_has_tasks(self):
        """Test routing after decomposition with tasks."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.PLANNING_COMPLETE.value,
            "tasks": [{"id": "task-1"}],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_decomposition(state)
        assert result == "execute_wave"

    def test_route_after_decomposition_no_tasks(self):
        """Test routing after decomposition without tasks."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.PLANNING_COMPLETE.value,
            "tasks": [],
            "completed_tasks": [],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_decomposition(state)
        assert result == "finalize"

    def test_route_after_wave_more_waves(self):
        """Test routing after wave with more waves."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "current_wave": 1,
            "total_waves": 3,
            "task_results": [],
            "completed_tasks": ["task-1"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_wave(state)
        assert result == "execute_wave"

    def test_route_after_wave_with_conflicts(self):
        """Test routing after wave with conflicts."""
        state: KappaState = {
            "project_id": "test",
            "project_name": "test",
            "requirements_text": "",
            "workspace_path": "",
            "status": ExecutionStatus.EXECUTING.value,
            "current_wave": 3,
            "total_waves": 3,
            "task_results": [
                {
                    "task_id": "task-1",
                    "files_modified": ["shared.py"],
                    "files_created": [],
                },
                {
                    "task_id": "task-2",
                    "files_modified": ["shared.py"],  # Same file!
                    "files_created": [],
                },
            ],
            "completed_tasks": ["task-1", "task-2"],
            "failed_tasks": [],
            "skipped_tasks": [],
            "created_files": [],
            "modified_files": [],
            "execution_logs": [],
        }

        result = route_after_wave(state)
        assert result == "resolve_conflicts"
