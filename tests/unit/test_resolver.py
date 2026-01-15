"""Unit tests for legacy conflict resolution API compatibility."""

import pytest

from src.conflict.detector import (
    Conflict,
    ConflictType,
    ConflictSeverity,
    detect_conflicts,
)
from src.conflict.resolver import resolve_conflict
from src.conflict.strategies import (
    MergeStrategy,
    NewerWinsStrategy,
    SemanticMergeStrategy,
)


class TestConflictDetector:
    """Tests for legacy ConflictDetector detect_conflicts function."""

    @pytest.mark.asyncio
    async def test_detect_no_conflicts(self) -> None:
        """Test detection when no conflicts exist."""
        results = [
            {
                "session_id": "a",
                "files_modified": ["src/file1.py"],
            },
            {
                "session_id": "b",
                "files_modified": ["src/file2.py"],
            },
        ]

        conflicts = await detect_conflicts(results)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts(self, sample_task_results: list) -> None:
        """Test detection of conflicts."""
        conflicts = await detect_conflicts(sample_task_results)

        # src/models/user.py is modified by both session-2 and session-3
        assert len(conflicts) == 1
        assert conflicts[0]["file_path"] == "src/models/user.py"

    @pytest.mark.asyncio
    async def test_detect_multiple_conflicts(self) -> None:
        """Test detection of multiple conflicts."""
        results = [
            {"session_id": "a", "files_modified": ["file1.py", "file2.py"]},
            {"session_id": "b", "files_modified": ["file1.py", "file2.py"]},
            {"session_id": "c", "files_modified": ["file2.py"]},
        ]

        conflicts = await detect_conflicts(results)

        assert len(conflicts) == 2


class TestConflictResolver:
    """Tests for legacy ConflictResolver resolve_conflict function."""

    @pytest.mark.asyncio
    async def test_resolve_merge_conflict(self) -> None:
        """Test resolving a merge conflict."""
        conflict = {
            "file_path": "src/main.py",
            "conflict_type": "merge",
            "session_a_id": "a",
            "session_b_id": "b",
        }

        resolution = await resolve_conflict(conflict)

        assert resolution is not None
        assert "src/main.py" in resolution or "Merged" in resolution

    @pytest.mark.asyncio
    async def test_resolve_overwrite_conflict(self) -> None:
        """Test resolving an overwrite conflict."""
        conflict = {
            "file_path": "src/config.py",
            "conflict_type": "overwrite",
            "sessions": [
                {"session_id": "a"},
                {"session_id": "b"},
            ],
        }

        resolution = await resolve_conflict(conflict)

        assert resolution is not None

    @pytest.mark.asyncio
    async def test_resolve_all(self, sample_conflicts: list) -> None:
        """Test resolving all conflicts."""
        resolved = []
        for conflict in sample_conflicts:
            result = await resolve_conflict(conflict)
            resolved.append({"conflict": conflict, "resolved": result})

        assert len(resolved) == len(sample_conflicts)
        for item in resolved:
            assert "resolved" in item

    def test_register_strategy(self) -> None:
        """Test that legacy strategies can be instantiated."""
        # Legacy strategies should still be importable
        merge_strategy = MergeStrategy()
        assert merge_strategy is not None


class TestResolutionStrategies:
    """Tests for legacy resolution strategies."""

    @pytest.mark.asyncio
    async def test_merge_strategy(self, tmp_path) -> None:
        """Test MergeStrategy."""
        strategy = MergeStrategy()
        conflict = Conflict(
            id="test-conflict-1",
            conflict_type=ConflictType.FILE_WRITE,
            severity=ConflictSeverity.HIGH,
            description="Test merge conflict",
            affected_tasks=["task-a", "task-b"],
            affected_files=["test.py"],
            metadata={
                "content_a": "def foo(): pass",
                "content_b": "def bar(): pass",
            },
        )

        result = await strategy.resolve(conflict, workspace=tmp_path)

        assert result is not None
        assert "test.py" in str(result)

    @pytest.mark.asyncio
    async def test_newer_wins_strategy(self, tmp_path) -> None:
        """Test NewerWinsStrategy."""
        strategy = NewerWinsStrategy()
        conflict = Conflict(
            id="test-conflict-2",
            conflict_type=ConflictType.FILE_WRITE,
            severity=ConflictSeverity.MEDIUM,
            description="Test newer wins conflict",
            affected_tasks=["old-task", "new-task"],
            affected_files=["test.py"],
        )

        result = await strategy.resolve(conflict, workspace=tmp_path)

        assert result is not None
        assert "new" in str(result)

    @pytest.mark.asyncio
    async def test_semantic_merge_strategy(self, tmp_path) -> None:
        """Test SemanticMergeStrategy."""
        strategy = SemanticMergeStrategy()
        conflict = Conflict(
            id="test-conflict-3",
            conflict_type=ConflictType.IMPORT_COLLISION,
            severity=ConflictSeverity.LOW,
            description="Test semantic merge conflict",
            affected_tasks=["task-1"],
            affected_files=["test.py"],
        )

        result = await strategy.resolve(conflict, workspace=tmp_path)

        assert result is not None
        assert "test.py" in str(result)
