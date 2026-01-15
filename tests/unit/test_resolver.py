"""Unit tests for conflict resolution."""

import pytest

from src.conflict.detector import ConflictDetector
from src.conflict.resolver import ConflictResolver
from src.conflict.strategies import (
    MergeStrategy,
    NewerWinsStrategy,
    SemanticMergeStrategy,
)


class TestConflictDetector:
    """Tests for ConflictDetector."""

    @pytest.mark.asyncio
    async def test_detect_no_conflicts(self) -> None:
        """Test detection when no conflicts exist."""
        detector = ConflictDetector()
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

        conflicts = await detector.detect(results)
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts(self, sample_task_results: list) -> None:
        """Test detection of conflicts."""
        detector = ConflictDetector()

        conflicts = await detector.detect(sample_task_results)

        # src/models/user.py is modified by both session-2 and session-3
        assert len(conflicts) == 1
        assert conflicts[0]["file_path"] == "src/models/user.py"

    @pytest.mark.asyncio
    async def test_detect_multiple_conflicts(self) -> None:
        """Test detection of multiple conflicts."""
        detector = ConflictDetector()
        results = [
            {"session_id": "a", "files_modified": ["file1.py", "file2.py"]},
            {"session_id": "b", "files_modified": ["file1.py", "file2.py"]},
            {"session_id": "c", "files_modified": ["file2.py"]},
        ]

        conflicts = await detector.detect(results)

        assert len(conflicts) == 2


class TestConflictResolver:
    """Tests for ConflictResolver."""

    @pytest.mark.asyncio
    async def test_resolve_merge_conflict(self) -> None:
        """Test resolving a merge conflict."""
        resolver = ConflictResolver()
        conflict = {
            "file_path": "src/main.py",
            "conflict_type": "merge",
            "session_a_id": "a",
            "session_b_id": "b",
        }

        resolution = await resolver.resolve(conflict)

        assert resolution is not None
        assert "src/main.py" in resolution

    @pytest.mark.asyncio
    async def test_resolve_overwrite_conflict(self) -> None:
        """Test resolving an overwrite conflict."""
        resolver = ConflictResolver()
        conflict = {
            "file_path": "src/config.py",
            "conflict_type": "overwrite",
            "sessions": [
                {"session_id": "a"},
                {"session_id": "b"},
            ],
        }

        resolution = await resolver.resolve(conflict)

        assert resolution is not None

    @pytest.mark.asyncio
    async def test_resolve_all(self, sample_conflicts: list) -> None:
        """Test resolving all conflicts."""
        resolver = ConflictResolver()

        resolved = await resolver.resolve_all(sample_conflicts)

        assert len(resolved) == len(sample_conflicts)
        for conflict in resolved:
            assert "resolved" in conflict

    def test_register_strategy(self) -> None:
        """Test registering a custom strategy."""
        resolver = ConflictResolver()
        custom_strategy = MergeStrategy()

        resolver.register_strategy("custom", custom_strategy)

        assert "custom" in resolver._strategies


class TestResolutionStrategies:
    """Tests for resolution strategies."""

    @pytest.mark.asyncio
    async def test_merge_strategy(self) -> None:
        """Test MergeStrategy."""
        strategy = MergeStrategy()
        conflict = {
            "file_path": "test.py",
            "content_a": "def foo(): pass",
            "content_b": "def bar(): pass",
        }

        resolution = await strategy.resolve(conflict)

        assert "test.py" in resolution

    @pytest.mark.asyncio
    async def test_newer_wins_strategy(self) -> None:
        """Test NewerWinsStrategy."""
        strategy = NewerWinsStrategy()
        conflict = {
            "file_path": "test.py",
            "sessions": [
                {"session_id": "old"},
                {"session_id": "new"},
            ],
        }

        resolution = await strategy.resolve(conflict)

        assert "new" in resolution

    @pytest.mark.asyncio
    async def test_semantic_merge_strategy(self) -> None:
        """Test SemanticMergeStrategy."""
        strategy = SemanticMergeStrategy()
        conflict = {"file_path": "test.py"}

        resolution = await strategy.resolve(conflict)

        assert "test.py" in resolution
