"""
Unit tests for the ConflictResolver module.

Tests the conflict resolution system including:
- ResolutionPlan dataclass
- ConflictResolver class
- Strategy registration and selection
- Resolution execution and rollback
"""


import pytest

from src.conflict.detector import (
    Conflict,
    ConflictReport,
    ConflictSeverity,
    ConflictType,
)
from src.conflict.resolver import (
    ConflictResolver,
    ResolutionPlan,
    resolve_conflict,
)
from src.conflict.strategies import (
    FileWriteStrategy,
    ResolutionResult,
    ResolutionStatus,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def resolver(tmp_path):
    """Create ConflictResolver with temp workspace."""
    return ConflictResolver(workspace_path=tmp_path)


@pytest.fixture
def sample_conflict():
    """Create a sample conflict for testing."""
    return Conflict(
        id="CONFLICT-001",
        conflict_type=ConflictType.FILE_WRITE,
        severity=ConflictSeverity.HIGH,
        description="Multiple tasks write to src/index.ts",
        affected_tasks=["task-1", "task-2"],
        affected_files=["src/index.ts"],
        auto_resolvable=True,
    )


@pytest.fixture
def sample_report():
    """Create a sample ConflictReport for testing."""
    report = ConflictReport()
    report.add_conflict(
        Conflict(
            id="C1",
            conflict_type=ConflictType.FILE_WRITE,
            severity=ConflictSeverity.HIGH,
            description="File conflict 1",
            affected_tasks=["t1", "t2"],
            affected_files=["src/a.ts"],
            auto_resolvable=True,
        )
    )
    report.add_conflict(
        Conflict(
            id="C2",
            conflict_type=ConflictType.NAMING_CONFLICT,
            severity=ConflictSeverity.MEDIUM,
            description="Naming conflict",
            affected_tasks=["t3", "t4"],
            auto_resolvable=True,
            metadata={"name": "Button", "name_type": "component"},
        )
    )
    return report


# =============================================================================
# TEST ResolutionPlan DATACLASS
# =============================================================================


class TestResolutionPlan:
    """Tests for ResolutionPlan dataclass."""

    def test_create_plan(self):
        """Test creating a ResolutionPlan."""
        conflict = Conflict(
            id="C1",
            conflict_type=ConflictType.FILE_WRITE,
            severity=ConflictSeverity.HIGH,
            description="Test",
            affected_tasks=["t1"],
        )

        plan = ResolutionPlan(
            conflicts=[conflict],
            resolution_order=["C1"],
            strategies={"C1": "FileWriteStrategy"},
        )

        assert len(plan.conflicts) == 1
        assert plan.resolution_order == ["C1"]

    def test_plan_to_dict(self):
        """Test ResolutionPlan.to_dict() serialization."""
        conflict = Conflict(
            id="C1",
            conflict_type=ConflictType.FILE_WRITE,
            severity=ConflictSeverity.HIGH,
            description="Test",
            affected_tasks=["t1"],
        )

        plan = ResolutionPlan(
            conflicts=[conflict],
            resolution_order=["C1"],
            strategies={"C1": "FileWriteStrategy"},
            requires_manual=["C2"],
        )

        d = plan.to_dict()

        assert "conflicts" in d
        assert "resolution_order" in d
        assert "strategies" in d
        assert "requires_manual" in d


# =============================================================================
# TEST ConflictResolver CLASS
# =============================================================================


class TestConflictResolver:
    """Tests for ConflictResolver class."""

    def test_resolver_init(self, resolver):
        """Test ConflictResolver initialization."""
        assert resolver.workspace is not None
        assert len(resolver._strategies) > 0

    def test_resolver_has_default_strategies(self, resolver):
        """Test resolver registers default strategies."""
        assert ConflictType.FILE_WRITE in resolver._strategies
        assert ConflictType.IMPORT_COLLISION in resolver._strategies
        assert ConflictType.NAMING_CONFLICT in resolver._strategies
        assert ConflictType.TYPE_MISMATCH in resolver._strategies
        assert ConflictType.DEPENDENCY_VIOLATION in resolver._strategies
        assert ConflictType.RESOURCE_CONTENTION in resolver._strategies

    def test_register_custom_strategy(self, resolver):
        """Test registering a custom strategy."""
        custom_strategy = FileWriteStrategy()

        resolver.register_strategy(ConflictType.FILE_WRITE, custom_strategy)

        assert resolver._strategies[ConflictType.FILE_WRITE] == custom_strategy

    @pytest.mark.asyncio
    async def test_resolve_single(self, resolver, sample_conflict):
        """Test resolving a single conflict."""
        result = await resolver.resolve_single(sample_conflict)

        assert isinstance(result, ResolutionResult)
        assert result.conflict_id == "CONFLICT-001"

    @pytest.mark.asyncio
    async def test_resolve_all(self, resolver, sample_report):
        """Test resolving all conflicts in a report."""
        results = await resolver.resolve_all(sample_report)

        assert len(results) == 2
        assert all(isinstance(r, ResolutionResult) for r in results)

    @pytest.mark.asyncio
    async def test_resolve_all_auto_only(self, resolver):
        """Test resolving only auto-resolvable conflicts."""
        report = ConflictReport()
        report.add_conflict(
            Conflict(
                id="C1",
                conflict_type=ConflictType.FILE_WRITE,
                severity=ConflictSeverity.HIGH,
                description="Auto",
                affected_tasks=["t1"],
                auto_resolvable=True,
            )
        )
        report.add_conflict(
            Conflict(
                id="C2",
                conflict_type=ConflictType.DEPENDENCY_VIOLATION,
                severity=ConflictSeverity.CRITICAL,
                description="Manual",
                affected_tasks=["t2"],
                auto_resolvable=False,
            )
        )

        results = await resolver.resolve_all(report, auto_only=True)

        # C2 should be skipped
        skipped = [r for r in results if r.status == ResolutionStatus.SKIPPED]
        assert len(skipped) == 1

    @pytest.mark.asyncio
    async def test_preview_resolution(self, resolver, sample_conflict):
        """Test previewing resolution."""
        preview = await resolver.preview_resolution(sample_conflict)

        assert "conflict_id" in preview
        assert "strategy" in preview
        assert "auto_resolvable" in preview

    def test_get_resolution_history(self, resolver):
        """Test getting resolution history."""
        history = resolver.get_resolution_history()

        assert isinstance(history, list)

    def test_clear_history(self, resolver):
        """Test clearing resolution history."""
        resolver.clear_history()

        assert len(resolver._resolution_history) == 0


# =============================================================================
# TEST ROLLBACK FUNCTIONALITY
# =============================================================================


@pytest.mark.asyncio
class TestConflictResolverRollback:
    """Tests for ConflictResolver rollback functionality."""

    async def test_rollback_restore_file(self, tmp_path):
        """Test rolling back a file restore action."""
        resolver = ConflictResolver(workspace_path=tmp_path)

        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        # Create a result with rollback action
        result = ResolutionResult(
            conflict_id="C1",
            status=ResolutionStatus.RESOLVED,
            strategy_used="TestStrategy",
            rollback_actions=[
                {
                    "type": "restore_file",
                    "path": str(test_file),
                    "original_content": "original content",
                }
            ],
        )

        # Modify the file
        test_file.write_text("modified content")

        # Rollback
        success = await resolver.rollback(result)

        assert success is True
        assert test_file.read_text() == "original content"

    async def test_rollback_no_actions(self, tmp_path):
        """Test rollback with no actions."""
        resolver = ConflictResolver(workspace_path=tmp_path)

        result = ResolutionResult(
            conflict_id="C1",
            status=ResolutionStatus.RESOLVED,
            strategy_used="TestStrategy",
        )

        success = await resolver.rollback(result)

        assert success is False


# =============================================================================
# TEST LEGACY resolve_conflict FUNCTION
# =============================================================================


@pytest.mark.asyncio
class TestResolveConflictFunction:
    """Tests for legacy resolve_conflict function."""

    async def test_resolve_conflict_merge(self):
        """Test resolve_conflict with merge type."""
        conflict = {
            "file_path": "src/index.ts",
            "conflict_type": "merge",
            "session_a_id": "session-1",
            "session_b_id": "session-2",
        }

        resolution = await resolve_conflict(conflict)

        assert resolution is not None
        assert "src/index.ts" in resolution or "Merged" in resolution

    async def test_resolve_conflict_overwrite(self):
        """Test resolve_conflict with overwrite type."""
        conflict = {
            "file_path": "src/index.ts",
            "conflict_type": "overwrite",
            "session_a_id": "session-1",
            "session_b_id": "session-2",
        }

        resolution = await resolve_conflict(conflict)

        assert resolution is not None

    async def test_resolve_conflict_semantic(self):
        """Test resolve_conflict with semantic type."""
        conflict = {
            "file_path": "src/index.ts",
            "conflict_type": "semantic",
            "session_a_id": "session-1",
            "session_b_id": "session-2",
        }

        resolution = await resolve_conflict(conflict)

        assert resolution is not None
