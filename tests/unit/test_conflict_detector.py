"""
Unit tests for the ConflictDetector module.

Tests the conflict detection system including:
- ConflictType and ConflictSeverity enums
- Conflict dataclass
- ConflictReport dataclass
- ConflictDetector class with all detection methods
"""


import pytest

from src.conflict.detector import (
    Conflict,
    ConflictDetector,
    ConflictReport,
    ConflictSeverity,
    ConflictType,
    detect_conflicts,
)
from src.decomposition.models import DependencyGraph, SessionType, TaskSpec

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing with no conflicts."""
    return [
        TaskSpec(
            id="task-1",
            title="Create User types",
            description="Create TypeScript types for User model",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/types/user.ts"],
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="task-2",
            title="Create Product types",
            description="Create TypeScript types for Product model",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/types/product.ts"],
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="task-3",
            title="Create Order service",
            description="Create Order service for order processing",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/services/order.ts"],
            files_to_modify=[],
            dependencies=["task-1"],
            wave_number=1,
        ),
    ]


@pytest.fixture
def conflicting_tasks():
    """Create tasks with file conflicts."""
    return [
        TaskSpec(
            id="task-a",
            title="Create index",
            description="Create index file",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/index.ts"],
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="task-b",
            title="Update index",
            description="Update index file with exports",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/index.ts"],  # Same file - conflict!
            files_to_modify=[],
            wave_number=0,
        ),
    ]


@pytest.fixture
def detector():
    """Create ConflictDetector instance."""
    return ConflictDetector()


# =============================================================================
# TEST ConflictType ENUM
# =============================================================================


class TestConflictType:
    """Tests for ConflictType enum."""

    def test_conflict_type_values(self):
        """Test ConflictType enum has expected values."""
        assert ConflictType.FILE_WRITE.value == "file_write"
        assert ConflictType.IMPORT_COLLISION.value == "import_collision"
        assert ConflictType.NAMING_CONFLICT.value == "naming_conflict"
        assert ConflictType.TYPE_MISMATCH.value == "type_mismatch"
        assert ConflictType.DEPENDENCY_VIOLATION.value == "dependency_violation"
        assert ConflictType.RESOURCE_CONTENTION.value == "resource_contention"

    def test_conflict_type_is_string_enum(self):
        """Test ConflictType inherits from str and Enum."""
        assert isinstance(ConflictType.FILE_WRITE, str)
        assert ConflictType.FILE_WRITE == "file_write"


# =============================================================================
# TEST ConflictSeverity ENUM
# =============================================================================


class TestConflictSeverity:
    """Tests for ConflictSeverity enum."""

    def test_severity_values(self):
        """Test ConflictSeverity enum has expected values."""
        assert ConflictSeverity.CRITICAL.value == "critical"
        assert ConflictSeverity.HIGH.value == "high"
        assert ConflictSeverity.MEDIUM.value == "medium"
        assert ConflictSeverity.LOW.value == "low"


# =============================================================================
# TEST Conflict DATACLASS
# =============================================================================


class TestConflict:
    """Tests for Conflict dataclass."""

    def test_create_conflict(self):
        """Test creating a Conflict instance."""
        conflict = Conflict(
            id="CONFLICT-0001",
            conflict_type=ConflictType.FILE_WRITE,
            severity=ConflictSeverity.CRITICAL,
            description="Multiple tasks write to same file",
            affected_tasks=["task-1", "task-2"],
            affected_files=["src/index.ts"],
        )

        assert conflict.id == "CONFLICT-0001"
        assert conflict.conflict_type == ConflictType.FILE_WRITE
        assert conflict.severity == ConflictSeverity.CRITICAL
        assert len(conflict.affected_tasks) == 2

    def test_conflict_to_dict(self):
        """Test Conflict.to_dict() serialization."""
        conflict = Conflict(
            id="CONFLICT-0001",
            conflict_type=ConflictType.FILE_WRITE,
            severity=ConflictSeverity.HIGH,
            description="Test conflict",
            affected_tasks=["task-1"],
            affected_files=["file.ts"],
            auto_resolvable=True,
        )

        d = conflict.to_dict()

        assert d["id"] == "CONFLICT-0001"
        assert d["type"] == "file_write"
        assert d["severity"] == "high"
        assert d["auto_resolvable"] is True

    def test_conflict_with_metadata(self):
        """Test Conflict with metadata."""
        conflict = Conflict(
            id="CONFLICT-0002",
            conflict_type=ConflictType.NAMING_CONFLICT,
            severity=ConflictSeverity.MEDIUM,
            description="Duplicate component name",
            affected_tasks=["task-1", "task-2"],
            metadata={"name": "Button", "name_type": "component"},
        )

        assert conflict.metadata["name"] == "Button"
        assert conflict.metadata["name_type"] == "component"


# =============================================================================
# TEST ConflictReport DATACLASS
# =============================================================================


class TestConflictReport:
    """Tests for ConflictReport dataclass."""

    def test_create_empty_report(self):
        """Test creating an empty ConflictReport."""
        report = ConflictReport()

        assert report.total_conflicts == 0
        assert report.critical_count == 0
        assert report.can_proceed is True
        assert len(report.conflicts) == 0

    def test_add_conflict_updates_counts(self):
        """Test adding conflicts updates counts correctly."""
        report = ConflictReport()

        # Add HIGH conflict
        report.add_conflict(
            Conflict(
                id="C1",
                conflict_type=ConflictType.FILE_WRITE,
                severity=ConflictSeverity.HIGH,
                description="Test",
                affected_tasks=["t1"],
            )
        )

        assert report.total_conflicts == 1
        assert report.high_count == 1
        assert report.can_proceed is True

    def test_critical_conflict_blocks_proceed(self):
        """Test CRITICAL conflict sets can_proceed to False."""
        report = ConflictReport()

        report.add_conflict(
            Conflict(
                id="C1",
                conflict_type=ConflictType.DEPENDENCY_VIOLATION,
                severity=ConflictSeverity.CRITICAL,
                description="Critical issue",
                affected_tasks=["t1"],
            )
        )

        assert report.critical_count == 1
        assert report.can_proceed is False

    def test_get_by_type(self):
        """Test filtering conflicts by type."""
        report = ConflictReport()

        report.add_conflict(
            Conflict(
                id="C1",
                conflict_type=ConflictType.FILE_WRITE,
                severity=ConflictSeverity.HIGH,
                description="File write",
                affected_tasks=["t1"],
            )
        )
        report.add_conflict(
            Conflict(
                id="C2",
                conflict_type=ConflictType.NAMING_CONFLICT,
                severity=ConflictSeverity.MEDIUM,
                description="Naming",
                affected_tasks=["t2"],
            )
        )

        file_conflicts = report.get_by_type(ConflictType.FILE_WRITE)

        assert len(file_conflicts) == 1
        assert file_conflicts[0].id == "C1"

    def test_get_auto_resolvable(self):
        """Test filtering auto-resolvable conflicts."""
        report = ConflictReport()

        report.add_conflict(
            Conflict(
                id="C1",
                conflict_type=ConflictType.FILE_WRITE,
                severity=ConflictSeverity.HIGH,
                description="Not auto",
                affected_tasks=["t1"],
                auto_resolvable=False,
            )
        )
        report.add_conflict(
            Conflict(
                id="C2",
                conflict_type=ConflictType.NAMING_CONFLICT,
                severity=ConflictSeverity.MEDIUM,
                description="Auto",
                affected_tasks=["t2"],
                auto_resolvable=True,
            )
        )

        auto = report.get_auto_resolvable()

        assert len(auto) == 1
        assert auto[0].id == "C2"

    def test_report_to_dict(self):
        """Test ConflictReport.to_dict() serialization."""
        report = ConflictReport()
        report.warnings.append("Test warning")

        d = report.to_dict()

        assert "total_conflicts" in d
        assert "can_proceed" in d
        assert "warnings" in d
        assert "conflicts" in d


# =============================================================================
# TEST ConflictDetector CLASS
# =============================================================================


class TestConflictDetector:
    """Tests for ConflictDetector class."""

    def test_detector_init(self, detector):
        """Test ConflictDetector initialization."""
        assert detector._conflict_counter == 0

    def test_analyze_no_conflicts(self, detector, sample_tasks):
        """Test analyze with no conflicts."""
        report = detector.analyze(sample_tasks)

        assert report.total_conflicts == 0
        assert report.can_proceed is True

    def test_analyze_detects_file_write_conflict(self, detector, conflicting_tasks):
        """Test analyze detects file write conflicts."""
        report = detector.analyze(conflicting_tasks)

        assert report.total_conflicts > 0
        file_conflicts = report.get_by_type(ConflictType.FILE_WRITE)
        assert len(file_conflicts) > 0

    def test_analyze_with_dependency_graph(self, detector, sample_tasks):
        """Test analyze with dependency graph."""
        graph = DependencyGraph(
            nodes={t.id: t for t in sample_tasks},
            edges={"task-1": ["task-3"]},
            waves=[["task-1", "task-2"], ["task-3"]],
        )

        report = detector.analyze(sample_tasks, graph)

        assert report is not None

    def test_analyze_wave(self, detector, sample_tasks):
        """Test analyze_wave for wave-specific analysis."""
        wave_tasks = [t for t in sample_tasks if t.wave_number == 0]

        report = detector.analyze_wave(
            wave_tasks=wave_tasks,
            completed_tasks=set(),
            context={},
        )

        assert report is not None

    def test_conflict_id_generation(self, detector):
        """Test unique conflict ID generation."""
        id1 = detector._generate_conflict_id()
        id2 = detector._generate_conflict_id()

        assert id1 != id2
        assert id1.startswith("CONFLICT-")
        assert id2.startswith("CONFLICT-")


class TestConflictDetectorHelpers:
    """Tests for ConflictDetector helper methods."""

    @pytest.fixture
    def detector(self):
        return ConflictDetector()

    def test_normalize_path(self, detector):
        """Test path normalization."""
        path1 = detector._normalize_path("src/index.ts")
        path2 = detector._normalize_path("./src/index.ts")

        # Both should normalize to the same path
        assert path1 == "src/index.ts"

    def test_extract_names_from_task(self, detector):
        """Test extracting names from task description."""
        task = TaskSpec(
            id="test",
            title="Create UserComponent",
            description="Create the User component with useState hook",
            session_type=SessionType.TERMINAL,
            files_to_create=[],
            files_to_modify=[],
        )

        names = detector._extract_names_from_task(task)

        assert "components" in names
        assert "functions" in names
        assert "types" in names


# =============================================================================
# TEST LEGACY detect_conflicts FUNCTION
# =============================================================================


@pytest.mark.asyncio
class TestDetectConflictsFunction:
    """Tests for legacy detect_conflicts function."""

    async def test_detect_conflicts_no_conflicts(self):
        """Test detect_conflicts with no conflicts."""
        results = [
            {
                "session_id": "session-1",
                "task_id": "task-1",
                "files_modified": ["src/a.ts"],
            },
            {
                "session_id": "session-2",
                "task_id": "task-2",
                "files_modified": ["src/b.ts"],
            },
        ]

        conflicts = await detect_conflicts(results)

        assert len(conflicts) == 0

    async def test_detect_conflicts_finds_conflict(self):
        """Test detect_conflicts finds file conflicts."""
        results = [
            {
                "session_id": "session-1",
                "task_id": "task-1",
                "files_modified": ["src/index.ts"],
            },
            {
                "session_id": "session-2",
                "task_id": "task-2",
                "files_modified": ["src/index.ts"],  # Same file
            },
        ]

        conflicts = await detect_conflicts(results)

        assert len(conflicts) > 0
        assert conflicts[0]["file_path"] == "src/index.ts"

    async def test_detect_conflicts_multiple_files(self):
        """Test detect_conflicts with multiple conflicting files."""
        results = [
            {
                "session_id": "session-1",
                "task_id": "task-1",
                "files_modified": ["src/a.ts", "src/b.ts"],
            },
            {
                "session_id": "session-2",
                "task_id": "task-2",
                "files_modified": ["src/b.ts", "src/c.ts"],
            },
        ]

        conflicts = await detect_conflicts(results)

        # Only src/b.ts should be conflicting
        assert len(conflicts) == 1
        assert conflicts[0]["file_path"] == "src/b.ts"
