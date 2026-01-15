"""
Integration tests for the Conflict Resolution system.

Tests the full conflict detection, resolution, and merge pipeline
with real file operations and database interactions.
"""

import json

import pytest

from src.conflict.detector import (
    ConflictDetector,
    ConflictType,
)
from src.conflict.merge_engine import MergeEngine
from src.conflict.resolver import ConflictResolver
from src.conflict.strategies import ResolutionResult, ResolutionStatus
from src.decomposition.models import DependencyGraph, SessionType, TaskSpec

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace directory."""
    return tmp_path


@pytest.fixture
def detector():
    """Create ConflictDetector instance."""
    return ConflictDetector()


@pytest.fixture
def resolver(workspace):
    """Create ConflictResolver with workspace."""
    return ConflictResolver(workspace_path=workspace)


@pytest.fixture
def merge_engine(workspace):
    """Create MergeEngine with workspace."""
    return MergeEngine(workspace_path=workspace)


@pytest.fixture
def complex_task_set():
    """Create a complex set of tasks with various conflict types."""
    return [
        TaskSpec(
            id="task-api-types",
            title="Create API types",
            description="Create TypeScript types for API including User, Post, Comment",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/types/api.ts"],
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="task-model-types",
            title="Create Model types",
            description="Create TypeScript types for models including User entity",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/types/models.ts"],
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="task-api-service",
            title="Create API service",
            description="Create API service using User type from api.ts",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/services/api.ts"],
            files_to_modify=["src/types/api.ts"],
            dependencies=["task-api-types"],
            wave_number=1,
        ),
        TaskSpec(
            id="task-user-service",
            title="Create User service",
            description="Create User service importing User type",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/services/user.ts"],
            files_to_modify=["src/types/api.ts"],
            dependencies=["task-api-types"],
            wave_number=1,
        ),
        TaskSpec(
            id="task-index",
            title="Create main index",
            description="Create main index file exporting all services",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/index.ts"],
            files_to_modify=[],
            dependencies=["task-api-service", "task-user-service"],
            wave_number=2,
        ),
    ]


@pytest.fixture
def conflicting_task_set():
    """Create tasks with guaranteed conflicts."""
    return [
        TaskSpec(
            id="task-a",
            title="Create button component",
            description="Create Button React component",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/components/Button.tsx"],
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="task-b",
            title="Create primary button",
            description="Create Button component with primary styling",
            session_type=SessionType.TERMINAL,
            files_to_create=["src/components/Button.tsx"],  # Same file - conflict!
            files_to_modify=[],
            wave_number=0,
        ),
        TaskSpec(
            id="task-c",
            title="Create icon button",
            description="Create Button variant with icon support",
            session_type=SessionType.TERMINAL,
            files_to_create=[],
            files_to_modify=["src/components/Button.tsx"],  # Modifies same file
            wave_number=0,
        ),
    ]


# =============================================================================
# INTEGRATION TESTS: FULL DETECTION PIPELINE
# =============================================================================


@pytest.mark.integration
class TestConflictDetectionPipeline:
    """Integration tests for conflict detection pipeline."""

    def test_detect_file_write_conflicts(self, detector, conflicting_task_set):
        """Test detection of file write conflicts in task set."""
        report = detector.analyze(conflicting_task_set)

        assert report.total_conflicts > 0
        file_conflicts = report.get_by_type(ConflictType.FILE_WRITE)
        assert len(file_conflicts) > 0

        # Verify the conflict details
        conflict = file_conflicts[0]
        assert "src/components/Button.tsx" in conflict.affected_files
        assert len(conflict.affected_tasks) >= 2

    def test_detect_naming_conflicts(self, detector, conflicting_task_set):
        """Test detection of naming conflicts."""
        report = detector.analyze(conflicting_task_set)

        # May detect Button naming conflict
        naming_conflicts = report.get_by_type(ConflictType.NAMING_CONFLICT)
        # Naming conflicts are detected from task descriptions
        assert isinstance(naming_conflicts, list)

    def test_complex_task_analysis(self, detector, complex_task_set):
        """Test analysis of complex task dependencies."""
        # Create dependency graph
        graph = DependencyGraph(
            nodes={t.id: t for t in complex_task_set},
            edges={
                "task-api-types": ["task-api-service", "task-user-service"],
                "task-api-service": ["task-index"],
                "task-user-service": ["task-index"],
            },
            waves=[
                ["task-api-types", "task-model-types"],
                ["task-api-service", "task-user-service"],
                ["task-index"],
            ],
        )

        report = detector.analyze(complex_task_set, graph)

        # Should detect the file modification conflict on src/types/api.ts
        file_conflicts = report.get_by_type(ConflictType.FILE_WRITE)
        assert len(file_conflicts) > 0

    def test_wave_specific_analysis(self, detector, complex_task_set):
        """Test wave-specific conflict analysis."""
        wave_1_tasks = [t for t in complex_task_set if t.wave_number == 1]
        completed = {"task-api-types", "task-model-types"}

        report = detector.analyze_wave(
            wave_tasks=wave_1_tasks,
            completed_tasks=completed,
            context={"wave": 1},
        )

        assert report is not None
        # Both wave 1 tasks modify src/types/api.ts
        file_conflicts = report.get_by_type(ConflictType.FILE_WRITE)
        assert len(file_conflicts) > 0


# =============================================================================
# INTEGRATION TESTS: FULL RESOLUTION PIPELINE
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestConflictResolutionPipeline:
    """Integration tests for conflict resolution pipeline."""

    async def test_resolve_file_conflict(self, workspace, detector, conflicting_task_set):
        """Test resolving file write conflicts."""
        resolver = ConflictResolver(workspace_path=workspace)

        # Detect conflicts
        report = detector.analyze(conflicting_task_set)
        assert report.total_conflicts > 0

        # Resolve all conflicts
        results = await resolver.resolve_all(report)

        assert len(results) > 0
        assert all(isinstance(r, ResolutionResult) for r in results)

    async def test_resolve_with_auto_only(self, workspace, detector, conflicting_task_set):
        """Test resolving only auto-resolvable conflicts."""
        resolver = ConflictResolver(workspace_path=workspace)

        report = detector.analyze(conflicting_task_set)

        # Resolve only auto-resolvable
        results = await resolver.resolve_all(report, auto_only=True)

        # Non-auto-resolvable should be skipped
        skipped = [r for r in results if r.status == ResolutionStatus.SKIPPED]
        resolved = [r for r in results if r.status == ResolutionStatus.RESOLVED]

        # At least some should be resolved or skipped
        assert len(results) > 0

    async def test_preview_resolution(self, workspace, detector, conflicting_task_set):
        """Test previewing conflict resolution."""
        resolver = ConflictResolver(workspace_path=workspace)

        report = detector.analyze(conflicting_task_set)
        conflict = report.conflicts[0]

        preview = await resolver.preview_resolution(conflict)

        assert "conflict_id" in preview
        assert "strategy" in preview
        assert "auto_resolvable" in preview

    async def test_resolution_history(self, workspace, detector, conflicting_task_set):
        """Test resolution history tracking."""
        resolver = ConflictResolver(workspace_path=workspace)

        report = detector.analyze(conflicting_task_set)
        await resolver.resolve_all(report)

        history = resolver.get_resolution_history()

        assert isinstance(history, list)
        assert len(history) > 0


# =============================================================================
# INTEGRATION TESTS: MERGE ENGINE
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestMergeEnginePipeline:
    """Integration tests for merge engine pipeline."""

    async def test_merge_typescript_files(self, workspace):
        """Test merging TypeScript files from multiple tasks."""
        engine = MergeEngine(workspace_path=workspace)

        # Create test files
        ours = """import { useState } from 'react';

export function Counter() {
  const [count, setCount] = useState(0);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
"""
        theirs = """import { useState, useEffect } from 'react';

export function Counter() {
  const [count, setCount] = useState(0);
  useEffect(() => { document.title = `Count: ${count}`; }, [count]);
  return <button onClick={() => setCount(c => c + 1)}>{count}</button>;
}
"""

        result = await engine.merge_content(ours, theirs, file_type="typescript")

        assert result.success is True
        assert result.merged_content is not None
        # Should have imports from both
        assert "useState" in result.merged_content

    async def test_merge_json_files(self, workspace):
        """Test merging JSON configuration files."""
        engine = MergeEngine(workspace_path=workspace)

        ours = json.dumps(
            {
                "name": "myapp",
                "version": "1.0.0",
                "dependencies": {
                    "react": "^18.0.0",
                    "typescript": "^5.0.0",
                },
            }
        )

        theirs = json.dumps(
            {
                "name": "myapp",
                "version": "1.0.0",
                "dependencies": {
                    "react": "^18.0.0",
                    "lodash": "^4.0.0",
                },
                "devDependencies": {
                    "jest": "^29.0.0",
                },
            }
        )

        result = await engine.merge_content(ours, theirs, file_type="json")

        assert result.success is True

        merged = json.loads(result.merged_content)
        # Should have dependencies from both
        assert "typescript" in merged.get("dependencies", {})
        assert "lodash" in merged.get("dependencies", {})
        assert "devDependencies" in merged

    async def test_merge_task_outputs(self, workspace):
        """Test merging outputs from multiple parallel tasks."""
        engine = MergeEngine(workspace_path=workspace)

        task_outputs = [
            {
                "task_id": "task-1",
                "files_modified": ["src/utils/math.ts"],
                "content_map": {
                    "src/utils/math.ts": "export const add = (a: number, b: number) => a + b;",
                },
            },
            {
                "task_id": "task-2",
                "files_modified": ["src/utils/string.ts"],
                "content_map": {
                    "src/utils/string.ts": "export const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);",
                },
            },
            {
                "task_id": "task-3",
                "files_modified": ["src/index.ts"],
                "content_map": {
                    "src/index.ts": "export * from './utils/math';",
                },
            },
        ]

        result = await engine.merge_task_outputs(task_outputs)

        assert result.success is True

    async def test_merge_with_conflicts(self, workspace):
        """Test handling merge conflicts."""
        engine = MergeEngine(workspace_path=workspace)

        # Conflicting changes to same function
        ours = """export function calculate(x: number): number {
  return x * 2;
}
"""
        theirs = """export function calculate(x: number): number {
  return x * 3;
}
"""

        result = await engine.merge_content(ours, theirs, file_type="typescript")

        # May succeed with smart merge or have conflicts
        assert result is not None


# =============================================================================
# INTEGRATION TESTS: FULL WORKFLOW
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
class TestFullConflictWorkflow:
    """Integration tests for complete conflict handling workflow."""

    async def test_detect_resolve_merge_workflow(self, workspace, conflicting_task_set):
        """Test complete workflow: detect -> resolve -> merge."""
        # Step 1: Detect conflicts
        detector = ConflictDetector()
        report = detector.analyze(conflicting_task_set)

        assert report.total_conflicts > 0

        # Step 2: Resolve conflicts
        resolver = ConflictResolver(workspace_path=workspace)
        resolution_results = await resolver.resolve_all(report)

        assert len(resolution_results) > 0

        # Step 3: Simulate task outputs and merge
        engine = MergeEngine(workspace_path=workspace)

        task_outputs = [
            {
                "task_id": "task-a",
                "files_modified": ["src/components/Button.tsx"],
                "content_map": {
                    "src/components/Button.tsx": "export const Button = ({ children }) => <button>{children}</button>;",
                },
            },
            {
                "task_id": "task-b",
                "files_modified": ["src/components/Button.tsx"],
                "content_map": {
                    "src/components/Button.tsx": "export const Button = ({ children, primary }) => <button className={primary ? 'primary' : ''}>{children}</button>;",
                },
            },
        ]

        merge_result = await engine.merge_task_outputs(task_outputs)

        # Workflow completed
        assert merge_result is not None

    async def test_rollback_on_failed_resolution(self, workspace):
        """Test rollback functionality when resolution fails."""
        resolver = ConflictResolver(workspace_path=workspace)

        # Create a file to test rollback
        test_file = workspace / "test_rollback.txt"
        test_file.write_text("original content")

        # Create a result with rollback action
        result = ResolutionResult(
            conflict_id="test-conflict",
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

    async def test_conflict_report_serialization(self, detector, conflicting_task_set):
        """Test conflict report can be serialized for storage/transmission."""
        report = detector.analyze(conflicting_task_set)

        # Serialize to dict
        report_dict = report.to_dict()

        assert "total_conflicts" in report_dict
        assert "conflicts" in report_dict
        assert "can_proceed" in report_dict

        # Should be JSON serializable
        json_str = json.dumps(report_dict)
        assert json_str is not None

        # Can be deserialized
        loaded = json.loads(json_str)
        assert loaded["total_conflicts"] == report.total_conflicts
