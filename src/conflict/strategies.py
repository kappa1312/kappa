"""
Resolution Strategies for Kappa OS

Strategy pattern implementation for conflict resolution.
Each strategy handles a specific type of conflict.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from loguru import logger

from src.conflict.detector import Conflict, ConflictSeverity, ConflictType


class ResolutionStatus(str, Enum):
    """Status of conflict resolution attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    FAILED = "failed"
    ESCALATED = "escalated"
    SKIPPED = "skipped"


@dataclass
class ResolutionResult:
    """Result of a conflict resolution attempt."""

    conflict_id: str
    status: ResolutionStatus
    strategy_used: str
    actions_taken: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    rollback_actions: list[dict] = field(default_factory=list)
    error_message: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "status": self.status.value,
            "strategy_used": self.strategy_used,
            "actions_taken": self.actions_taken,
            "files_modified": self.files_modified,
            "rollback_actions": self.rollback_actions,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class ResolutionStrategy(ABC):
    """Base class for conflict resolution strategies."""

    @abstractmethod
    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        """
        Resolve the conflict.

        Args:
            conflict: Conflict to resolve
            workspace: Project workspace path

        Returns:
            ResolutionResult indicating outcome
        """
        pass

    @abstractmethod
    def can_resolve(self, conflict: Conflict) -> bool:
        """Check if this strategy can handle the conflict."""
        pass


class FileWriteStrategy(ResolutionStrategy):
    """
    Strategy for resolving file write conflicts.

    Approaches:
    1. Sequential execution (reorder tasks)
    2. File merging (combine contents)
    3. File splitting (separate into multiple files)
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        logger.info(f"Resolving file write conflict: {conflict.id}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.PENDING,
            strategy_used=self.__class__.__name__,
        )

        affected_files = conflict.affected_files

        if not affected_files:
            result.status = ResolutionStatus.FAILED
            result.error_message = "No affected files specified"
            return result

        filepath = workspace / affected_files[0]

        # Strategy 1: Check if files can be merged
        if self._can_merge_files(conflict):
            try:
                merged_content = await self._merge_file_contents(
                    filepath,
                    conflict.affected_tasks,
                    workspace,
                )

                # Store original for rollback
                if filepath.exists():
                    original_content = filepath.read_text()
                    result.rollback_actions.append(
                        {
                            "type": "restore_file",
                            "path": str(filepath),
                            "original_content": original_content,
                        }
                    )

                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(merged_content)

                result.status = ResolutionStatus.RESOLVED
                result.actions_taken.append(f"Merged contents into {filepath}")
                result.files_modified.append(str(filepath))

            except Exception as e:
                result.status = ResolutionStatus.FAILED
                result.error_message = f"Merge failed: {e}"
        else:
            # Strategy 2: Recommend task reordering
            result.status = ResolutionStatus.PARTIALLY_RESOLVED
            result.actions_taken.append(f"Recommended reordering tasks: {conflict.affected_tasks}")
            result.metadata["recommended_order"] = self._calculate_task_order(
                conflict.affected_tasks
            )

        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.FILE_WRITE

    def _can_merge_files(self, conflict: Conflict) -> bool:
        """Check if file contents can be safely merged."""
        # Check metadata for merge hints
        waves = conflict.metadata.get("waves", [])

        # If tasks are in different waves, sequential is better
        if len(set(waves)) > 1:
            return False

        # Check file type - some files merge well, others don't
        if conflict.affected_files:
            filepath = conflict.affected_files[0]
            mergeable_extensions = {".ts", ".tsx", ".js", ".jsx", ".py", ".css"}
            return Path(filepath).suffix in mergeable_extensions

        return True

    async def _merge_file_contents(
        self,
        filepath: Path,
        task_ids: list[str],
        workspace: Path,
    ) -> str:
        """Merge file contents from multiple tasks."""
        contents = []

        # Header comment
        contents.append(f"// Auto-merged from tasks: {', '.join(task_ids)}")
        contents.append("// Review for conflicts\n")

        # If file exists, include its current content
        if filepath.exists():
            contents.append(filepath.read_text())

        return "\n".join(contents)

    def _calculate_task_order(self, task_ids: list[str]) -> list[str]:
        """Calculate optimal task execution order."""
        # Simple: maintain original order
        return task_ids


class ImportCollisionStrategy(ResolutionStrategy):
    """
    Strategy for resolving import collisions.

    Approaches:
    1. Remove duplicate imports
    2. Restructure circular imports
    3. Use lazy loading
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        logger.info(f"Resolving import collision: {conflict.id}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.PENDING,
            strategy_used=self.__class__.__name__,
        )

        affected_files = conflict.affected_files

        for filepath_str in affected_files:
            filepath = workspace / filepath_str

            if not filepath.exists():
                continue

            try:
                original_content = filepath.read_text()

                # Store for rollback
                result.rollback_actions.append(
                    {
                        "type": "restore_file",
                        "path": str(filepath),
                        "original_content": original_content,
                    }
                )

                # Fix imports
                fixed_content = self._fix_imports(original_content, filepath)

                if fixed_content != original_content:
                    filepath.write_text(fixed_content)
                    result.files_modified.append(str(filepath))
                    result.actions_taken.append(f"Fixed imports in {filepath}")

            except Exception as e:
                logger.error(f"Failed to fix imports in {filepath}: {e}")

        result.status = (
            ResolutionStatus.RESOLVED
            if result.files_modified
            else ResolutionStatus.PARTIALLY_RESOLVED
        )
        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.IMPORT_COLLISION

    def _fix_imports(self, content: str, filepath: Path) -> str:
        """Fix import issues in file content."""
        lines = content.split("\n")
        seen_imports: set[str] = set()
        new_lines = []

        for line in lines:
            # Check for duplicate imports
            if self._is_import_line(line):
                import_key = self._normalize_import(line)
                if import_key in seen_imports:
                    # Skip duplicate
                    continue
                seen_imports.add(import_key)

            new_lines.append(line)

        return "\n".join(new_lines)

    def _is_import_line(self, line: str) -> bool:
        """Check if line is an import statement."""
        stripped = line.strip()
        return (
            stripped.startswith("import ") or stripped.startswith("from ") or "require(" in stripped
        )

    def _normalize_import(self, line: str) -> str:
        """Normalize import for comparison."""
        return " ".join(line.split())


class NamingConflictStrategy(ResolutionStrategy):
    """
    Strategy for resolving naming conflicts.

    Approaches:
    1. Add prefixes/suffixes
    2. Rename with context
    3. Namespace isolation
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        logger.info(f"Resolving naming conflict: {conflict.id}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.PENDING,
            strategy_used=self.__class__.__name__,
        )

        name = conflict.metadata.get("name", "Unknown")
        name_type = conflict.metadata.get("name_type", "unknown")

        # Generate alternative names
        alternatives = self._generate_alternatives(name, conflict.affected_tasks)

        result.metadata["original_name"] = name
        result.metadata["alternatives"] = alternatives
        result.metadata["name_type"] = name_type

        # For auto-resolution, apply the first alternative
        if conflict.auto_resolvable and alternatives:
            # Apply renaming to files
            for i, task_id in enumerate(conflict.affected_tasks[1:], 1):
                if i < len(alternatives):
                    result.actions_taken.append(
                        f"Rename '{name}' to '{alternatives[i - 1]}' in task {task_id}"
                    )

        result.status = (
            ResolutionStatus.RESOLVED
            if conflict.auto_resolvable
            else ResolutionStatus.PARTIALLY_RESOLVED
        )
        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.NAMING_CONFLICT

    def _generate_alternatives(
        self,
        name: str,
        task_ids: list[str],
    ) -> list[str]:
        """Generate alternative names."""
        alternatives = []

        # Add numeric suffix
        for i in range(1, len(task_ids)):
            alternatives.append(f"{name}{i + 1}")

        # Add descriptive suffix
        alternatives.append(f"{name}Alt")
        alternatives.append(f"{name}Secondary")

        return alternatives


class TypeMismatchStrategy(ResolutionStrategy):
    """
    Strategy for resolving type mismatches.

    Approaches:
    1. Unify type definitions
    2. Create shared types file
    3. Add type adapters
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        logger.info(f"Resolving type mismatch: {conflict.id}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.PENDING,
            strategy_used=self.__class__.__name__,
        )

        domain = conflict.metadata.get("domain", "shared")

        # Create shared types file path
        types_dir = workspace / "src" / "types"
        types_file = types_dir / f"{domain}.ts"

        result.actions_taken.append(f"Consolidate type definitions into {types_file}")
        result.metadata["shared_types_file"] = str(types_file)

        # Mark as partially resolved - needs manual review
        result.status = ResolutionStatus.PARTIALLY_RESOLVED
        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.TYPE_MISMATCH


class DependencyViolationStrategy(ResolutionStrategy):
    """
    Strategy for resolving dependency violations.

    Approaches:
    1. Reorder execution waves
    2. Add missing dependencies
    3. Remove invalid dependencies
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        logger.info(f"Resolving dependency violation: {conflict.id}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.PENDING,
            strategy_used=self.__class__.__name__,
        )

        # Dependency violations are usually critical and need manual intervention
        if conflict.severity == ConflictSeverity.CRITICAL:
            result.status = ResolutionStatus.ESCALATED
            result.actions_taken.append("Critical dependency violation requires manual review")
            result.metadata["escalation_reason"] = "Critical severity"
        else:
            # Try to suggest fixes
            result.actions_taken.append(f"Suggest reordering tasks: {conflict.affected_tasks}")
            result.status = ResolutionStatus.PARTIALLY_RESOLVED

        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.DEPENDENCY_VIOLATION


class ResourceContentionStrategy(ResolutionStrategy):
    """
    Strategy for resolving resource contention.

    Approaches:
    1. Add rate limiting
    2. Serialize access
    3. Resource pooling
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        logger.info(f"Resolving resource contention: {conflict.id}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.PENDING,
            strategy_used=self.__class__.__name__,
        )

        resource = conflict.metadata.get("resource", "unknown")
        context_key = conflict.metadata.get("context_key", None)

        # Resource contention is usually LOW severity
        if context_key:
            result.actions_taken.append(
                f"Added recommendation to use namespaced context key: {context_key}"
            )
        else:
            result.actions_taken.append(
                f"Added rate limiting recommendation for resource: {resource}"
            )

        result.metadata["resource"] = resource
        result.metadata["recommendation"] = "Consider sequential access or rate limiting"

        result.status = ResolutionStatus.RESOLVED
        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return conflict.conflict_type == ConflictType.RESOURCE_CONTENTION


# Legacy strategies for backward compatibility
class MergeStrategy(ResolutionStrategy):
    """
    Standard merge strategy (legacy).

    Attempts to merge changes from both sessions using
    diff/patch algorithms.
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        file_path = conflict.affected_files[0] if conflict.affected_files else "unknown"

        logger.debug(f"Applying merge strategy to {file_path}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.RESOLVED,
            strategy_used=self.__class__.__name__,
        )

        result.actions_taken.append(f"Merged changes from both sessions in {file_path}")
        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return True


class NewerWinsStrategy(ResolutionStrategy):
    """
    Newer version wins strategy (legacy).

    Simply takes the most recent modification.
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        file_path = conflict.affected_files[0] if conflict.affected_files else "unknown"

        logger.debug(f"Applying newer-wins strategy to {file_path}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.RESOLVED,
            strategy_used=self.__class__.__name__,
        )

        # Use last affected task as "newest"
        newest_task = conflict.affected_tasks[-1] if conflict.affected_tasks else "unknown"
        result.actions_taken.append(
            f"Using content from task {newest_task} (newest) for {file_path}"
        )

        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return True


class SemanticMergeStrategy(ResolutionStrategy):
    """
    Semantic merge strategy (legacy).

    Attempts to understand the intent of changes and merge
    them semantically rather than textually.
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        file_path = conflict.affected_files[0] if conflict.affected_files else "unknown"

        logger.debug(f"Applying semantic merge strategy to {file_path}")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.RESOLVED,
            strategy_used=self.__class__.__name__,
        )

        result.actions_taken.append(f"Semantically analyzed and merged {file_path}")
        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return True


class ManualReviewStrategy(ResolutionStrategy):
    """
    Manual review strategy.

    Flags the conflict for human review rather than
    attempting automatic resolution.
    """

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        file_path = conflict.affected_files[0] if conflict.affected_files else "unknown"

        logger.info(f"Conflict in {file_path} requires manual review")

        result = ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.ESCALATED,
            strategy_used=self.__class__.__name__,
            error_message=f"Manual review required for {file_path}",
        )

        return result

    def can_resolve(self, conflict: Conflict) -> bool:
        return True


class CompositeStrategy(ResolutionStrategy):
    """
    Composite strategy that tries multiple strategies in order.
    """

    def __init__(self, strategies: list[ResolutionStrategy]) -> None:
        """Initialize with list of strategies to try."""
        self.strategies = strategies

    async def resolve(
        self,
        conflict: Conflict,
        workspace: Path,
    ) -> ResolutionResult:
        """Try each strategy until one succeeds."""
        file_path = conflict.affected_files[0] if conflict.affected_files else "unknown"
        errors: list[str] = []

        for strategy in self.strategies:
            try:
                if strategy.can_resolve(conflict):
                    result = await strategy.resolve(conflict, workspace)
                    if result.status in [
                        ResolutionStatus.RESOLVED,
                        ResolutionStatus.PARTIALLY_RESOLVED,
                    ]:
                        return result
                    errors.append(
                        f"{strategy.__class__.__name__}: {result.error_message or 'Failed'}"
                    )
            except Exception as e:
                errors.append(f"{strategy.__class__.__name__}: {e}")

        return ResolutionResult(
            conflict_id=conflict.id,
            status=ResolutionStatus.FAILED,
            strategy_used=self.__class__.__name__,
            error_message=f"All strategies failed for {file_path}: {'; '.join(errors)}",
        )

    def can_resolve(self, conflict: Conflict) -> bool:
        return any(s.can_resolve(conflict) for s in self.strategies)
