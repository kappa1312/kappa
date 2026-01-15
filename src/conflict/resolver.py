"""
Conflict Resolution System for Kappa OS

Resolves detected conflicts using appropriate strategies:
- Auto-resolution for common conflicts
- Strategy pattern for different conflict types
- Manual escalation for complex cases
- Rollback support for failed resolutions
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.conflict.detector import (
    Conflict,
    ConflictReport,
    ConflictSeverity,
    ConflictType,
)
from src.conflict.strategies import (
    DependencyViolationStrategy,
    FileWriteStrategy,
    ImportCollisionStrategy,
    MergeStrategy,
    NamingConflictStrategy,
    NewerWinsStrategy,
    ResolutionResult,
    ResolutionStatus,
    ResolutionStrategy,
    ResourceContentionStrategy,
    SemanticMergeStrategy,
    TypeMismatchStrategy,
)


@dataclass
class ResolutionPlan:
    """Plan for resolving multiple conflicts."""

    conflicts: list[Conflict]
    resolution_order: list[str]  # Conflict IDs in resolution order
    strategies: dict[str, str]  # conflict_id -> strategy_name
    estimated_time: float = 0.0
    requires_manual: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolution_order": self.resolution_order,
            "strategies": self.strategies,
            "estimated_time": self.estimated_time,
            "requires_manual": self.requires_manual,
        }


class ConflictResolver:
    """
    Resolves conflicts using appropriate strategies.

    Usage:
        resolver = ConflictResolver()
        results = await resolver.resolve_all(conflict_report)

        for result in results:
            if result.status == ResolutionStatus.FAILED:
                # Handle failed resolution
                pass
    """

    def __init__(self, workspace_path: Path | None = None):
        self.workspace = workspace_path or Path.cwd()
        self._strategies: dict[ConflictType, ResolutionStrategy] = {}
        self._legacy_strategies: dict[str, ResolutionStrategy] = {}
        self._resolution_history: list[ResolutionResult] = []
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default resolution strategies for each conflict type."""
        self._strategies[ConflictType.FILE_WRITE] = FileWriteStrategy()
        self._strategies[ConflictType.IMPORT_COLLISION] = ImportCollisionStrategy()
        self._strategies[ConflictType.NAMING_CONFLICT] = NamingConflictStrategy()
        self._strategies[ConflictType.TYPE_MISMATCH] = TypeMismatchStrategy()
        self._strategies[ConflictType.DEPENDENCY_VIOLATION] = DependencyViolationStrategy()
        self._strategies[ConflictType.RESOURCE_CONTENTION] = ResourceContentionStrategy()

        # Legacy strategies for backward compatibility
        self._legacy_strategies["merge"] = MergeStrategy()
        self._legacy_strategies["overwrite"] = NewerWinsStrategy()
        self._legacy_strategies["semantic"] = SemanticMergeStrategy()

    def register_strategy(
        self,
        conflict_type: ConflictType,
        strategy: ResolutionStrategy,
    ):
        """Register a custom resolution strategy."""
        self._strategies[conflict_type] = strategy
        logger.info(f"Registered custom strategy for {conflict_type.value}")

    async def resolve_all(
        self,
        report: ConflictReport,
        auto_only: bool = False,
    ) -> list[ResolutionResult]:
        """
        Resolve all conflicts in report.

        Args:
            report: ConflictReport from ConflictDetector
            auto_only: If True, only resolve auto-resolvable conflicts

        Returns:
            List of ResolutionResult for each conflict
        """
        logger.info(f"Resolving {report.total_conflicts} conflicts")

        results = []
        plan = self._create_resolution_plan(report, auto_only)

        for conflict_id in plan.resolution_order:
            conflict = self._find_conflict(conflict_id, report.conflicts)

            if conflict is None:
                continue

            if auto_only and not conflict.auto_resolvable:
                results.append(
                    ResolutionResult(
                        conflict_id=conflict_id,
                        status=ResolutionStatus.SKIPPED,
                        strategy_used="none",
                        actions_taken=["Skipped - requires manual resolution"],
                    )
                )
                continue

            result = await self._resolve_single(conflict)
            results.append(result)
            self._resolution_history.append(result)

            # Stop if critical resolution fails
            if (
                conflict.severity == ConflictSeverity.CRITICAL
                and result.status == ResolutionStatus.FAILED
            ):
                logger.error(f"Critical conflict resolution failed: {conflict_id}")
                break

        resolved_count = sum(
            1
            for r in results
            if r.status in [ResolutionStatus.RESOLVED, ResolutionStatus.PARTIALLY_RESOLVED]
        )
        logger.info(f"Resolution complete: {resolved_count}/{len(results)} resolved")

        return results

    async def resolve_single(self, conflict: Conflict) -> ResolutionResult:
        """Resolve a single conflict."""
        return await self._resolve_single(conflict)

    async def _resolve_single(self, conflict: Conflict) -> ResolutionResult:
        """Internal method to resolve a single conflict."""
        logger.info(f"Resolving conflict {conflict.id}: {conflict.conflict_type.value}")

        strategy = self._strategies.get(conflict.conflict_type)

        if strategy is None:
            return ResolutionResult(
                conflict_id=conflict.id,
                status=ResolutionStatus.FAILED,
                strategy_used="none",
                error_message=f"No strategy registered for {conflict.conflict_type.value}",
            )

        try:
            result = await strategy.resolve(conflict, self.workspace)
            return result
        except Exception as e:
            logger.error(f"Resolution failed for {conflict.id}: {e}")
            return ResolutionResult(
                conflict_id=conflict.id,
                status=ResolutionStatus.FAILED,
                strategy_used=strategy.__class__.__name__,
                error_message=str(e),
            )

    def _create_resolution_plan(
        self,
        report: ConflictReport,
        auto_only: bool,
    ) -> ResolutionPlan:
        """Create optimized resolution plan."""
        # Sort conflicts by severity and resolvability
        sorted_conflicts = sorted(
            report.conflicts,
            key=lambda c: (
                c.severity != ConflictSeverity.CRITICAL,
                c.severity != ConflictSeverity.HIGH,
                not c.auto_resolvable,
            ),
        )

        plan = ResolutionPlan(
            conflicts=sorted_conflicts,
            resolution_order=[c.id for c in sorted_conflicts],
            strategies={
                c.id: self._strategies.get(c.conflict_type, MergeStrategy()).__class__.__name__
                for c in sorted_conflicts
            },
        )

        # Mark conflicts requiring manual intervention
        for conflict in sorted_conflicts:
            if not conflict.auto_resolvable:
                plan.requires_manual.append(conflict.id)

        return plan

    def _find_conflict(
        self,
        conflict_id: str,
        conflicts: list[Conflict],
    ) -> Conflict | None:
        """Find conflict by ID."""
        for conflict in conflicts:
            if conflict.id == conflict_id:
                return conflict
        return None

    async def rollback(self, result: ResolutionResult) -> bool:
        """
        Rollback a resolution attempt.

        Args:
            result: ResolutionResult to rollback

        Returns:
            True if rollback successful
        """
        if not result.rollback_actions:
            logger.warning(f"No rollback actions for {result.conflict_id}")
            return False

        logger.info(f"Rolling back resolution for {result.conflict_id}")

        try:
            for action in reversed(result.rollback_actions):
                await self._execute_rollback_action(action)
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def _execute_rollback_action(self, action: dict):
        """Execute a single rollback action."""
        action_type = action.get("type")

        if action_type == "restore_file":
            filepath = Path(action["path"])
            content = action["original_content"]
            filepath.write_text(content)

        elif action_type == "delete_file":
            filepath = Path(action["path"])
            if filepath.exists():
                filepath.unlink()

        elif action_type == "rename_file":
            src = Path(action["new_path"])
            dst = Path(action["original_path"])
            if src.exists():
                src.rename(dst)

        logger.debug(f"Rollback action executed: {action_type}")

    def get_resolution_history(self) -> list[ResolutionResult]:
        """Get history of all resolution attempts."""
        return self._resolution_history.copy()

    def clear_history(self):
        """Clear resolution history."""
        self._resolution_history.clear()

    async def preview_resolution(
        self,
        conflict: Conflict,
    ) -> dict:
        """
        Preview how a conflict would be resolved without applying.

        Args:
            conflict: Conflict to preview

        Returns:
            Preview information dictionary
        """
        strategy = self._strategies.get(conflict.conflict_type)
        strategy_name = strategy.__class__.__name__ if strategy else "Unknown"

        return {
            "conflict_id": conflict.id,
            "conflict_type": conflict.conflict_type.value,
            "strategy": strategy_name,
            "auto_resolvable": conflict.auto_resolvable,
            "suggested_resolution": conflict.suggested_resolution,
        }

    # Legacy methods for backward compatibility
    async def resolve(
        self,
        conflict: dict[str, Any],
    ) -> str:
        """
        Resolve a conflict (legacy method).

        Args:
            conflict: Conflict dictionary.

        Returns:
            Resolution description.
        """
        conflict_type = conflict.get("conflict_type", "merge")
        file_path = conflict.get("file_path", "unknown")

        logger.info(f"Resolving {conflict_type} conflict in {file_path}")

        # Select strategy
        strategy = self._legacy_strategies.get(conflict_type)
        if not strategy:
            strategy = self._legacy_strategies["merge"]  # Default

        try:
            # Create a minimal Conflict object for the strategy
            minimal_conflict = Conflict(
                id="legacy",
                conflict_type=ConflictType.FILE_WRITE,
                severity=ConflictSeverity.MEDIUM,
                description=conflict.get("description", ""),
                affected_tasks=[
                    conflict.get("session_a_id", ""),
                    conflict.get("session_b_id", ""),
                ],
                affected_files=[file_path],
            )

            result = await strategy.resolve(minimal_conflict, self.workspace)
            logger.info(f"Resolved conflict in {file_path}")
            return result.actions_taken[0] if result.actions_taken else f"Resolved {file_path}"

        except Exception as e:
            logger.error(f"Failed to resolve conflict in {file_path}: {e}")
            raise ValueError(f"Cannot resolve conflict: {e}") from e

    async def resolve_all_legacy(
        self,
        conflicts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Attempt to resolve all conflicts (legacy method).

        Args:
            conflicts: List of conflict dictionaries.

        Returns:
            List of conflicts with resolution status updated.
        """
        logger.info(f"Resolving {len(conflicts)} conflicts")

        resolved = []
        for conflict in conflicts:
            try:
                resolution = await self.resolve(conflict)
                conflict["resolution"] = resolution
                conflict["resolved"] = True
            except ValueError:
                conflict["resolution"] = None
                conflict["resolved"] = False

            resolved.append(conflict)

        success_count = sum(1 for c in resolved if c.get("resolved"))
        logger.info(f"Resolved {success_count}/{len(conflicts)} conflicts")

        return resolved


async def resolve_conflict(conflict: dict[str, Any]) -> str:
    """Convenience function to resolve a conflict (legacy).

    Args:
        conflict: Conflict dictionary.

    Returns:
        Resolution string.
    """
    resolver = ConflictResolver()
    return await resolver.resolve(conflict)
