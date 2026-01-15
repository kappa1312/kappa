"""Conflict resolver - apply resolution strategies."""

from typing import Any

from loguru import logger

from src.conflict.strategies import (
    MergeStrategy,
    NewerWinsStrategy,
    ResolutionStrategy,
    SemanticMergeStrategy,
)


class ConflictResolver:
    """
    Resolve conflicts between session outputs.

    Applies appropriate resolution strategies based on
    conflict type and file characteristics.

    Example:
        >>> resolver = ConflictResolver()
        >>> resolution = await resolver.resolve(conflict)
        >>> print(resolution)
        'Merged changes from sessions a and b'
    """

    def __init__(self) -> None:
        """Initialize the conflict resolver."""
        self._strategies: dict[str, ResolutionStrategy] = {
            "merge": MergeStrategy(),
            "overwrite": NewerWinsStrategy(),
            "semantic": SemanticMergeStrategy(),
        }

    async def resolve(
        self,
        conflict: dict[str, Any],
    ) -> str:
        """
        Resolve a conflict.

        Args:
            conflict: Conflict dictionary.

        Returns:
            Resolution description.

        Raises:
            ValueError: If conflict cannot be resolved.

        Example:
            >>> conflict = {
            ...     "file_path": "src/main.py",
            ...     "conflict_type": "merge",
            ...     "session_a_id": "a",
            ...     "session_b_id": "b",
            ... }
            >>> resolution = await resolver.resolve(conflict)
        """
        conflict_type = conflict.get("conflict_type", "merge")
        file_path = conflict.get("file_path", "unknown")

        logger.info(f"Resolving {conflict_type} conflict in {file_path}")

        # Select strategy
        strategy = self._strategies.get(conflict_type)
        if not strategy:
            strategy = self._strategies["merge"]  # Default

        try:
            resolution = await strategy.resolve(conflict)
            logger.info(f"Resolved conflict in {file_path}")
            return resolution

        except Exception as e:
            logger.error(f"Failed to resolve conflict in {file_path}: {e}")
            raise ValueError(f"Cannot resolve conflict: {e}") from e

    async def resolve_all(
        self,
        conflicts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Attempt to resolve all conflicts.

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

    def register_strategy(
        self,
        conflict_type: str,
        strategy: ResolutionStrategy,
    ) -> None:
        """Register a custom resolution strategy.

        Args:
            conflict_type: Type of conflict this strategy handles.
            strategy: ResolutionStrategy instance.
        """
        self._strategies[conflict_type] = strategy
        logger.debug(f"Registered strategy for {conflict_type}")

    async def preview_resolution(
        self,
        conflict: dict[str, Any],
    ) -> str:
        """
        Preview how a conflict would be resolved without applying.

        Args:
            conflict: Conflict dictionary.

        Returns:
            Description of proposed resolution.
        """
        conflict_type = conflict.get("conflict_type", "merge")
        strategy = self._strategies.get(conflict_type, self._strategies["merge"])

        return f"Would apply {strategy.__class__.__name__} to {conflict.get('file_path')}"


async def resolve_conflict(conflict: dict[str, Any]) -> str:
    """Convenience function to resolve a conflict.

    Args:
        conflict: Conflict dictionary.

    Returns:
        Resolution string.
    """
    resolver = ConflictResolver()
    return await resolver.resolve(conflict)
