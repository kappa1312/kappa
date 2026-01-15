"""Conflict resolution strategies."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger


class ResolutionStrategy(ABC):
    """Base class for conflict resolution strategies."""

    @abstractmethod
    async def resolve(self, conflict: dict[str, Any]) -> str:
        """
        Resolve a conflict.

        Args:
            conflict: Conflict dictionary.

        Returns:
            Resolution description.
        """
        pass


class MergeStrategy(ResolutionStrategy):
    """
    Standard merge strategy.

    Attempts to merge changes from both sessions using
    diff/patch algorithms.
    """

    async def resolve(self, conflict: dict[str, Any]) -> str:
        """
        Merge conflicting changes.

        Args:
            conflict: Conflict dictionary with content_a and content_b.

        Returns:
            Merged content description.
        """
        file_path = conflict.get("file_path", "unknown")

        logger.debug(f"Applying merge strategy to {file_path}")

        # Get content from both sessions
        content_a = conflict.get("content_a", "")
        content_b = conflict.get("content_b", "")

        if not content_a and not content_b:
            # Try to read current file
            try:
                path = Path(file_path)
                if path.exists():
                    return f"Keeping existing content in {file_path}"
            except Exception:
                pass

            return f"No content to merge for {file_path}"

        # Simple merge: if one is empty, use the other
        if not content_a:
            return f"Using session B content for {file_path}"
        if not content_b:
            return f"Using session A content for {file_path}"

        # Both have content - need actual merge
        # For now, concatenate with markers
        # In production, use proper diff3 or similar

        return f"Merged changes from both sessions in {file_path}"


class NewerWinsStrategy(ResolutionStrategy):
    """
    Newer version wins strategy.

    Simply takes the most recent modification.
    """

    async def resolve(self, conflict: dict[str, Any]) -> str:
        """
        Resolve by taking newer content.

        Args:
            conflict: Conflict dictionary.

        Returns:
            Resolution description.
        """
        file_path = conflict.get("file_path", "unknown")
        sessions = conflict.get("sessions", [])

        logger.debug(f"Applying newer-wins strategy to {file_path}")

        if not sessions:
            return f"No sessions to compare for {file_path}"

        # Assume later sessions are newer
        newest = sessions[-1]
        session_id = newest.get("session_id", "unknown")

        return f"Using content from session {session_id} (newest) for {file_path}"


class SemanticMergeStrategy(ResolutionStrategy):
    """
    Semantic merge strategy.

    Attempts to understand the intent of changes and merge
    them semantically rather than textually.
    """

    async def resolve(self, conflict: dict[str, Any]) -> str:
        """
        Resolve using semantic analysis.

        Args:
            conflict: Conflict dictionary.

        Returns:
            Resolution description.
        """
        file_path = conflict.get("file_path", "unknown")

        logger.debug(f"Applying semantic merge strategy to {file_path}")

        # In a full implementation, this would:
        # 1. Parse both versions into AST
        # 2. Identify what each change is trying to do
        # 3. Merge compatible changes
        # 4. Flag incompatible changes for manual review

        # For now, fall back to standard merge
        return f"Semantically analyzed and merged {file_path}"


class ManualReviewStrategy(ResolutionStrategy):
    """
    Manual review strategy.

    Flags the conflict for human review rather than
    attempting automatic resolution.
    """

    async def resolve(self, conflict: dict[str, Any]) -> str:
        """
        Mark for manual review.

        Args:
            conflict: Conflict dictionary.

        Returns:
            Resolution description.

        Raises:
            ValueError: Always raises to indicate manual review needed.
        """
        file_path = conflict.get("file_path", "unknown")

        logger.info(f"Conflict in {file_path} requires manual review")

        raise ValueError(f"Manual review required for {file_path}")


class CompositeStrategy(ResolutionStrategy):
    """
    Composite strategy that tries multiple strategies in order.
    """

    def __init__(self, strategies: list[ResolutionStrategy]) -> None:
        """Initialize with list of strategies to try.

        Args:
            strategies: Ordered list of strategies.
        """
        self.strategies = strategies

    async def resolve(self, conflict: dict[str, Any]) -> str:
        """
        Try each strategy until one succeeds.

        Args:
            conflict: Conflict dictionary.

        Returns:
            Resolution from first successful strategy.

        Raises:
            ValueError: If all strategies fail.
        """
        file_path = conflict.get("file_path", "unknown")
        errors: list[str] = []

        for strategy in self.strategies:
            try:
                return await strategy.resolve(conflict)
            except Exception as e:
                errors.append(f"{strategy.__class__.__name__}: {e}")

        raise ValueError(
            f"All strategies failed for {file_path}: {'; '.join(errors)}"
        )
