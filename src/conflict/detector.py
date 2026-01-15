"""Conflict detection - identify overlapping modifications."""

from collections import defaultdict
from typing import Any

from loguru import logger


class ConflictDetector:
    """
    Detect conflicts between session outputs.

    Analyzes task results to identify when multiple sessions
    have modified the same files or created conflicting changes.

    Example:
        >>> detector = ConflictDetector()
        >>> conflicts = await detector.detect(task_results)
        >>> len(conflicts)
        2
    """

    def __init__(self) -> None:
        """Initialize the conflict detector."""
        pass

    async def detect(
        self,
        task_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Detect conflicts in task results.

        Args:
            task_results: List of TaskResult dictionaries.

        Returns:
            List of conflict dictionaries.

        Example:
            >>> results = [
            ...     {"session_id": "a", "files_modified": ["src/main.py"]},
            ...     {"session_id": "b", "files_modified": ["src/main.py"]},
            ... ]
            >>> conflicts = await detector.detect(results)
            >>> conflicts[0]["file_path"]
            'src/main.py'
        """
        logger.info(f"Detecting conflicts in {len(task_results)} task results")

        # Build file -> sessions mapping
        file_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for result in task_results:
            session_id = result.get("session_id", "")
            files_modified = result.get("files_modified", [])

            for file_path in files_modified:
                file_sessions[file_path].append({
                    "session_id": session_id,
                    "task_id": result.get("task_id", ""),
                    "output": result.get("output", ""),
                })

        # Find conflicts (files modified by multiple sessions)
        conflicts: list[dict[str, Any]] = []

        for file_path, sessions in file_sessions.items():
            if len(sessions) > 1:
                conflict = self._create_conflict(file_path, sessions)
                conflicts.append(conflict)

        logger.info(f"Detected {len(conflicts)} conflicts")
        return conflicts

    def _create_conflict(
        self,
        file_path: str,
        sessions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a conflict dictionary.

        Args:
            file_path: Path to the conflicting file.
            sessions: List of sessions that modified the file.

        Returns:
            Conflict dictionary.
        """
        # Determine conflict type
        conflict_type = self._determine_conflict_type(file_path, sessions)

        # Use first two sessions for conflict (simplified)
        session_a = sessions[0]
        session_b = sessions[1] if len(sessions) > 1 else sessions[0]

        return {
            "file_path": file_path,
            "session_a_id": session_a["session_id"],
            "session_b_id": session_b["session_id"],
            "conflict_type": conflict_type,
            "description": f"File {file_path} modified by {len(sessions)} sessions",
            "sessions": sessions,
        }

    def _determine_conflict_type(
        self,
        file_path: str,
        sessions: list[dict[str, Any]],
    ) -> str:
        """Determine the type of conflict.

        Args:
            file_path: Path to the file.
            sessions: Sessions that modified the file.

        Returns:
            Conflict type string.
        """
        # Simple heuristics for conflict type

        # If file has specific patterns, might be semantic
        if any(ext in file_path for ext in [".py", ".ts", ".js"]):
            return "semantic"

        # Default to merge conflict
        return "merge"

    async def detect_semantic_conflicts(
        self,
        task_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Detect semantic conflicts (e.g., conflicting implementations).

        More advanced conflict detection that looks at the actual
        changes rather than just file overlap.

        Args:
            task_results: List of TaskResult dictionaries.

        Returns:
            List of semantic conflict dictionaries.
        """
        logger.debug("Detecting semantic conflicts")

        # This would require more sophisticated analysis
        # For now, return empty list
        # Future: Use Claude to analyze semantic compatibility

        return []


async def detect_conflicts(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convenience function to detect conflicts.

    Args:
        results: Task results to analyze.

    Returns:
        List of detected conflicts.
    """
    detector = ConflictDetector()
    return await detector.detect(results)
