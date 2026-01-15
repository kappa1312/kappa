"""Context manager - cross-session context sharing."""

from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.models import ContextSnapshot, Decision


class ContextManager:
    """
    Manage cross-session context sharing.

    Enables sessions to share discoveries, decisions, and context
    through a PostgreSQL-backed knowledge base.

    Example:
        >>> async with get_db_session() as db:
        ...     manager = ContextManager(db)
        ...     await manager.store_context(
        ...         session_id="abc",
        ...         context_type="decision",
        ...         key="auth_method",
        ...         content="Use JWT tokens"
        ...     )
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize context manager.

        Args:
            session: Async database session.
        """
        self.session = session

    async def store_context(
        self,
        session_id: str,
        context_type: str,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContextSnapshot:
        """
        Store a context snapshot.

        Args:
            session_id: Source session ID.
            context_type: Type of context (decision, discovery, etc).
            key: Unique key for this context.
            content: Context content.
            metadata: Optional metadata.

        Returns:
            Created ContextSnapshot.

        Example:
            >>> snapshot = await manager.store_context(
            ...     session_id="123",
            ...     context_type="file_read",
            ...     key="src/models/user.py",
            ...     content="<file contents>"
            ... )
        """
        logger.debug(f"Storing context: {context_type}/{key}")

        snapshot = ContextSnapshot(
            session_id=session_id,
            context_type=context_type,
            key=key,
            content=content,
            snapshot_metadata=metadata or {},
        )

        self.session.add(snapshot)
        await self.session.flush()

        return snapshot

    async def get_context(
        self,
        context_type: str,
        key: str,
    ) -> ContextSnapshot | None:
        """
        Get the latest context snapshot for a key.

        Args:
            context_type: Type of context.
            key: Context key.

        Returns:
            Most recent ContextSnapshot or None.
        """
        stmt = (
            select(ContextSnapshot)
            .where(ContextSnapshot.context_type == context_type)
            .where(ContextSnapshot.key == key)
            .order_by(ContextSnapshot.created_at.desc())
            .limit(1)
        )

        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all_context(
        self,
        context_type: str | None = None,
        session_id: str | None = None,
    ) -> list[ContextSnapshot]:
        """
        Get all context snapshots matching filters.

        Args:
            context_type: Optional type filter.
            session_id: Optional session filter.

        Returns:
            List of matching ContextSnapshots.
        """
        stmt = select(ContextSnapshot)

        if context_type:
            stmt = stmt.where(ContextSnapshot.context_type == context_type)
        if session_id:
            stmt = stmt.where(ContextSnapshot.session_id == session_id)

        stmt = stmt.order_by(ContextSnapshot.created_at.desc())

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def store_decision(
        self,
        project_id: str,
        category: str,
        decision: str,
        rationale: str | None = None,
        alternatives: list[str] | None = None,
        made_by: str | None = None,
    ) -> Decision:
        """
        Store an architectural/design decision.

        Args:
            project_id: Project ID.
            category: Decision category.
            decision: The decision made.
            rationale: Why this decision was made.
            alternatives: Other options considered.
            made_by: Who/what made the decision.

        Returns:
            Created Decision.

        Example:
            >>> decision = await manager.store_decision(
            ...     project_id="proj-123",
            ...     category="architecture",
            ...     decision="Use REST API",
            ...     rationale="Better tooling support",
            ...     alternatives=["GraphQL", "gRPC"]
            ... )
        """
        logger.info(f"Storing decision: {category} - {decision[:50]}...")

        dec = Decision(
            project_id=project_id,
            category=category,
            decision=decision,
            rationale=rationale,
            alternatives_considered=alternatives or [],
            made_by=made_by,
        )

        self.session.add(dec)
        await self.session.flush()

        return dec

    async def get_decisions(
        self,
        project_id: str,
        category: str | None = None,
    ) -> list[Decision]:
        """
        Get decisions for a project.

        Args:
            project_id: Project ID.
            category: Optional category filter.

        Returns:
            List of Decision objects.
        """
        stmt = select(Decision).where(Decision.project_id == project_id)

        if category:
            stmt = stmt.where(Decision.category == category)

        stmt = stmt.order_by(Decision.created_at.desc())

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_file_context(
        self,
        file_path: str,
    ) -> str | None:
        """
        Get cached file content from context.

        Args:
            file_path: Path to the file.

        Returns:
            File content if cached, None otherwise.
        """
        snapshot = await self.get_context("file_read", file_path)
        return snapshot.content if snapshot else None

    async def store_file_context(
        self,
        session_id: str,
        file_path: str,
        content: str,
    ) -> ContextSnapshot:
        """
        Store file content in context.

        Args:
            session_id: Source session ID.
            file_path: Path to the file.
            content: File content.

        Returns:
            Created ContextSnapshot.
        """
        return await self.store_context(
            session_id=session_id,
            context_type="file_read",
            key=file_path,
            content=content,
            metadata={"length": len(content)},
        )

    async def get_discovery(
        self,
        key: str,
    ) -> str | None:
        """
        Get a discovery from context.

        Args:
            key: Discovery key.

        Returns:
            Discovery content if found.
        """
        snapshot = await self.get_context("discovery", key)
        return snapshot.content if snapshot else None

    async def store_discovery(
        self,
        session_id: str,
        key: str,
        content: str,
    ) -> ContextSnapshot:
        """
        Store a discovery in context.

        Discoveries are findings about the codebase or project
        that should be shared across sessions.

        Args:
            session_id: Source session ID.
            key: Discovery key.
            content: Discovery content.

        Returns:
            Created ContextSnapshot.
        """
        return await self.store_context(
            session_id=session_id,
            context_type="discovery",
            key=key,
            content=content,
        )

    async def build_context_summary(
        self,
        project_id: str,
    ) -> str:
        """
        Build a summary of all context for a project.

        Args:
            project_id: Project ID.

        Returns:
            Formatted context summary string.
        """
        lines = ["# Project Context Summary", ""]

        # Get decisions
        decisions = await self.get_decisions(project_id)
        if decisions:
            lines.append("## Decisions")
            for dec in decisions[:10]:  # Limit to recent
                lines.append(f"- **{dec.category}**: {dec.decision}")
            lines.append("")

        # Get discoveries
        discoveries = await self.get_all_context(context_type="discovery")
        if discoveries:
            lines.append("## Discoveries")
            for disc in discoveries[:10]:
                lines.append(f"- **{disc.key}**: {disc.content[:100]}...")
            lines.append("")

        return "\n".join(lines)
