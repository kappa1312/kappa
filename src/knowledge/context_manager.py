"""
Context manager - cross-session context sharing for parallel execution.

This module provides comprehensive context sharing capabilities between
parallel Claude sessions, enabling type definitions, imports, discoveries,
and decisions to flow between tasks executing in different waves.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.knowledge.models import ContextSnapshot, Decision

# =============================================================================
# CONTEXT TYPES
# =============================================================================


class ContextType:
    """Constants for context types."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    DISCOVERY = "discovery"
    DECISION = "decision"
    TYPE_DEFINITION = "type_definition"
    IMPORT = "import"
    EXPORT = "export"
    ERROR = "error"
    VALIDATION = "validation"
    WAVE_OUTPUT = "wave_output"
    TASK_OUTPUT = "task_output"


# =============================================================================
# IN-MEMORY CONTEXT (for fast access during execution)
# =============================================================================


@dataclass
class SharedContext:
    """
    In-memory shared context for fast access during parallel execution.

    This provides a fast, non-persistent context store that accumulates
    context during a single execution run. It's designed for high-frequency
    access during wave execution.

    Example:
        >>> ctx = SharedContext(project_id="proj-123")
        >>> ctx.add_type("User", "interface User { id: string; }")
        >>> ctx.add_export("src/types/user.ts", ["User", "UserRole"])
    """

    project_id: str
    wave_number: int = 0

    # Type definitions (name -> definition)
    types: dict[str, str] = field(default_factory=dict)

    # Exports (file_path -> list of export names)
    exports: dict[str, list[str]] = field(default_factory=dict)

    # Imports (module -> list of import names)
    imports: dict[str, list[str]] = field(default_factory=dict)

    # File contents cache (path -> content)
    file_cache: dict[str, str] = field(default_factory=dict)

    # Discoveries (key -> content)
    discoveries: dict[str, str] = field(default_factory=dict)

    # Decisions (category -> list of decisions)
    decisions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Task outputs (task_id -> output dict)
    task_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Wave outputs (wave_number -> list of task outputs)
    wave_outputs: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    # Errors (task_id -> error info)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_type(self, name: str, definition: str) -> None:
        """Add a type definition."""
        self.types[name] = definition
        logger.debug(f"[Context] Added type: {name}")

    def add_export(self, file_path: str, exports: list[str]) -> None:
        """Add exports from a file."""
        if file_path not in self.exports:
            self.exports[file_path] = []
        self.exports[file_path].extend(exports)
        logger.debug(f"[Context] Added exports from {file_path}: {exports}")

    def add_import(self, module: str, imports: list[str]) -> None:
        """Add available imports from a module."""
        if module not in self.imports:
            self.imports[module] = []
        self.imports[module].extend(imports)

    def cache_file(self, path: str, content: str) -> None:
        """Cache file content."""
        self.file_cache[path] = content

    def get_file(self, path: str) -> str | None:
        """Get cached file content."""
        return self.file_cache.get(path)

    def add_discovery(self, key: str, content: str) -> None:
        """Add a discovery."""
        self.discoveries[key] = content
        logger.debug(f"[Context] Added discovery: {key}")

    def add_decision(
        self,
        category: str,
        decision: str,
        rationale: str = "",
        made_by: str = "",
    ) -> None:
        """Add a decision."""
        if category not in self.decisions:
            self.decisions[category] = []
        self.decisions[category].append(
            {
                "decision": decision,
                "rationale": rationale,
                "made_by": made_by,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        logger.debug(f"[Context] Added decision ({category}): {decision[:50]}...")

    def record_task_output(
        self,
        task_id: str,
        output: dict[str, Any],
    ) -> None:
        """Record output from a completed task."""
        self.task_outputs[task_id] = output

        # Also add to wave outputs
        wave = output.get("wave_number", self.wave_number)
        if wave not in self.wave_outputs:
            self.wave_outputs[wave] = []
        self.wave_outputs[wave].append(output)

        # Extract and register exports
        if "files_created" in output:
            for file_path in output["files_created"]:
                if "exports" in output:
                    self.add_export(file_path, output["exports"].get(file_path, []))

        logger.debug(f"[Context] Recorded task output: {task_id}")

    def record_error(
        self,
        task_id: str,
        error_type: str,
        error_message: str,
        traceback: str = "",
    ) -> None:
        """Record an error from a task."""
        self.errors[task_id] = {
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_context_for_wave(self, wave_number: int) -> dict[str, Any]:
        """Get all context accumulated up to a specific wave."""
        context = {
            "types": self.types.copy(),
            "exports": self.exports.copy(),
            "imports": self.imports.copy(),
            "discoveries": self.discoveries.copy(),
            "decisions": {k: v.copy() for k, v in self.decisions.items()},
            "previous_waves": {},
        }

        # Add outputs from previous waves
        for w in range(wave_number):
            if w in self.wave_outputs:
                context["previous_waves"][w] = self.wave_outputs[w]

        return context

    def get_types_for_task(self, task_id: str, dependencies: list[str]) -> dict[str, str]:
        """Get type definitions relevant for a task based on its dependencies."""
        relevant_types: dict[str, str] = {}

        # Get types from dependent tasks
        for dep_id in dependencies:
            if dep_id in self.task_outputs:
                output = self.task_outputs[dep_id]
                if "types_exported" in output:
                    for type_name in output["types_exported"]:
                        if type_name in self.types:
                            relevant_types[type_name] = self.types[type_name]

        return relevant_types

    def get_imports_for_task(self, task_id: str, dependencies: list[str]) -> dict[str, list[str]]:
        """Get imports available for a task based on its dependencies."""
        relevant_imports: dict[str, list[str]] = {}

        for dep_id in dependencies:
            if dep_id in self.task_outputs:
                output = self.task_outputs[dep_id]
                if "files_created" in output:
                    for file_path in output["files_created"]:
                        if file_path in self.exports:
                            relevant_imports[file_path] = self.exports[file_path]

        return relevant_imports

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "wave_number": self.wave_number,
            "types": self.types,
            "exports": self.exports,
            "imports": self.imports,
            "discoveries": self.discoveries,
            "decisions": self.decisions,
            "task_outputs": self.task_outputs,
            "wave_outputs": self.wave_outputs,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedContext":
        """Create from dictionary."""
        ctx = cls(
            project_id=data.get("project_id", ""),
            wave_number=data.get("wave_number", 0),
        )
        ctx.types = data.get("types", {})
        ctx.exports = data.get("exports", {})
        ctx.imports = data.get("imports", {})
        ctx.discoveries = data.get("discoveries", {})
        ctx.decisions = data.get("decisions", {})
        ctx.task_outputs = data.get("task_outputs", {})
        ctx.wave_outputs = data.get("wave_outputs", {})
        ctx.errors = data.get("errors", {})
        return ctx


# =============================================================================
# DATABASE-BACKED CONTEXT MANAGER
# =============================================================================


class ContextManager:
    """
    Manage cross-session context sharing with database persistence.

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
        self._cache: dict[str, SharedContext] = {}

    def get_shared_context(self, project_id: str) -> SharedContext:
        """Get or create in-memory shared context for a project."""
        if project_id not in self._cache:
            self._cache[project_id] = SharedContext(project_id=project_id)
        return self._cache[project_id]

    # -------------------------------------------------------------------------
    # CONTEXT SNAPSHOT OPERATIONS
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # DECISION OPERATIONS
    # -------------------------------------------------------------------------

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

        # Also update in-memory context
        shared = self.get_shared_context(project_id)
        shared.add_decision(category, decision, rationale or "", made_by or "")

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

    # -------------------------------------------------------------------------
    # FILE CONTEXT OPERATIONS
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # DISCOVERY OPERATIONS
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # TYPE DEFINITION OPERATIONS
    # -------------------------------------------------------------------------

    async def store_type_definition(
        self,
        session_id: str,
        project_id: str,
        type_name: str,
        definition: str,
        source_file: str | None = None,
    ) -> ContextSnapshot:
        """
        Store a type/interface definition for sharing.

        Args:
            session_id: Source session ID.
            project_id: Project ID.
            type_name: Name of the type.
            definition: Type definition code.
            source_file: File where type is defined.

        Returns:
            Created ContextSnapshot.
        """
        snapshot = await self.store_context(
            session_id=session_id,
            context_type=ContextType.TYPE_DEFINITION,
            key=type_name,
            content=definition,
            metadata={"source_file": source_file},
        )

        # Also update in-memory context
        shared = self.get_shared_context(project_id)
        shared.add_type(type_name, definition)

        return snapshot

    async def get_type_definitions(
        self,
        project_id: str | None = None,
    ) -> dict[str, str]:
        """
        Get all type definitions.

        Args:
            project_id: Optional project filter.

        Returns:
            Dict of type_name -> definition.
        """
        # Try in-memory cache first
        if project_id and project_id in self._cache:
            return self._cache[project_id].types.copy()

        # Fall back to database
        snapshots = await self.get_all_context(context_type=ContextType.TYPE_DEFINITION)
        return {s.key: s.content for s in snapshots}

    # -------------------------------------------------------------------------
    # EXPORT/IMPORT OPERATIONS
    # -------------------------------------------------------------------------

    async def store_exports(
        self,
        session_id: str,
        project_id: str,
        file_path: str,
        exports: list[str],
    ) -> ContextSnapshot:
        """
        Store exports from a file.

        Args:
            session_id: Source session ID.
            project_id: Project ID.
            file_path: Path to the file.
            exports: List of exported names.

        Returns:
            Created ContextSnapshot.
        """
        snapshot = await self.store_context(
            session_id=session_id,
            context_type=ContextType.EXPORT,
            key=file_path,
            content=",".join(exports),
            metadata={"count": len(exports)},
        )

        # Update in-memory context
        shared = self.get_shared_context(project_id)
        shared.add_export(file_path, exports)

        return snapshot

    async def get_exports(
        self,
        project_id: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Get all exports.

        Returns:
            Dict of file_path -> list of export names.
        """
        if project_id and project_id in self._cache:
            return self._cache[project_id].exports.copy()

        snapshots = await self.get_all_context(context_type=ContextType.EXPORT)
        return {s.key: s.content.split(",") for s in snapshots}

    # -------------------------------------------------------------------------
    # TASK OUTPUT OPERATIONS
    # -------------------------------------------------------------------------

    async def store_task_output(
        self,
        session_id: str,
        project_id: str,
        task_id: str,
        output: dict[str, Any],
    ) -> ContextSnapshot:
        """
        Store output from a completed task.

        Args:
            session_id: Session ID.
            project_id: Project ID.
            task_id: Task ID.
            output: Task output dictionary.

        Returns:
            Created ContextSnapshot.
        """
        import json

        snapshot = await self.store_context(
            session_id=session_id,
            context_type=ContextType.TASK_OUTPUT,
            key=task_id,
            content=json.dumps(output),
            metadata={
                "wave_number": output.get("wave_number"),
                "success": output.get("success", False),
            },
        )

        # Update in-memory context
        shared = self.get_shared_context(project_id)
        shared.record_task_output(task_id, output)

        return snapshot

    async def get_task_output(
        self,
        task_id: str,
    ) -> dict[str, Any] | None:
        """
        Get output from a task.

        Args:
            task_id: Task ID.

        Returns:
            Task output dict if found.
        """
        import json

        snapshot = await self.get_context(ContextType.TASK_OUTPUT, task_id)
        if snapshot:
            return json.loads(snapshot.content)
        return None

    # -------------------------------------------------------------------------
    # WAVE CONTEXT OPERATIONS
    # -------------------------------------------------------------------------

    async def store_wave_output(
        self,
        session_id: str,
        project_id: str,
        wave_number: int,
        outputs: list[dict[str, Any]],
    ) -> ContextSnapshot:
        """
        Store outputs from a completed wave.

        Args:
            session_id: Session ID (can be "orchestrator").
            project_id: Project ID.
            wave_number: Wave number.
            outputs: List of task outputs from the wave.

        Returns:
            Created ContextSnapshot.
        """
        import json

        snapshot = await self.store_context(
            session_id=session_id,
            context_type=ContextType.WAVE_OUTPUT,
            key=f"wave_{wave_number}",
            content=json.dumps(outputs),
            metadata={
                "wave_number": wave_number,
                "task_count": len(outputs),
                "success_count": sum(1 for o in outputs if o.get("success", False)),
            },
        )

        # Update in-memory context
        shared = self.get_shared_context(project_id)
        shared.wave_outputs[wave_number] = outputs

        return snapshot

    async def get_wave_output(
        self,
        wave_number: int,
    ) -> list[dict[str, Any]] | None:
        """
        Get outputs from a wave.

        Args:
            wave_number: Wave number.

        Returns:
            List of task outputs if found.
        """
        import json

        snapshot = await self.get_context(ContextType.WAVE_OUTPUT, f"wave_{wave_number}")
        if snapshot:
            return json.loads(snapshot.content)
        return None

    # -------------------------------------------------------------------------
    # CONTEXT BUILDING
    # -------------------------------------------------------------------------

    async def build_context_for_task(
        self,
        project_id: str,
        task_id: str,
        dependencies: list[str],
        wave_number: int,
    ) -> dict[str, Any]:
        """
        Build complete context for a task based on its dependencies.

        Args:
            project_id: Project ID.
            task_id: Task ID.
            dependencies: List of dependency task IDs.
            wave_number: Current wave number.

        Returns:
            Dict containing all relevant context.
        """
        shared = self.get_shared_context(project_id)

        # Get types from dependencies
        types = shared.get_types_for_task(task_id, dependencies)

        # Get imports from dependencies
        imports = shared.get_imports_for_task(task_id, dependencies)

        # Get relevant decisions
        decisions = []
        for category, decs in shared.decisions.items():
            for dec in decs:
                decisions.append(
                    {
                        "category": category,
                        "decision": dec["decision"],
                        "rationale": dec.get("rationale", ""),
                    }
                )

        # Get outputs from previous waves
        previous_wave_outputs = {}
        for w in range(wave_number):
            if w in shared.wave_outputs:
                previous_wave_outputs[w] = shared.wave_outputs[w]

        return {
            "types": types,
            "imports": imports,
            "discoveries": shared.discoveries.copy(),
            "decisions": decisions,
            "previous_waves": previous_wave_outputs,
        }

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

        # Get type definitions
        types = await self.get_type_definitions(project_id)
        if types:
            lines.append("## Type Definitions")
            for name in list(types.keys())[:10]:
                lines.append(f"- `{name}`")
            lines.append("")

        # Get exports
        exports = await self.get_exports(project_id)
        if exports:
            lines.append("## Available Exports")
            for file_path, export_list in list(exports.items())[:10]:
                lines.append(f"- `{file_path}`: {', '.join(export_list[:5])}")
            lines.append("")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------

    def clear_cache(self, project_id: str | None = None) -> None:
        """
        Clear in-memory cache.

        Args:
            project_id: Optional project to clear. If None, clears all.
        """
        if project_id:
            self._cache.pop(project_id, None)
        else:
            self._cache.clear()
