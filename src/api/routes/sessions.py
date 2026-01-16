"""
Sessions API Routes.

Query and manage Claude sessions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from src.knowledge.database import get_db_session
from src.knowledge.models import ContextSnapshot, Session, Task

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class SessionResponse(BaseModel):
    """Session response model."""

    id: str
    project_id: str
    task_id: str | None
    status: str
    files_modified: list[str]
    token_usage: dict[str, Any]
    started_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class SessionDetailResponse(SessionResponse):
    """Detailed session response with output."""

    output: str | None
    error: str | None
    metrics: dict[str, Any]


class ContextSnapshotResponse(BaseModel):
    """Context snapshot response model."""

    id: str
    context_type: str
    key: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime


# ============================================================================
# Routes
# ============================================================================


@router.get("/", response_model=list[SessionResponse])
async def list_sessions(
    project_id: str = Query(..., description="Project ID to filter sessions"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[SessionResponse]:
    """
    List sessions for a project.

    Args:
        project_id: Required project ID.
        status: Optional status filter.
        limit: Maximum number of results.
        offset: Number of results to skip.

    Returns:
        List of sessions.
    """
    async with get_db_session() as db:
        query = select(Session).where(Session.project_id == project_id)

        if status:
            query = query.where(Session.status == status)

        query = query.order_by(Session.started_at.desc()).offset(offset).limit(limit)

        result = await db.execute(query)
        sessions = result.scalars().all()

        return [
            SessionResponse(
                id=s.id,
                project_id=s.project_id,
                task_id=s.task_id,
                status=s.status,
                files_modified=s.files_modified or [],
                token_usage=s.token_usage or {},
                started_at=s.started_at,
                completed_at=s.completed_at,
            )
            for s in sessions
        ]


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str) -> SessionDetailResponse:
    """
    Get single session by ID.

    Args:
        session_id: The session UUID.

    Returns:
        Detailed session information.

    Raises:
        HTTPException: If session not found.
    """
    async with get_db_session() as db:
        result = await db.execute(select(Session).where(Session.id == session_id))
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionDetailResponse(
            id=session.id,
            project_id=session.project_id,
            task_id=session.task_id,
            status=session.status,
            files_modified=session.files_modified or [],
            token_usage=session.token_usage or {},
            started_at=session.started_at,
            completed_at=session.completed_at,
            output=session.output,
            error=session.error,
            metrics=session.metrics or {},
        )


@router.get("/{session_id}/output")
async def get_session_output(
    session_id: str,
    tail: int = Query(None, ge=1, le=10000, description="Only return last N lines"),
) -> dict[str, Any]:
    """
    Get full session output.

    Args:
        session_id: The session UUID.
        tail: Optional - only return last N lines.

    Returns:
        Session output text.

    Raises:
        HTTPException: If session not found.
    """
    async with get_db_session() as db:
        result = await db.execute(
            select(Session.id, Session.output, Session.status).where(Session.id == session_id)
        )
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        output = row.output or ""

        if tail and output:
            lines = output.split("\n")
            output = "\n".join(lines[-tail:])

        return {
            "session_id": row.id,
            "status": row.status,
            "output": output,
            "line_count": len(output.split("\n")) if output else 0,
        }


@router.get("/{session_id}/context", response_model=list[ContextSnapshotResponse])
async def get_session_context(
    session_id: str,
    context_type: str | None = Query(None, description="Filter by context type"),
) -> list[ContextSnapshotResponse]:
    """
    Get context snapshots for a session.

    Args:
        session_id: The session UUID.
        context_type: Optional type filter (e.g., "file_read", "decision").

    Returns:
        List of context snapshots.

    Raises:
        HTTPException: If session not found.
    """
    async with get_db_session() as db:
        # Verify session exists
        session_result = await db.execute(select(Session.id).where(Session.id == session_id))
        if not session_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Session not found")

        query = select(ContextSnapshot).where(ContextSnapshot.session_id == session_id)

        if context_type:
            query = query.where(ContextSnapshot.context_type == context_type)

        query = query.order_by(ContextSnapshot.created_at)

        result = await db.execute(query)
        snapshots = result.scalars().all()

        return [
            ContextSnapshotResponse(
                id=s.id,
                context_type=s.context_type,
                key=s.key,
                content=s.content,
                metadata=s.snapshot_metadata or {},
                created_at=s.created_at,
            )
            for s in snapshots
        ]


@router.get("/{session_id}/task")
async def get_session_task(session_id: str) -> dict[str, Any]:
    """
    Get the task associated with a session.

    Args:
        session_id: The session UUID.

    Returns:
        Task details.

    Raises:
        HTTPException: If session or task not found.
    """
    async with get_db_session() as db:
        # Get session
        session_result = await db.execute(select(Session.task_id).where(Session.id == session_id))
        row = session_result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        if not row.task_id:
            raise HTTPException(status_code=404, detail="No task associated with this session")

        # Get task
        task_result = await db.execute(select(Task).where(Task.id == row.task_id))
        task = task_result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "id": task.id,
            "project_id": task.project_id,
            "name": task.name,
            "description": task.description,
            "category": task.category,
            "complexity": task.complexity,
            "wave": task.wave,
            "status": task.status,
            "dependencies": task.dependencies or [],
            "file_targets": task.file_targets or [],
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        }
