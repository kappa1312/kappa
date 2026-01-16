"""
Projects API Routes.

CRUD operations for Kappa projects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Integer, func, select
from sqlalchemy.sql.expression import cast

from src.knowledge.database import get_db_session
from src.knowledge.models import Conflict, Project, Session, Task

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class ProjectResponse(BaseModel):
    """Project response model."""

    id: str
    name: str
    status: str
    progress: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    running_tasks: int
    pending_tasks: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class ProjectDetailResponse(ProjectResponse):
    """Detailed project response with additional fields."""

    specification: str
    project_path: str
    config: dict[str, Any]
    final_output: str | None
    error: str | None


class ProjectCreate(BaseModel):
    """Project creation request model."""

    name: str
    requirements: str
    workspace_path: str | None = None


class WaveResponse(BaseModel):
    """Wave response model."""

    id: int
    status: str
    total_tasks: int
    completed_tasks: int
    tasks: list[dict[str, Any]]


class ChatMessage(BaseModel):
    """Chat history message model."""

    role: str
    content: str
    timestamp: str


# ============================================================================
# Routes
# ============================================================================


@router.get("/", response_model=list[ProjectResponse])
async def list_projects(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[ProjectResponse]:
    """
    List all projects with optional filtering.

    Args:
        status: Optional status filter.
        limit: Maximum number of results.
        offset: Number of results to skip.

    Returns:
        List of project summaries.
    """
    async with get_db_session() as db:
        query = select(Project)

        if status:
            query = query.where(Project.status == status)

        query = query.order_by(Project.created_at.desc()).offset(offset).limit(limit)

        result = await db.execute(query)
        projects = result.scalars().all()

        responses = []
        for project in projects:
            # Get task counts
            task_query = select(
                func.count(Task.id).label("total"),
                func.sum(func.cast(Task.status == "completed", type_=Integer)).label("completed"),
                func.sum(func.cast(Task.status == "failed", type_=Integer)).label("failed"),
                func.sum(func.cast(Task.status == "running", type_=Integer)).label("running"),
                func.sum(func.cast(Task.status == "pending", type_=Integer)).label("pending"),
            ).where(Task.project_id == project.id)

            task_result = await db.execute(task_query)
            counts = task_result.first()

            total = counts.total or 0
            completed = counts.completed or 0
            failed = counts.failed or 0
            running = counts.running or 0
            pending = counts.pending or 0

            responses.append(
                ProjectResponse(
                    id=project.id,
                    name=project.name,
                    status=project.status,
                    progress=(completed / total * 100) if total > 0 else 0,
                    total_tasks=total,
                    completed_tasks=completed,
                    failed_tasks=failed,
                    running_tasks=running,
                    pending_tasks=pending,
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                    completed_at=project.completed_at,
                )
            )

        return responses


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(project_id: str) -> ProjectDetailResponse:
    """
    Get single project by ID.

    Args:
        project_id: The project UUID.

    Returns:
        Detailed project information.

    Raises:
        HTTPException: If project not found.
    """
    async with get_db_session() as db:
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get task counts
        task_query = select(
            func.count(Task.id).label("total"),
            func.sum(func.cast(Task.status == "completed", type_=Integer)).label("completed"),
            func.sum(func.cast(Task.status == "failed", type_=Integer)).label("failed"),
            func.sum(func.cast(Task.status == "running", type_=Integer)).label("running"),
            func.sum(func.cast(Task.status == "pending", type_=Integer)).label("pending"),
        ).where(Task.project_id == project.id)

        task_result = await db.execute(task_query)
        counts = task_result.first()

        total = counts.total or 0
        completed = counts.completed or 0
        failed = counts.failed or 0
        running = counts.running or 0
        pending = counts.pending or 0

        return ProjectDetailResponse(
            id=project.id,
            name=project.name,
            status=project.status,
            progress=(completed / total * 100) if total > 0 else 0,
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            running_tasks=running,
            pending_tasks=pending,
            created_at=project.created_at,
            updated_at=project.updated_at,
            completed_at=project.completed_at,
            specification=project.specification,
            project_path=project.project_path,
            config=project.config or {},
            final_output=project.final_output,
            error=project.error,
        )


@router.get("/{project_id}/waves", response_model=list[WaveResponse])
async def get_project_waves(project_id: str) -> list[WaveResponse]:
    """
    Get waves for a project.

    Args:
        project_id: The project UUID.

    Returns:
        List of waves with their tasks.

    Raises:
        HTTPException: If project not found.
    """
    async with get_db_session() as db:
        # Verify project exists
        project_result = await db.execute(select(Project.id).where(Project.id == project_id))
        if not project_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Project not found")

        # Get all tasks for project
        result = await db.execute(
            select(Task).where(Task.project_id == project_id).order_by(Task.wave, Task.name)
        )
        tasks = result.scalars().all()

        # Group by wave
        waves: dict[int, WaveResponse] = {}

        for task in tasks:
            wave_num = task.wave or 0

            if wave_num not in waves:
                waves[wave_num] = WaveResponse(
                    id=wave_num,
                    status="pending",
                    total_tasks=0,
                    completed_tasks=0,
                    tasks=[],
                )

            waves[wave_num].tasks.append(
                {
                    "id": task.id,
                    "name": task.name,
                    "status": task.status,
                    "category": task.category,
                }
            )
            waves[wave_num].total_tasks += 1

            if task.status == "completed":
                waves[wave_num].completed_tasks += 1

        # Determine wave status
        for wave in waves.values():
            if wave.completed_tasks == wave.total_tasks:
                wave.status = "completed"
            elif wave.completed_tasks > 0:
                wave.status = "executing"
            else:
                # Check if any task is running
                if any(t["status"] == "running" for t in wave.tasks):
                    wave.status = "executing"

        return sorted(waves.values(), key=lambda w: w.id)


@router.get("/{project_id}/chat-history", response_model=list[ChatMessage])
async def get_chat_history(project_id: str) -> list[ChatMessage]:
    """
    Get chat/log history for project.

    Args:
        project_id: The project UUID.

    Returns:
        List of chat messages representing project history.

    Raises:
        HTTPException: If project not found.
    """
    async with get_db_session() as db:
        # Get project
        result = await db.execute(select(Project).where(Project.id == project_id))
        project = result.scalar_one_or_none()

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        history: list[ChatMessage] = []

        # Add initial requirement
        history.append(
            ChatMessage(
                role="user",
                content=f"Build project: {project.name}",
                timestamp=project.created_at.isoformat(),
            )
        )

        # Add specification parsing
        history.append(
            ChatMessage(
                role="system",
                content="Parsing requirements...",
                timestamp=project.created_at.isoformat(),
            )
        )

        # Get tasks ordered by creation time
        task_result = await db.execute(
            select(Task).where(Task.project_id == project_id).order_by(Task.created_at)
        )
        tasks = task_result.scalars().all()

        for task in tasks:
            if task.started_at:
                history.append(
                    ChatMessage(
                        role="system",
                        content=f"Starting: {task.name}",
                        timestamp=task.started_at.isoformat(),
                    )
                )

            if task.completed_at:
                status_emoji = {
                    "completed": "OK",
                    "failed": "ERR",
                    "skipped": "SKIP",
                }.get(task.status, "")
                history.append(
                    ChatMessage(
                        role="system",
                        content=f"[{status_emoji}] {task.name}",
                        timestamp=task.completed_at.isoformat(),
                    )
                )

        # Add conflicts if any
        conflict_result = await db.execute(
            select(Conflict).where(Conflict.project_id == project_id)
        )
        conflicts = conflict_result.scalars().all()

        for conflict in conflicts:
            history.append(
                ChatMessage(
                    role="system",
                    content=f"Conflict detected: {conflict.file_path}",
                    timestamp=conflict.created_at.isoformat(),
                )
            )
            if conflict.resolved_at:
                history.append(
                    ChatMessage(
                        role="system",
                        content=f"Conflict resolved: {conflict.file_path}",
                        timestamp=conflict.resolved_at.isoformat(),
                    )
                )

        # Add completion status
        if project.completed_at:
            if project.status == "completed":
                history.append(
                    ChatMessage(
                        role="assistant",
                        content="Project completed successfully!",
                        timestamp=project.completed_at.isoformat(),
                    )
                )
            elif project.status == "failed":
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=f"Project failed: {project.error}",
                        timestamp=project.completed_at.isoformat(),
                    )
                )

        return history


@router.get("/{project_id}/sessions", response_model=list[dict[str, Any]])
async def get_project_sessions(project_id: str) -> list[dict[str, Any]]:
    """
    Get all sessions for a project.

    Args:
        project_id: The project UUID.

    Returns:
        List of session summaries.

    Raises:
        HTTPException: If project not found.
    """
    async with get_db_session() as db:
        # Verify project exists
        project_result = await db.execute(select(Project.id).where(Project.id == project_id))
        if not project_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Project not found")

        result = await db.execute(
            select(Session).where(Session.project_id == project_id).order_by(Session.started_at)
        )
        sessions = result.scalars().all()

        return [
            {
                "id": s.id,
                "task_id": s.task_id,
                "status": s.status,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "files_modified": s.files_modified or [],
                "token_usage": s.token_usage or {},
            }
            for s in sessions
        ]


@router.get("/{project_id}/conflicts", response_model=list[dict[str, Any]])
async def get_project_conflicts(project_id: str) -> list[dict[str, Any]]:
    """
    Get all conflicts for a project.

    Args:
        project_id: The project UUID.

    Returns:
        List of conflict details.

    Raises:
        HTTPException: If project not found.
    """
    async with get_db_session() as db:
        # Verify project exists
        project_result = await db.execute(select(Project.id).where(Project.id == project_id))
        if not project_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Project not found")

        result = await db.execute(
            select(Conflict).where(Conflict.project_id == project_id).order_by(Conflict.created_at)
        )
        conflicts = result.scalars().all()

        return [
            {
                "id": c.id,
                "file_path": c.file_path,
                "conflict_type": c.conflict_type,
                "description": c.description,
                "session_a_id": c.session_a_id,
                "session_b_id": c.session_b_id,
                "resolved": c.resolved_at is not None,
                "resolved_by": c.resolved_by,
                "created_at": c.created_at.isoformat(),
                "resolved_at": c.resolved_at.isoformat() if c.resolved_at else None,
            }
            for c in conflicts
        ]
