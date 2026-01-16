"""
Tasks API Routes.

Query and manage tasks within projects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from src.knowledge.database import get_db_session
from src.knowledge.models import Session, Task

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class TaskResponse(BaseModel):
    """Task response model."""

    id: str
    project_id: str
    name: str
    description: str
    category: str
    complexity: str
    wave: int
    status: str
    dependencies: list[str]
    file_targets: list[str]
    session_id: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class TaskDetailResponse(TaskResponse):
    """Detailed task response with result and error."""

    result: dict[str, Any] | None
    error: str | None


class TaskLogEntry(BaseModel):
    """Task log entry model."""

    timestamp: str
    level: str
    message: str


# ============================================================================
# Routes
# ============================================================================


@router.get("/", response_model=list[TaskResponse])
async def list_tasks(
    project_id: str = Query(..., description="Project ID to filter tasks"),
    wave: int | None = Query(None, description="Filter by wave number"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[TaskResponse]:
    """
    List tasks for a project.

    Args:
        project_id: Required project ID.
        wave: Optional wave filter.
        status: Optional status filter.
        limit: Maximum number of results.
        offset: Number of results to skip.

    Returns:
        List of tasks.
    """
    async with get_db_session() as db:
        query = select(Task).where(Task.project_id == project_id)

        if wave is not None:
            query = query.where(Task.wave == wave)

        if status:
            query = query.where(Task.status == status)

        query = query.order_by(Task.wave, Task.name).offset(offset).limit(limit)

        result = await db.execute(query)
        tasks = result.scalars().all()

        responses = []
        for task in tasks:
            # Get session ID if exists
            session_query = select(Session.id).where(Session.task_id == task.id)
            session_result = await db.execute(session_query)
            session_id = session_result.scalar_one_or_none()

            responses.append(
                TaskResponse(
                    id=task.id,
                    project_id=task.project_id,
                    name=task.name,
                    description=task.description,
                    category=task.category,
                    complexity=task.complexity,
                    wave=task.wave,
                    status=task.status,
                    dependencies=task.dependencies or [],
                    file_targets=task.file_targets or [],
                    session_id=session_id,
                    created_at=task.created_at,
                    started_at=task.started_at,
                    completed_at=task.completed_at,
                )
            )

        return responses


@router.get("/{task_id}", response_model=TaskDetailResponse)
async def get_task(task_id: str) -> TaskDetailResponse:
    """
    Get single task by ID.

    Args:
        task_id: The task UUID.

    Returns:
        Detailed task information.

    Raises:
        HTTPException: If task not found.
    """
    async with get_db_session() as db:
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Get session ID if exists
        session_query = select(Session.id).where(Session.task_id == task.id)
        session_result = await db.execute(session_query)
        session_id = session_result.scalar_one_or_none()

        return TaskDetailResponse(
            id=task.id,
            project_id=task.project_id,
            name=task.name,
            description=task.description,
            category=task.category,
            complexity=task.complexity,
            wave=task.wave,
            status=task.status,
            dependencies=task.dependencies or [],
            file_targets=task.file_targets or [],
            session_id=session_id,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result,
            error=task.error,
        )


@router.get("/{task_id}/logs", response_model=list[TaskLogEntry])
async def get_task_logs(task_id: str) -> list[TaskLogEntry]:
    """
    Get execution logs for a task.

    Args:
        task_id: The task UUID.

    Returns:
        List of log entries.

    Raises:
        HTTPException: If task not found.
    """
    async with get_db_session() as db:
        # Get task
        result = await db.execute(select(Task).where(Task.id == task_id))
        task = result.scalar_one_or_none()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Get session output if exists
        session_query = select(Session).where(Session.task_id == task_id)
        session_result = await db.execute(session_query)
        session = session_result.scalar_one_or_none()

        logs: list[TaskLogEntry] = []

        # Add task lifecycle events
        if task.created_at:
            logs.append(
                TaskLogEntry(
                    timestamp=task.created_at.isoformat(),
                    level="INFO",
                    message=f"Task created: {task.name}",
                )
            )

        if task.started_at:
            logs.append(
                TaskLogEntry(
                    timestamp=task.started_at.isoformat(),
                    level="INFO",
                    message="Task execution started",
                )
            )

        # Add session output as logs
        if session and session.output:
            # Split output into lines and create log entries
            lines = session.output.split("\n")
            for line in lines[:100]:  # Limit to first 100 lines
                if line.strip():
                    logs.append(
                        TaskLogEntry(
                            timestamp=session.started_at.isoformat() if session.started_at else "",
                            level="DEBUG",
                            message=line.strip(),
                        )
                    )

        if task.completed_at:
            level = "INFO" if task.status == "completed" else "ERROR"
            message = f"Task {task.status}"
            if task.error:
                message += f": {task.error}"

            logs.append(
                TaskLogEntry(
                    timestamp=task.completed_at.isoformat(),
                    level=level,
                    message=message,
                )
            )

        return logs


@router.get("/{task_id}/session", response_model=dict[str, Any])
async def get_task_session(task_id: str) -> dict[str, Any]:
    """
    Get the session associated with a task.

    Args:
        task_id: The task UUID.

    Returns:
        Session details.

    Raises:
        HTTPException: If task or session not found.
    """
    async with get_db_session() as db:
        # Verify task exists
        task_result = await db.execute(select(Task.id).where(Task.id == task_id))
        if not task_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Task not found")

        # Get session
        result = await db.execute(select(Session).where(Session.task_id == task_id))
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(status_code=404, detail="No session found for this task")

        return {
            "id": session.id,
            "project_id": session.project_id,
            "task_id": session.task_id,
            "status": session.status,
            "output": session.output,
            "error": session.error,
            "files_modified": session.files_modified or [],
            "token_usage": session.token_usage or {},
            "metrics": session.metrics or {},
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        }
