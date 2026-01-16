"""
Metrics API Routes.

System and project metrics for monitoring and analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import Integer, func, select

from src.knowledge.database import get_db_session
from src.knowledge.models import Conflict, Project, Session, Task

router = APIRouter()


# ============================================================================
# Response Models
# ============================================================================


class SystemMetrics(BaseModel):
    """System-wide metrics."""

    total_projects: int
    active_projects: int
    completed_projects: int
    failed_projects: int
    total_tasks: int
    total_sessions: int
    total_conflicts: int
    resolved_conflicts: int


class ProjectMetrics(BaseModel):
    """Metrics for a specific project."""

    project_id: str
    task_metrics: dict[str, int]
    wave_metrics: list[dict[str, Any]]
    session_metrics: dict[str, Any]
    conflict_metrics: dict[str, int]
    timing_metrics: dict[str, float]


class ExecutionMetrics(BaseModel):
    """Execution timing metrics."""

    avg_task_duration_seconds: float
    avg_session_duration_seconds: float
    avg_project_duration_seconds: float
    tasks_per_hour: float
    parallelism_factor: float


class PerformanceMetrics(BaseModel):
    """Performance statistics."""

    active_connections: int
    cache_stats: dict[str, Any]
    database_status: str
    uptime_seconds: float


# ============================================================================
# Module State
# ============================================================================

_start_time = datetime.now()


# ============================================================================
# Routes
# ============================================================================


@router.get("/", response_model=SystemMetrics)
async def get_system_metrics() -> SystemMetrics:
    """
    Get system-wide metrics.

    Returns:
        System metrics summary.
    """
    async with get_db_session() as db:
        # Project counts
        project_query = select(
            func.count(Project.id).label("total"),
            func.sum(func.cast(Project.status == "running", type_=Integer)).label("active"),
            func.sum(func.cast(Project.status == "completed", type_=Integer)).label("completed"),
            func.sum(func.cast(Project.status == "failed", type_=Integer)).label("failed"),
        )
        project_result = await db.execute(project_query)
        project_counts = project_result.first()

        # Task count
        task_result = await db.execute(select(func.count(Task.id)))
        total_tasks = task_result.scalar() or 0

        # Session count
        session_result = await db.execute(select(func.count(Session.id)))
        total_sessions = session_result.scalar() or 0

        # Conflict counts
        conflict_query = select(
            func.count(Conflict.id).label("total"),
            func.sum(func.cast(Conflict.resolved_at.isnot(None), type_=Integer)).label("resolved"),
        )
        conflict_result = await db.execute(conflict_query)
        conflict_counts = conflict_result.first()

        return SystemMetrics(
            total_projects=project_counts.total or 0,
            active_projects=project_counts.active or 0,
            completed_projects=project_counts.completed or 0,
            failed_projects=project_counts.failed or 0,
            total_tasks=total_tasks,
            total_sessions=total_sessions,
            total_conflicts=conflict_counts.total or 0,
            resolved_conflicts=conflict_counts.resolved or 0,
        )


@router.get("/project/{project_id}", response_model=ProjectMetrics)
async def get_project_metrics(project_id: str) -> ProjectMetrics:
    """
    Get metrics for a specific project.

    Args:
        project_id: The project UUID.

    Returns:
        Project-specific metrics.
    """
    async with get_db_session() as db:
        # Task metrics
        task_query = select(
            func.count(Task.id).label("total"),
            func.sum(func.cast(Task.status == "completed", type_=Integer)).label("completed"),
            func.sum(func.cast(Task.status == "failed", type_=Integer)).label("failed"),
            func.sum(func.cast(Task.status == "running", type_=Integer)).label("running"),
            func.sum(func.cast(Task.status == "pending", type_=Integer)).label("pending"),
            func.sum(func.cast(Task.status == "skipped", type_=Integer)).label("skipped"),
        ).where(Task.project_id == project_id)
        task_result = await db.execute(task_query)
        task_counts = task_result.first()

        task_metrics = {
            "total": task_counts.total or 0,
            "completed": task_counts.completed or 0,
            "failed": task_counts.failed or 0,
            "running": task_counts.running or 0,
            "pending": task_counts.pending or 0,
            "skipped": task_counts.skipped or 0,
        }

        # Wave metrics
        wave_query = (
            select(
                Task.wave,
                func.count(Task.id).label("total"),
                func.sum(func.cast(Task.status == "completed", type_=Integer)).label("completed"),
            )
            .where(Task.project_id == project_id)
            .group_by(Task.wave)
            .order_by(Task.wave)
        )
        wave_result = await db.execute(wave_query)
        wave_rows = wave_result.all()

        wave_metrics = [
            {
                "wave": row.wave,
                "total_tasks": row.total,
                "completed_tasks": row.completed or 0,
                "progress": ((row.completed or 0) / row.total * 100) if row.total > 0 else 0,
            }
            for row in wave_rows
        ]

        # Session metrics
        session_query = select(
            func.count(Session.id).label("total"),
            func.sum(func.cast(Session.status == "completed", type_=Integer)).label("completed"),
            func.sum(func.cast(Session.status == "failed", type_=Integer)).label("failed"),
            func.avg(
                func.extract(
                    "epoch",
                    Session.completed_at - Session.started_at,
                )
            ).label("avg_duration"),
        ).where(Session.project_id == project_id)
        session_result = await db.execute(session_query)
        session_counts = session_result.first()

        session_metrics = {
            "total": session_counts.total or 0 if session_counts else 0,
            "completed": session_counts.completed or 0 if session_counts else 0,
            "failed": session_counts.failed or 0 if session_counts else 0,
            "avg_duration_seconds": float(session_counts.avg_duration or 0) if session_counts else 0.0,
        }

        # Conflict metrics
        conflict_query = select(
            func.count(Conflict.id).label("total"),
            func.sum(func.cast(Conflict.resolved_at.isnot(None), type_=Integer)).label("resolved"),
        ).where(Conflict.project_id == project_id)
        conflict_result = await db.execute(conflict_query)
        conflict_counts = conflict_result.first()

        conflict_metrics = {
            "total": conflict_counts.total or 0,
            "resolved": conflict_counts.resolved or 0,
            "unresolved": (conflict_counts.total or 0) - (conflict_counts.resolved or 0),
        }

        # Timing metrics
        project_query = select(
            Project.created_at,
            Project.completed_at,
        ).where(Project.id == project_id)
        project_result = await db.execute(project_query)
        project_row = project_result.first()

        timing_metrics = {}
        if project_row:
            if project_row.completed_at and project_row.created_at:
                duration = (project_row.completed_at - project_row.created_at).total_seconds()
                timing_metrics["total_duration_seconds"] = duration
            else:
                timing_metrics["total_duration_seconds"] = 0

        return ProjectMetrics(
            project_id=project_id,
            task_metrics=task_metrics,
            wave_metrics=wave_metrics,
            session_metrics=session_metrics,
            conflict_metrics=conflict_metrics,
            timing_metrics=timing_metrics,
        )


@router.get("/execution", response_model=ExecutionMetrics)
async def get_execution_metrics(
    since: datetime | None = Query(None, description="Only include data after this time"),
) -> ExecutionMetrics:
    """
    Get execution timing metrics.

    Args:
        since: Optional time filter.

    Returns:
        Execution timing metrics.
    """
    async with get_db_session() as db:
        # Build base filters
        task_filter: Any = Task.completed_at.isnot(None)
        session_filter: Any = Session.completed_at.isnot(None)
        project_filter: Any = Project.completed_at.isnot(None)

        if since:
            task_filter = task_filter & (Task.completed_at >= since)
            session_filter = session_filter & (Session.completed_at >= since)
            project_filter = project_filter & (Project.completed_at >= since)

        # Average task duration
        task_query = select(
            func.avg(func.extract("epoch", Task.completed_at - Task.started_at)).label(
                "avg_duration"
            ),
            func.count(Task.id).label("count"),
        ).where(task_filter & Task.started_at.isnot(None))
        task_result = await db.execute(task_query)
        task_row = task_result.first()

        avg_task_duration = float(task_row.avg_duration or 0) if task_row else 0.0

        # Average session duration
        session_query = select(
            func.avg(func.extract("epoch", Session.completed_at - Session.started_at)).label(
                "avg_duration"
            ),
        ).where(session_filter)
        session_result = await db.execute(session_query)
        session_row = session_result.first()

        avg_session_duration = float(session_row.avg_duration or 0) if session_row else 0.0

        # Average project duration
        project_query = select(
            func.avg(func.extract("epoch", Project.completed_at - Project.created_at)).label(
                "avg_duration"
            ),
        ).where(project_filter)
        project_result = await db.execute(project_query)
        project_row = project_result.first()

        avg_project_duration = float(project_row.avg_duration or 0) if project_row else 0.0

        # Tasks per hour (based on last hour or since filter)
        time_window = since or (datetime.now() - timedelta(hours=1))
        hourly_query = select(func.count(Task.id)).where(
            Task.completed_at.isnot(None) & (Task.completed_at >= time_window)
        )
        hourly_result = await db.execute(hourly_query)
        hourly_tasks = hourly_result.scalar() or 0

        hours_elapsed = (datetime.now() - time_window).total_seconds() / 3600
        tasks_per_hour = hourly_tasks / hours_elapsed if hours_elapsed > 0 else 0

        # Parallelism factor (average tasks running concurrently)
        # Simplified: average wave size
        wave_size_query = select(
            func.count(Task.id).label("wave_size"),
        ).group_by(Task.project_id, Task.wave)
        wave_size_result = await db.execute(wave_size_query)
        wave_sizes = [row.wave_size for row in wave_size_result.all()]
        parallelism = sum(wave_sizes) / len(wave_sizes) if wave_sizes else 1

        return ExecutionMetrics(
            avg_task_duration_seconds=avg_task_duration,
            avg_session_duration_seconds=avg_session_duration,
            avg_project_duration_seconds=avg_project_duration,
            tasks_per_hour=tasks_per_hour,
            parallelism_factor=parallelism,
        )


@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics() -> PerformanceMetrics:
    """
    Get performance statistics.

    Returns:
        Performance metrics.
    """
    from src.api.websocket import ws_manager
    from src.knowledge.database import health_check

    # Check database
    db_healthy = await health_check()

    # Calculate uptime
    uptime = (datetime.now() - _start_time).total_seconds()

    return PerformanceMetrics(
        active_connections=ws_manager.connection_count,
        cache_stats={
            "note": "Cache statistics would be populated from PerformanceOptimizer",
            "enabled": True,
        },
        database_status="healthy" if db_healthy else "unhealthy",
        uptime_seconds=uptime,
    )


@router.get("/summary")
async def get_metrics_summary() -> dict[str, Any]:
    """
    Get a combined summary of all metrics.

    Returns:
        Combined metrics summary.
    """
    system = await get_system_metrics()
    execution = await get_execution_metrics()
    performance = await get_performance_metrics()

    return {
        "system": system.model_dump(),
        "execution": execution.model_dump(),
        "performance": performance.model_dump(),
        "generated_at": datetime.now().isoformat(),
    }
