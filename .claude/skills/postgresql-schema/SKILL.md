# PostgreSQL Schema Skill

## Activation

This skill activates when working on:
- Database schema design
- SQLAlchemy models
- Migrations
- Query optimization

## Kappa Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐
│    projects     │       │    sessions     │
├─────────────────┤       ├─────────────────┤
│ id (PK)         │───┐   │ id (PK)         │
│ name            │   │   │ project_id (FK) │───┐
│ spec            │   │   │ task_id (FK)    │   │
│ status          │   │   │ status          │   │
│ created_at      │   │   │ started_at      │   │
│ updated_at      │   └──→│ completed_at    │   │
└─────────────────┘       │ error           │   │
                          └─────────────────┘   │
                                    │           │
┌─────────────────┐       ┌─────────────────┐   │
│     tasks       │       │context_snapshots│   │
├─────────────────┤       ├─────────────────┤   │
│ id (PK)         │←──────│ id (PK)         │   │
│ project_id (FK) │       │ session_id (FK) │←──┘
│ name            │       │ context_type    │
│ description     │       │ content         │
│ wave            │       │ created_at      │
│ status          │       └─────────────────┘
│ dependencies    │
│ created_at      │       ┌─────────────────┐
└─────────────────┘       │    conflicts    │
                          ├─────────────────┤
┌─────────────────┐       │ id (PK)         │
│    decisions    │       │ project_id (FK) │
├─────────────────┤       │ file_path       │
│ id (PK)         │       │ session_a_id    │
│ project_id (FK) │       │ session_b_id    │
│ category        │       │ status          │
│ decision        │       │ resolution      │
│ rationale       │       │ created_at      │
│ created_at      │       │ resolved_at     │
└─────────────────┘       └─────────────────┘
```

### SQLAlchemy Models

```python
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    specification: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("pending", "running", "completed", "failed", name="project_status"),
        default="pending"
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    tasks: Mapped[list["Task"]] = relationship(back_populates="project")
    sessions: Mapped[list["Session"]] = relationship(back_populates="project")
    decisions: Mapped[list["Decision"]] = relationship(back_populates="project")


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_project_wave", "project_id", "wave"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    wave: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(
        Enum("pending", "running", "completed", "failed", name="task_status"),
        default="pending"
    )
    dependencies: Mapped[list[str]] = mapped_column(JSON, default=list)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="tasks")
    session: Mapped[Optional["Session"]] = relationship(back_populates="task")


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False
    )
    task_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True
    )
    status: Mapped[str] = mapped_column(
        Enum("starting", "running", "completed", "failed", name="session_status"),
        default="starting"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="sessions")
    task: Mapped[Optional["Task"]] = relationship(back_populates="session")
    context_snapshots: Mapped[list["ContextSnapshot"]] = relationship(
        back_populates="session"
    )
```

### Async Database Operations

```python
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Create async engine
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/kappa_db",
    echo=False,
    pool_size=10,
    max_overflow=20,
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Context manager for sessions
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### Common Query Patterns

```python
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Get project with all tasks
async def get_project_with_tasks(
    session: AsyncSession,
    project_id: str
) -> Project | None:
    stmt = (
        select(Project)
        .options(selectinload(Project.tasks))
        .where(Project.id == project_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()

# Get tasks by wave
async def get_tasks_by_wave(
    session: AsyncSession,
    project_id: str,
    wave: int
) -> list[Task]:
    stmt = (
        select(Task)
        .where(Task.project_id == project_id)
        .where(Task.wave == wave)
        .order_by(Task.created_at)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())

# Bulk update task statuses
async def update_task_statuses(
    session: AsyncSession,
    task_ids: list[str],
    status: str
) -> None:
    from sqlalchemy import update

    stmt = (
        update(Task)
        .where(Task.id.in_(task_ids))
        .values(status=status)
    )
    await session.execute(stmt)
```

### Indexes Strategy

```sql
-- Primary queries optimized
CREATE INDEX ix_tasks_project_wave ON tasks(project_id, wave);
CREATE INDEX ix_sessions_project_status ON sessions(project_id, status);
CREATE INDEX ix_decisions_project_category ON decisions(project_id, category);

-- Time-based queries
CREATE INDEX ix_sessions_started_at ON sessions(started_at DESC);
CREATE INDEX ix_context_created_at ON context_snapshots(created_at DESC);

-- Full-text search on decisions
CREATE INDEX ix_decisions_content_gin ON decisions
    USING gin(to_tsvector('english', decision || ' ' || rationale));
```

## Best Practices

1. **Always use async** - Use `asyncpg` and async session makers
2. **Use UUIDs** - Better for distributed systems than auto-increment
3. **Add indexes early** - Profile queries and add indexes proactively
4. **Use JSON for flexible data** - Config, metrics, and results
5. **Cascade deletes** - Clean up related records automatically
6. **Use Enums** - For status fields with known values
7. **Timestamp everything** - created_at, updated_at for debugging
