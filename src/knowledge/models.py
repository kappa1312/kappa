"""SQLAlchemy ORM models for Kappa persistence."""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Project(Base):
    """Project model - represents a Kappa execution project."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    specification: Mapped[str] = mapped_column(Text, nullable=False)
    project_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum(
            "pending", "decomposing", "running", "resolving_conflicts",
            "completed", "failed",
            name="project_status",
        ),
        default="pending",
    )
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    final_output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    tasks: Mapped[list["Task"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )
    decisions: Mapped[list["Decision"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"Project(id={self.id}, name={self.name}, status={self.status})"


class Task(Base):
    """Task model - represents a single task in a project."""

    __tablename__ = "tasks"
    __table_args__ = (
        Index("ix_tasks_project_wave", "project_id", "wave"),
        Index("ix_tasks_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(50), default="business_logic")
    complexity: Mapped[str] = mapped_column(String(20), default="medium")
    wave: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        Enum("pending", "running", "completed", "failed", "skipped", name="task_status"),
        default="pending",
    )
    dependencies: Mapped[list] = mapped_column(JSON, default=list)
    file_targets: Mapped[list] = mapped_column(JSON, default=list)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="tasks")
    session: Mapped[Optional["Session"]] = relationship(
        back_populates="task",
        uselist=False,
    )

    def __repr__(self) -> str:
        return f"Task(id={self.id}, name={self.name}, wave={self.wave}, status={self.status})"


class Session(Base):
    """Session model - represents a Claude session."""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    task_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("tasks.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        Enum("starting", "running", "completed", "failed", "timeout", name="session_status"),
        default="starting",
    )
    output: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    files_modified: Mapped[list] = mapped_column(JSON, default=list)
    token_usage: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="sessions")
    task: Mapped[Optional["Task"]] = relationship(back_populates="session")
    context_snapshots: Mapped[list["ContextSnapshot"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"Session(id={self.id}, status={self.status})"


class ContextSnapshot(Base):
    """Context snapshot - stores context shared between sessions."""

    __tablename__ = "context_snapshots"
    __table_args__ = (
        Index("ix_context_session_type", "session_id", "context_type"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    context_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # "file_read", "decision", "discovery", etc.
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    session: Mapped["Session"] = relationship(back_populates="context_snapshots")

    def __repr__(self) -> str:
        return f"ContextSnapshot(id={self.id}, type={self.context_type}, key={self.key})"


class Decision(Base):
    """Decision model - stores architectural and design decisions."""

    __tablename__ = "decisions"
    __table_args__ = (
        Index("ix_decisions_project_category", "project_id", "category"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )  # "architecture", "technology", "pattern", etc.
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    alternatives_considered: Mapped[list] = mapped_column(JSON, default=list)
    made_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )  # session_id or "orchestrator"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="decisions")

    def __repr__(self) -> str:
        return f"Decision(id={self.id}, category={self.category})"


class Conflict(Base):
    """Conflict model - stores detected conflicts between sessions."""

    __tablename__ = "conflicts"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    session_a_id: Mapped[str] = mapped_column(String(36), nullable=False)
    session_b_id: Mapped[str] = mapped_column(String(36), nullable=False)
    conflict_type: Mapped[str] = mapped_column(
        String(50),
        default="merge",
    )  # "merge", "overwrite", "semantic"
    description: Mapped[str] = mapped_column(Text, nullable=False)
    content_a: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_b: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolved_by: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"Conflict(id={self.id}, file={self.file_path})"
