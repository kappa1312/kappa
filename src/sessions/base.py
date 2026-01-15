"""
Base session manager interface for Kappa OS.

All session types (terminal, web, native) inherit from this.
This module provides the foundation for parallel Claude session execution.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# ENUMS
# =============================================================================


class SessionStatus(str, Enum):
    """Possible states for a session."""

    PENDING = "pending"
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class SessionPriority(int, Enum):
    """Priority levels for session scheduling."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class SessionEventType(str, Enum):
    """Types of session events for callbacks."""

    STARTED = "started"
    PROGRESS = "progress"
    OUTPUT = "output"
    ERROR = "error"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SessionResult:
    """Result of a completed session."""

    session_id: str
    task_id: str
    status: SessionStatus
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Output
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None

    # Files
    files_created: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)

    # Metrics
    memory_peak_mb: float | None = None
    cpu_time_seconds: float | None = None
    token_usage: dict[str, int] = field(default_factory=dict)

    # Error info
    error_message: str | None = None
    error_traceback: str | None = None

    def is_success(self) -> bool:
        """Check if session completed successfully."""
        return self.status == SessionStatus.COMPLETED and (
            self.return_code is None or self.return_code == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_time_seconds": self.cpu_time_seconds,
            "token_usage": self.token_usage,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
        }


@dataclass
class SessionEvent:
    """Event emitted during session lifecycle."""

    event_type: SessionEventType
    session_id: str
    task_id: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


# Type for event callbacks
SessionEventCallback = Callable[[SessionEvent], Any]


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SessionConfig(BaseModel):
    """Configuration for a Claude session."""

    model_config = ConfigDict(frozen=False)

    # Execution limits
    max_turns: int = Field(default=100, ge=1, description="Maximum conversation turns")
    timeout_seconds: int = Field(default=1800, ge=60, description="Session timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=5, ge=1, description="Delay between retries")

    # Resource limits
    memory_limit_mb: int | None = Field(default=None, description="Memory limit in MB")
    cpu_limit_percent: float | None = Field(default=None, ge=0, le=100, description="CPU limit")

    # Workspace
    working_directory: str = Field(default=".", description="Working directory path")

    # Claude configuration
    system_prompt: str | None = Field(default=None, description="Custom system prompt")
    auto_accept_edits: bool = Field(default=True, description="Auto-accept file edits")
    dangerously_skip_permissions: bool = Field(default=True, description="Skip permission prompts")

    # Environment
    environment: dict[str, str] = Field(default_factory=dict, description="Additional env vars")

    # Priority
    priority: SessionPriority = Field(
        default=SessionPriority.NORMAL, description="Session priority"
    )

    # Tools
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        description="Allowed tools for the session",
    )


class SessionInfo(BaseModel):
    """Runtime information about a session."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(description="Session UUID")
    task_id: str | None = Field(default=None, description="Associated task ID")
    status: SessionStatus = Field(default=SessionStatus.PENDING, description="Current status")

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = Field(default=None)
    completed_at: datetime | None = Field(default=None)

    # Process info
    pid: int | None = Field(default=None, description="Process ID")

    # Metrics
    memory_mb: float | None = Field(default=None)
    cpu_percent: float | None = Field(default=None)

    # Error
    error: str | None = Field(default=None)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate duration if available."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()


# =============================================================================
# BASE SESSION MANAGER
# =============================================================================


class BaseSessionManager(ABC):
    """
    Abstract base class for all session managers.

    Implementations:
    - TerminalSessionManager: Claude Code terminal sessions
    - WebSessionManager: Browser automation sessions (future)
    - NativeSessionManager: Native app automation (future)

    Example:
        >>> class MyManager(BaseSessionManager):
        ...     async def create_session(self, task_id, prompt, workspace, context, config):
        ...         # Implementation
        ...         return session_id
    """

    def __init__(self, max_concurrent: int = 10, default_config: SessionConfig | None = None):
        """
        Initialize the session manager.

        Args:
            max_concurrent: Maximum concurrent sessions.
            default_config: Default session configuration.
        """
        self.max_concurrent = max_concurrent
        self.default_config = default_config or SessionConfig()
        self.active_sessions: dict[str, SessionInfo] = {}
        self.completed_sessions: dict[str, SessionResult] = {}
        self.event_callbacks: list[SessionEventCallback] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

    def add_event_callback(self, callback: SessionEventCallback) -> None:
        """Register a callback for session events."""
        self.event_callbacks.append(callback)

    def remove_event_callback(self, callback: SessionEventCallback) -> None:
        """Remove a registered callback."""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)

    async def _emit_event(self, event: SessionEvent) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self.event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    @abstractmethod
    async def create_session(
        self,
        task_id: str,
        prompt: str,
        workspace: str,
        context: dict[str, Any],
        config: SessionConfig | None = None,
    ) -> str:
        """
        Create and start a new session.

        Args:
            task_id: Unique identifier for the task.
            prompt: The prompt/instructions for Claude.
            workspace: Working directory path.
            context: Additional context (imports, types, etc.).
            config: Session configuration overrides.

        Returns:
            session_id: Unique identifier for the created session.
        """
        pass

    @abstractmethod
    async def monitor_session(self, session_id: str) -> dict[str, Any]:
        """
        Get real-time status and progress of a session.

        Returns dict with:
            - status: SessionStatus
            - elapsed_seconds: float
            - progress_percent: Optional[float]
            - current_activity: Optional[str]
            - memory_mb: Optional[float]
            - cpu_percent: Optional[float]
        """
        pass

    @abstractmethod
    async def get_output(self, session_id: str) -> SessionResult:
        """
        Get the complete output of a session.

        Should only be called after session completes.
        """
        pass

    @abstractmethod
    async def send_input(self, session_id: str, input_text: str) -> bool:
        """
        Send input to a running session (for interactive sessions).

        Returns True if input was sent successfully.
        """
        pass

    @abstractmethod
    async def kill_session(self, session_id: str, force: bool = False) -> bool:
        """
        Terminate a running session.

        Args:
            session_id: Session to kill.
            force: If True, use SIGKILL instead of SIGTERM.

        Returns:
            True if session was killed successfully.
        """
        pass

    @abstractmethod
    async def wait_for_completion(
        self, session_id: str, timeout: int | None = None, poll_interval: float = 1.0
    ) -> SessionResult:
        """
        Wait for a session to complete.

        Args:
            session_id: Session to wait for.
            timeout: Override default timeout (None = use config).
            poll_interval: How often to check status.

        Returns:
            SessionResult with final status and output.
        """
        pass

    async def get_active_count(self) -> int:
        """Get number of currently active sessions."""
        async with self._lock:
            return len(self.active_sessions)

    async def get_all_active(self) -> list[str]:
        """Get list of all active session IDs."""
        async with self._lock:
            return list(self.active_sessions.keys())

    async def kill_all(self, force: bool = False) -> dict[str, bool]:
        """Kill all active sessions."""
        results: dict[str, bool] = {}
        session_ids = await self.get_all_active()

        for session_id in session_ids:
            try:
                results[session_id] = await self.kill_session(session_id, force)
            except Exception as e:
                logger.error(f"Failed to kill session {session_id}: {e}")
                results[session_id] = False

        return results

    async def cleanup(self) -> None:
        """Cleanup resources (call on shutdown)."""
        await self.kill_all(force=True)
        self.active_sessions.clear()
        self.event_callbacks.clear()

    async def acquire_slot(self) -> None:
        """Acquire a session slot (blocks if at capacity)."""
        await self._semaphore.acquire()

    def release_slot(self) -> None:
        """Release a session slot."""
        self._semaphore.release()

    def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Get session info by ID."""
        return self.active_sessions.get(session_id)

    def get_completed_result(self, session_id: str) -> SessionResult | None:
        """Get completed session result by ID."""
        return self.completed_sessions.get(session_id)


# =============================================================================
# LEGACY SUPPORT
# =============================================================================

# Re-export for backward compatibility with existing code
from src.core.state import (
    TaskResult,
)


class BaseSession(ABC):
    """
    Legacy abstract base class for Claude sessions.

    Kept for backward compatibility with existing code.
    New code should use BaseSessionManager instead.
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the session."""
        self.id = session_id or str(uuid4())
        self.config = config or SessionConfig()
        self.info = SessionInfo(
            id=self.id,
            status=SessionStatus.STARTING,
        )
        self._started_at: datetime | None = None
        self._messages: list[dict[str, Any]] = []

    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.info.status in (SessionStatus.STARTING, SessionStatus.RUNNING)

    @abstractmethod
    async def execute(self, prompt: str) -> TaskResult:
        """Execute a prompt and return the result."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the session and cleanup resources."""
        pass

    async def start(self) -> None:
        """Start the session."""
        self._started_at = datetime.utcnow()
        self.info.status = SessionStatus.RUNNING
        self.info.started_at = self._started_at

    async def stop(self) -> None:
        """Stop the session."""
        self.info.status = SessionStatus.COMPLETED
        self.info.completed_at = datetime.utcnow()

    def record_message(self, role: str, content: str) -> None:
        """Record a message in the session history."""
        self._messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def get_history(self) -> list[dict[str, Any]]:
        """Get the session message history."""
        return self._messages.copy()

    def update_status(self, status: SessionStatus, error: str | None = None) -> None:
        """Update session status."""
        self.info.status = status
        if error:
            self.info.error = error
        if status in (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT):
            self.info.completed_at = datetime.utcnow()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.id}, status={self.info.status})"
