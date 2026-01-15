"""Base session class for Claude interactions."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.core.state import SessionInfo, SessionStatus, TaskResult


class SessionConfig(BaseModel):
    """Configuration for a Claude session."""

    model_config = ConfigDict(frozen=True)

    max_turns: int = Field(default=50, ge=1)
    timeout_seconds: int = Field(default=3600, ge=60)
    working_directory: str = Field(default=".")
    system_prompt: str | None = Field(default=None)
    auto_accept_edits: bool = Field(default=True)
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    )


class BaseSession(ABC):
    """
    Abstract base class for Claude sessions.

    Defines the interface for different session implementations
    (terminal, API, etc.).

    Example:
        >>> class MySession(BaseSession):
        ...     async def execute(self, prompt: str) -> TaskResult:
        ...         # Implementation
        ...         pass
    """

    def __init__(
        self,
        config: SessionConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        """Initialize the session.

        Args:
            config: Session configuration.
            session_id: Optional session ID (generated if not provided).
        """
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
        """
        Execute a prompt and return the result.

        Args:
            prompt: The prompt/task to execute.

        Returns:
            TaskResult with execution outcome.
        """
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
        """Record a message in the session history.

        Args:
            role: Message role (user, assistant, system).
            content: Message content.
        """
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def get_history(self) -> list[dict[str, Any]]:
        """Get the session message history.

        Returns:
            List of message dictionaries.
        """
        return self._messages.copy()

    def update_status(self, status: SessionStatus, error: str | None = None) -> None:
        """Update session status.

        Args:
            status: New status.
            error: Optional error message.
        """
        self.info.status = status
        if error:
            self.info.error = error
        if status in (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.TIMEOUT):
            self.info.completed_at = datetime.utcnow()

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(id={self.id}, status={self.info.status})"
