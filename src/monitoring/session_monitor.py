"""Session monitoring - track active Claude sessions."""

from datetime import datetime
from typing import Any

from loguru import logger

from src.core.state import SessionInfo, SessionStatus


class SessionMonitor:
    """
    Monitor and track Claude sessions.

    Provides real-time visibility into session status,
    resource usage, and performance metrics.

    Example:
        >>> monitor = SessionMonitor()
        >>> monitor.register_session(session_info)
        >>> status = monitor.get_status("session-123")
    """

    def __init__(self) -> None:
        """Initialize the session monitor."""
        self._sessions: dict[str, SessionInfo] = {}
        self._history: list[dict[str, Any]] = []

    def register_session(self, session: SessionInfo) -> None:
        """
        Register a new session for monitoring.

        Args:
            session: SessionInfo to track.
        """
        self._sessions[session.id] = session
        logger.debug(f"Registered session: {session.id}")

    def unregister_session(self, session_id: str) -> None:
        """
        Unregister a session.

        Args:
            session_id: Session ID to remove.
        """
        if session_id in self._sessions:
            session = self._sessions.pop(session_id)
            self._history.append({
                "session_id": session_id,
                "status": session.status.value,
                "started_at": session.started_at.isoformat(),
                "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            })
            logger.debug(f"Unregistered session: {session_id}")

    def update_status(
        self,
        session_id: str,
        status: SessionStatus,
        error: str | None = None,
    ) -> None:
        """
        Update session status.

        Args:
            session_id: Session ID.
            status: New status.
            error: Optional error message.
        """
        if session_id in self._sessions:
            self._sessions[session_id].status = status
            if error:
                self._sessions[session_id].error = error
            if status in (SessionStatus.COMPLETED, SessionStatus.FAILED):
                self._sessions[session_id].completed_at = datetime.utcnow()

    def get_status(self, session_id: str) -> SessionInfo | None:
        """
        Get session status.

        Args:
            session_id: Session ID.

        Returns:
            SessionInfo if found.
        """
        return self._sessions.get(session_id)

    def get_all_active(self) -> list[SessionInfo]:
        """
        Get all active sessions.

        Returns:
            List of active SessionInfo objects.
        """
        return [
            s for s in self._sessions.values()
            if s.status in (SessionStatus.STARTING, SessionStatus.RUNNING)
        ]

    def get_summary(self) -> dict[str, Any]:
        """
        Get monitoring summary.

        Returns:
            Summary dictionary with counts and status.
        """
        statuses = {}
        for session in self._sessions.values():
            status = session.status.value
            statuses[status] = statuses.get(status, 0) + 1

        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len(self.get_all_active()),
            "status_breakdown": statuses,
            "history_count": len(self._history),
        }

    def get_session_metrics(self, session_id: str) -> dict[str, Any]:
        """
        Get metrics for a specific session.

        Args:
            session_id: Session ID.

        Returns:
            Metrics dictionary.
        """
        session = self._sessions.get(session_id)
        if not session:
            return {}

        duration = 0.0
        if session.started_at:
            end_time = session.completed_at or datetime.utcnow()
            duration = (end_time - session.started_at).total_seconds()

        return {
            "session_id": session_id,
            "status": session.status.value,
            "duration_seconds": duration,
            "has_error": session.error is not None,
            **session.metrics,
        }

    def check_timeouts(self, timeout_seconds: int = 3600) -> list[str]:
        """
        Check for timed out sessions.

        Args:
            timeout_seconds: Timeout threshold.

        Returns:
            List of timed out session IDs.
        """
        now = datetime.utcnow()
        timed_out = []

        for session_id, session in self._sessions.items():
            if session.status not in (SessionStatus.STARTING, SessionStatus.RUNNING):
                continue

            if session.started_at:
                elapsed = (now - session.started_at).total_seconds()
                if elapsed > timeout_seconds:
                    timed_out.append(session_id)

        return timed_out

    def mark_timeout(self, session_id: str) -> None:
        """
        Mark a session as timed out.

        Args:
            session_id: Session ID.
        """
        self.update_status(
            session_id,
            SessionStatus.TIMEOUT,
            error="Session timed out",
        )
        logger.warning(f"Session {session_id} marked as timed out")
