"""Integration tests for session management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.sessions.base import SessionConfig
from src.sessions.router import SessionRouter
from src.sessions.terminal import TerminalSession


class TestSessionConfig:
    """Tests for SessionConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SessionConfig()

        assert config.max_turns == 50
        assert config.timeout_seconds == 3600
        assert config.auto_accept_edits is True
        assert "Read" in config.allowed_tools
        assert "Write" in config.allowed_tools

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SessionConfig(
            max_turns=10,
            timeout_seconds=600,
            working_directory="/custom/path",
            system_prompt="Custom prompt",
        )

        assert config.max_turns == 10
        assert config.timeout_seconds == 600
        assert config.working_directory == "/custom/path"


class TestTerminalSession:
    """Tests for TerminalSession."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self) -> None:
        """Test session start and close."""
        session = TerminalSession()

        await session.start()
        assert session.is_active

        await session.close()
        assert not session.is_active

    @pytest.mark.asyncio
    @patch("src.sessions.terminal.TerminalSession._execute_with_client")
    async def test_execute_success(
        self,
        mock_execute: AsyncMock,
    ) -> None:
        """Test successful execution."""
        mock_execute.return_value = "Task completed successfully"

        session = TerminalSession(task_id="task-1")
        await session.start()

        result = await session.execute("Test prompt")

        assert result.success
        assert result.task_id == "task-1"
        await session.close()

    @pytest.mark.asyncio
    @patch("src.sessions.terminal.TerminalSession._execute_with_client")
    async def test_execute_timeout(
        self,
        mock_execute: AsyncMock,
    ) -> None:
        """Test execution timeout."""
        # Simulate timeout by raising TimeoutError
        mock_execute.side_effect = TimeoutError("Mock timeout")

        config = SessionConfig(timeout_seconds=60)  # Minimum valid timeout
        session = TerminalSession(config=config)
        await session.start()

        result = await session.execute("Slow task")

        assert not result.success
        assert "timed out" in result.error.lower()
        await session.close()

    def test_record_message(self) -> None:
        """Test message recording."""
        session = TerminalSession()

        session.record_message("user", "Hello")
        session.record_message("assistant", "Hi there")

        history = session.get_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestSessionRouter:
    """Tests for SessionRouter."""

    @pytest.fixture
    def router(self) -> SessionRouter:
        """Create router for testing."""
        return SessionRouter(max_sessions=3, timeout=60)

    @pytest.mark.asyncio
    @patch("src.sessions.router.TerminalSession")
    async def test_execute_tasks(
        self,
        MockSession: MagicMock,
        router: SessionRouter,
    ) -> None:
        """Test executing multiple tasks."""
        # Setup mock session
        mock_session = AsyncMock()
        mock_session.id = "mock-session"
        mock_session.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                task_id="task-1",
                session_id="mock-session",
            )
        )
        mock_session.start = AsyncMock()
        mock_session.close = AsyncMock()
        MockSession.return_value = mock_session

        tasks = [
            {"id": "task-1", "name": "Task 1", "description": "Test task"},
        ]
        state = {"project_path": "/test"}

        results = await router.execute_tasks(tasks, state)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_cancel_all(self, router: SessionRouter) -> None:
        """Test cancelling all sessions."""
        await router.cancel_all()

        assert router.active_count == 0

    def test_get_session_not_found(self, router: SessionRouter) -> None:
        """Test getting non-existent session."""
        session = router.get_session("non-existent")

        assert session is None

    @pytest.mark.asyncio
    async def test_concurrency_limit(self) -> None:
        """Test that concurrency is limited."""
        router = SessionRouter(max_sessions=2)

        # The semaphore should limit concurrent sessions
        assert router._semaphore._value == 2
