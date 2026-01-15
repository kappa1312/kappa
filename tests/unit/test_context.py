"""Unit tests for context management."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.knowledge.context_manager import ContextManager


class TestContextManager:
    """Tests for ContextManager."""

    @pytest.mark.asyncio
    async def test_store_context(self, mock_db_session: AsyncMock) -> None:
        """Test storing context."""
        manager = ContextManager(mock_db_session)

        snapshot = await manager.store_context(
            session_id="session-1",
            context_type="decision",
            key="auth_method",
            content="Use JWT tokens for authentication",
        )

        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_decision(self, mock_db_session: AsyncMock) -> None:
        """Test storing a decision."""
        manager = ContextManager(mock_db_session)

        decision = await manager.store_decision(
            project_id="project-1",
            category="architecture",
            decision="Use REST API",
            rationale="Better tooling support",
            alternatives=["GraphQL", "gRPC"],
        )

        mock_db_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_file_context(self, mock_db_session: AsyncMock) -> None:
        """Test storing file context."""
        manager = ContextManager(mock_db_session)

        snapshot = await manager.store_file_context(
            session_id="session-1",
            file_path="src/models/user.py",
            content="class User:\n    pass",
        )

        mock_db_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_discovery(self, mock_db_session: AsyncMock) -> None:
        """Test storing a discovery."""
        manager = ContextManager(mock_db_session)

        snapshot = await manager.store_discovery(
            session_id="session-1",
            key="existing_auth",
            content="Project already has basic auth implementation",
        )

        mock_db_session.add.assert_called_once()
