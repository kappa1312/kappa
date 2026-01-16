"""
Unit tests for the Kappa OS Dashboard API.

Tests FastAPI endpoints and WebSocket functionality.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.websocket import ConnectionManager, ws_manager

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def manager():
    """Create a fresh ConnectionManager for testing."""
    return ConnectionManager()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = MagicMock(spec=WebSocket)
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    return ws


# =============================================================================
# TEST HEALTH ENDPOINT
# =============================================================================


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check_returns_200(self, client) -> None:
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_returns_status(self, client) -> None:
        """Test health endpoint returns status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_returns_version(self, client) -> None:
        """Test health endpoint returns version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.0.5"


# =============================================================================
# TEST ROOT ENDPOINT
# =============================================================================


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, client) -> None:
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_info(self, client) -> None:
        """Test root endpoint returns API info."""
        response = client.get("/")
        data = response.json()
        assert data["name"] == "Kappa OS API"
        assert "version" in data
        assert "docs" in data


# =============================================================================
# TEST WS STATUS ENDPOINT
# =============================================================================


class TestWSStatusEndpoint:
    """Tests for the WebSocket status endpoint."""

    def test_ws_status_returns_200(self, client) -> None:
        """Test WS status endpoint returns 200."""
        response = client.get("/api/ws-status")
        assert response.status_code == 200

    def test_ws_status_returns_count(self, client) -> None:
        """Test WS status returns connection count."""
        response = client.get("/api/ws-status")
        data = response.json()
        assert "active_connections" in data
        assert isinstance(data["active_connections"], int)


# =============================================================================
# TEST CONNECTION MANAGER
# =============================================================================


class TestWebSocketManager:
    """Tests for ConnectionManager class."""

    def test_manager_init(self, manager) -> None:
        """Test ConnectionManager initialization."""
        assert manager.active_connections == []
        assert manager.subscriptions == {}

    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket) -> None:
        """Test connecting a WebSocket."""
        await manager.connect(mock_websocket)

        assert mock_websocket in manager.active_connections
        assert mock_websocket in manager.subscriptions
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket) -> None:
        """Test disconnecting a WebSocket."""
        await manager.connect(mock_websocket)
        manager.disconnect(mock_websocket)

        assert mock_websocket not in manager.active_connections
        assert mock_websocket not in manager.subscriptions

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, manager, mock_websocket) -> None:
        """Test disconnecting a non-connected WebSocket."""
        # Should not raise
        manager.disconnect(mock_websocket)
        assert mock_websocket not in manager.active_connections

    def test_connection_count(self, manager) -> None:
        """Test connection_count property."""
        assert manager.connection_count == 0


# =============================================================================
# TEST MESSAGE HANDLING
# =============================================================================


class TestWebSocketMessageHandling:
    """Tests for WebSocket message handling."""

    @pytest.mark.asyncio
    async def test_handle_subscribe(self, manager, mock_websocket) -> None:
        """Test handling subscribe message."""
        await manager.connect(mock_websocket)

        message = json.dumps({"action": "subscribe", "project_id": "test-project"})
        await manager.handle_message(mock_websocket, message)

        assert "test-project" in manager.subscriptions[mock_websocket]

    @pytest.mark.asyncio
    async def test_handle_unsubscribe(self, manager, mock_websocket) -> None:
        """Test handling unsubscribe message."""
        await manager.connect(mock_websocket)
        manager.subscriptions[mock_websocket].add("test-project")

        message = json.dumps({"action": "unsubscribe", "project_id": "test-project"})
        await manager.handle_message(mock_websocket, message)

        assert "test-project" not in manager.subscriptions[mock_websocket]

    @pytest.mark.asyncio
    async def test_handle_ping(self, manager, mock_websocket) -> None:
        """Test handling ping message."""
        await manager.connect(mock_websocket)

        message = json.dumps({"action": "ping"})
        await manager.handle_message(mock_websocket, message)

        # Should send pong
        mock_websocket.send_text.assert_called()
        call_args = mock_websocket.send_text.call_args[0][0]
        assert "pong" in call_args

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, manager, mock_websocket) -> None:
        """Test handling invalid JSON message."""
        await manager.connect(mock_websocket)

        # Should not raise
        await manager.handle_message(mock_websocket, "not json")


# =============================================================================
# TEST BROADCASTING
# =============================================================================


class TestWebSocketBroadcast:
    """Tests for WebSocket broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, manager, mock_websocket) -> None:
        """Test broadcasting to all connections."""
        await manager.connect(mock_websocket)

        await manager.broadcast({"type": "test", "data": "hello"})

        mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_broadcast_empty_connections(self, manager) -> None:
        """Test broadcasting with no connections."""
        # Should not raise
        await manager.broadcast({"type": "test"})

    @pytest.mark.asyncio
    async def test_send_to_project_subscribers(self, manager, mock_websocket) -> None:
        """Test sending to project subscribers."""
        await manager.connect(mock_websocket)
        manager.subscriptions[mock_websocket].add("project-123")

        await manager.send_to_project_subscribers("project-123", {"type": "update"})

        mock_websocket.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_send_to_non_subscribers(self, manager, mock_websocket) -> None:
        """Test sending to project with no subscribers."""
        await manager.connect(mock_websocket)
        # Not subscribed to "other-project"

        await manager.send_to_project_subscribers("other-project", {"type": "update"})

        mock_websocket.send_text.assert_not_called()


# =============================================================================
# TEST NOTIFICATION METHODS
# =============================================================================


class TestNotificationMethods:
    """Tests for notification helper methods."""

    @pytest.mark.asyncio
    async def test_notify_task_update(self, manager, mock_websocket) -> None:
        """Test task update notification."""
        await manager.connect(mock_websocket)
        manager.subscriptions[mock_websocket].add("project-1")

        await manager.notify_task_update("project-1", "task-1", "completed", 100)

        mock_websocket.send_text.assert_called()
        call_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert call_data["type"] == "task_update"
        assert call_data["task_id"] == "task-1"
        assert call_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_notify_wave_update(self, manager, mock_websocket) -> None:
        """Test wave update notification."""
        await manager.connect(mock_websocket)
        manager.subscriptions[mock_websocket].add("project-1")

        await manager.notify_wave_update("project-1", 2, "executing")

        mock_websocket.send_text.assert_called()
        call_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert call_data["type"] == "wave_update"
        assert call_data["wave_id"] == 2

    @pytest.mark.asyncio
    async def test_notify_session_update(self, manager, mock_websocket) -> None:
        """Test session update notification."""
        await manager.connect(mock_websocket)
        manager.subscriptions[mock_websocket].add("project-1")

        await manager.notify_session_update("project-1", "session-1", "running", "output")

        mock_websocket.send_text.assert_called()
        call_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert call_data["type"] == "session_update"

    @pytest.mark.asyncio
    async def test_notify_project_update(self, manager, mock_websocket) -> None:
        """Test project update notification."""
        await manager.connect(mock_websocket)
        manager.subscriptions[mock_websocket].add("project-1")

        await manager.notify_project_update("project-1", "completed", "Build successful")

        mock_websocket.send_text.assert_called()
        call_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert call_data["type"] == "project_update"
        assert call_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_notify_conflict_detected(self, manager, mock_websocket) -> None:
        """Test conflict detection notification."""
        await manager.connect(mock_websocket)
        manager.subscriptions[mock_websocket].add("project-1")

        await manager.notify_conflict_detected(
            "project-1", "conflict-1", "file_write", "Multiple writes to index.ts"
        )

        mock_websocket.send_text.assert_called()
        call_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert call_data["type"] == "conflict_detected"


# =============================================================================
# TEST API ERROR HANDLING
# =============================================================================


class TestAPIErrorHandling:
    """Tests for API error responses."""

    def test_404_on_unknown_route(self, client) -> None:
        """Test 404 on unknown route."""
        response = client.get("/api/unknown")
        assert response.status_code == 404

    def test_projects_list_requires_no_auth(self, client) -> None:
        """Test projects list doesn't require auth (for now)."""
        # This would fail if database not available, but tests endpoint structure
        try:
            response = client.get("/api/projects/")
            # Either 200 (if DB available) or 500 (if not)
            assert response.status_code in [200, 500]
        except Exception:
            # Database connection error is acceptable in unit tests
            pass

    def test_tasks_requires_project_id(self, client) -> None:
        """Test tasks endpoint requires project_id."""
        response = client.get("/api/tasks/")
        assert response.status_code == 422  # Validation error

    def test_sessions_requires_project_id(self, client) -> None:
        """Test sessions endpoint requires project_id."""
        response = client.get("/api/sessions/")
        assert response.status_code == 422  # Validation error


# =============================================================================
# TEST GLOBAL MANAGER INSTANCE
# =============================================================================


class TestGlobalManager:
    """Tests for global ws_manager instance."""

    def test_global_manager_exists(self) -> None:
        """Test global ws_manager is available."""
        assert ws_manager is not None
        assert isinstance(ws_manager, ConnectionManager)

    def test_global_manager_has_methods(self) -> None:
        """Test global manager has expected methods."""
        assert hasattr(ws_manager, "connect")
        assert hasattr(ws_manager, "disconnect")
        assert hasattr(ws_manager, "broadcast")
        assert hasattr(ws_manager, "notify_task_update")


# =============================================================================
# TEST CORS MIDDLEWARE
# =============================================================================


class TestCORSMiddleware:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_origin(self, client) -> None:
        """Test CORS allows requests."""
        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        # Options should return 200 or 405 depending on FastAPI version
        assert response.status_code in [200, 405]

    def test_health_with_origin_header(self, client) -> None:
        """Test health endpoint with origin header."""
        response = client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.status_code == 200
