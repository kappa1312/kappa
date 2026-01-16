"""
WebSocket Connection Manager.

Handles real-time updates to dashboard clients.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts.

    Supports per-project subscriptions for targeted updates.
    """

    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: list[WebSocket] = []
        self.subscriptions: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept new WebSocket connection.

        Args:
            websocket: The WebSocket connection to accept.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = set()
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: The WebSocket connection that disconnected.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def handle_message(self, websocket: WebSocket, data: str) -> None:
        """
        Handle incoming WebSocket message.

        Supports subscribe/unsubscribe actions for project-specific updates.

        Args:
            websocket: The WebSocket that sent the message.
            data: The raw message data (JSON string).
        """
        try:
            message = json.loads(data)
            action = message.get("action")

            if action == "subscribe":
                project_id = message.get("project_id")
                if project_id and websocket in self.subscriptions:
                    self.subscriptions[websocket].add(project_id)
                    logger.debug(f"Client subscribed to project {project_id}")
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "subscription_confirmed",
                                "project_id": project_id,
                            }
                        )
                    )

            elif action == "unsubscribe":
                project_id = message.get("project_id")
                if project_id and websocket in self.subscriptions:
                    self.subscriptions[websocket].discard(project_id)
                    logger.debug(f"Client unsubscribed from project {project_id}")

            elif action == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

        except json.JSONDecodeError:
            logger.warning(f"Invalid WebSocket message: {data}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """
        Broadcast message to all connected clients.

        Args:
            message: The message to broadcast.
        """
        if not self.active_connections:
            return

        data = json.dumps(message)
        disconnected: list[WebSocket] = []

        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def send_to_project_subscribers(self, project_id: str, message: dict[str, Any]) -> None:
        """
        Send message to clients subscribed to specific project.

        Args:
            project_id: The project ID to send to.
            message: The message to send.
        """
        data = json.dumps(message)
        disconnected: list[WebSocket] = []

        for websocket, subscriptions in self.subscriptions.items():
            if project_id in subscriptions:
                try:
                    await websocket.send_text(data)
                except Exception as e:
                    logger.error(f"Send error: {e}")
                    disconnected.append(websocket)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def notify_task_update(
        self,
        project_id: str,
        task_id: str,
        status: str,
        progress: int | None = None,
    ) -> None:
        """
        Notify subscribers of task status change.

        Args:
            project_id: The project ID.
            task_id: The task ID.
            status: The new task status.
            progress: Optional progress percentage (0-100).
        """
        await self.send_to_project_subscribers(
            project_id,
            {
                "type": "task_update",
                "project_id": project_id,
                "task_id": task_id,
                "status": status,
                "progress": progress,
            },
        )

    async def notify_wave_update(
        self,
        project_id: str,
        wave_id: int,
        status: str,
    ) -> None:
        """
        Notify subscribers of wave status change.

        Args:
            project_id: The project ID.
            wave_id: The wave number.
            status: The new wave status.
        """
        await self.send_to_project_subscribers(
            project_id,
            {
                "type": "wave_update",
                "project_id": project_id,
                "wave_id": wave_id,
                "status": status,
            },
        )

    async def notify_session_update(
        self,
        project_id: str,
        session_id: str,
        status: str,
        output: str | None = None,
    ) -> None:
        """
        Notify subscribers of session output.

        Args:
            project_id: The project ID.
            session_id: The session ID.
            status: The session status.
            output: Optional output text.
        """
        await self.send_to_project_subscribers(
            project_id,
            {
                "type": "session_update",
                "project_id": project_id,
                "session_id": session_id,
                "status": status,
                "output": output,
            },
        )

    async def notify_project_update(
        self,
        project_id: str,
        status: str,
        message: str | None = None,
    ) -> None:
        """
        Notify subscribers of project status change.

        Args:
            project_id: The project ID.
            status: The new project status.
            message: Optional status message.
        """
        await self.send_to_project_subscribers(
            project_id,
            {
                "type": "project_update",
                "project_id": project_id,
                "status": status,
                "message": message,
            },
        )

    async def notify_conflict_detected(
        self,
        project_id: str,
        conflict_id: str,
        conflict_type: str,
        description: str,
    ) -> None:
        """
        Notify subscribers of a detected conflict.

        Args:
            project_id: The project ID.
            conflict_id: The conflict ID.
            conflict_type: The type of conflict.
            description: Human-readable description.
        """
        await self.send_to_project_subscribers(
            project_id,
            {
                "type": "conflict_detected",
                "project_id": project_id,
                "conflict_id": conflict_id,
                "conflict_type": conflict_type,
                "description": description,
            },
        )

    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)


# Global instance
ws_manager = ConnectionManager()
