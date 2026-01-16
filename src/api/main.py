"""
Kappa OS Dashboard API.

FastAPI backend for real-time dashboard updates.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.websocket import ws_manager


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Startup and shutdown events.

    Args:
        app: The FastAPI application instance.

    Yields:
        None during application runtime.
    """
    logger.info("Starting Kappa OS API...")
    yield
    logger.info("Shutting down Kappa OS API...")


app = FastAPI(
    title="Kappa OS API",
    description="Autonomous Development Operating System API",
    version="0.0.5",
    lifespan=lifespan,
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import and include routers
from src.api.routes import metrics, projects, sessions, tasks  # noqa: E402

app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time updates.

    Clients can subscribe to project updates by sending:
    {"action": "subscribe", "project_id": "<id>"}

    Args:
        websocket: The WebSocket connection.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await ws_manager.handle_message(websocket, data)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status and version.
    """
    return {"status": "healthy", "version": "0.0.5"}


@app.get("/")
async def root() -> dict[str, str]:
    """
    Root endpoint.

    Returns:
        Welcome message.
    """
    return {
        "name": "Kappa OS API",
        "version": "0.0.5",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api/ws-status")
async def ws_status() -> dict[str, int]:
    """
    Get WebSocket connection status.

    Returns:
        Number of active connections.
    """
    return {"active_connections": ws_manager.connection_count}
