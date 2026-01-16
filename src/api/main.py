"""
Kappa OS Dashboard API.

FastAPI backend for real-time dashboard updates.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.websocket import ws_manager

# Dashboard build directory
DASHBOARD_DIST = Path(__file__).parent.parent / "dashboard" / "dist"


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
    version="0.1.0-beta",
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
    return {"status": "healthy", "version": "0.1.0-beta"}


@app.get("/api/ws-status")
async def ws_status() -> dict[str, int]:
    """
    Get WebSocket connection status.

    Returns:
        Number of active connections.
    """
    return {"active_connections": ws_manager.connection_count}


# Mount static assets if dashboard is built
if (DASHBOARD_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_DIST / "assets"), name="assets")


@app.get("/", response_model=None)
async def serve_dashboard() -> FileResponse | dict[str, str]:
    """
    Serve the dashboard or return API info.

    Returns:
        Dashboard index.html if built, otherwise API info.
    """
    index_file = DASHBOARD_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return {
        "name": "Kappa OS API",
        "version": "0.1.0-beta",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "Not built. Run: cd src/dashboard && npm install && npm run build",
    }


@app.get("/{path:path}", response_model=None)
async def serve_spa(path: str) -> FileResponse:
    """
    Catch-all route for SPA routing.

    Args:
        path: The requested path.

    Returns:
        The requested file or index.html for SPA routing.

    Raises:
        HTTPException: 404 if the path is an unknown API route.
    """
    from fastapi import HTTPException

    # For API, WebSocket, and docs routes, raise 404 if they weren't matched by specific routes
    if path.startswith(("api/", "ws", "docs", "openapi", "health", "redoc")):
        raise HTTPException(status_code=404, detail="Not Found")

    # Try to serve the file directly from dist
    file_path = DASHBOARD_DIST / path
    if file_path.exists() and file_path.is_file():
        # Determine media type
        suffix = file_path.suffix.lower()
        media_types = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".html": "text/html",
            ".json": "application/json",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".ico": "image/x-icon",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
        }
        media_type = media_types.get(suffix, "application/octet-stream")
        return FileResponse(file_path, media_type=media_type)

    # Fall back to index.html for SPA routing
    index_file = DASHBOARD_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")

    # No dashboard built
    raise HTTPException(status_code=404, detail="Dashboard not built")
