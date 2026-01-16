"""
Kappa OS Dashboard API.

FastAPI backend for real-time dashboard updates.
"""

from src.api.main import app
from src.api.websocket import ConnectionManager, ws_manager

__all__ = ["app", "ConnectionManager", "ws_manager"]
