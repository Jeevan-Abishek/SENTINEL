"""
WebSocket Communication Module.
Manages client connections, heartbeats, and real-time telemetry/alert broadcasts.
"""

from app.api.main import ConnectionManager, manager

__all__ = ["ConnectionManager", "manager"]

