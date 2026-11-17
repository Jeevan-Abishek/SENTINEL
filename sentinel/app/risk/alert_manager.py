"""
Alert Manager Module.
Coordinates alert deduplication, cooldown windows, and operator notifications.
"""

from app.alerts.manager import AlertManager, AlertManagerWrapper
from app.alerts.notifications import WebSocketNotificationChannel, SoundNotificationChannel

__all__ = [
    "AlertManager",
    "AlertManagerWrapper",
    "WebSocketNotificationChannel",
    "SoundNotificationChannel",
]

