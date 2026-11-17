"""
Drone Adapter Module.
High-level drone stream adapter unifying video feed, telemetry sync, and camera motion compensation.
"""

from app.video.drone_adapter import DroneStreamAdapter, DroneFrameBundle
from app.video.telemetry import DroneTelemetry, GPSCoordinate, GimbalOrientation

__all__ = ["DroneStreamAdapter", "DroneFrameBundle", "DroneTelemetry", "GPSCoordinate", "GimbalOrientation"]

