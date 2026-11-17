"""
Telemetry Module.
Drone telemetry ingestion, timestamp synchronization, and mock/simulated fallback generation.
"""

from app.video.telemetry import (
    DroneTelemetry,
    GPSCoordinate,
    GimbalOrientation,
    DroneStatus,
    TelemetryBuffer,
    TelemetrySynchronizer,
    MockTelemetryProvider,
)

__all__ = [
    "DroneTelemetry",
    "GPSCoordinate",
    "GimbalOrientation",
    "DroneStatus",
    "TelemetryBuffer",
    "TelemetrySynchronizer",
    "MockTelemetryProvider",
]

