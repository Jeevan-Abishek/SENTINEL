"""
SENTINEL Geospatial & Mapping Package.
Provides camera calibration, ground coordinate homography, zone mapping, and telemetry synchronization.
"""

from app.geospatial.calibration import CoordinateCalibrator, GroundResolutionCalculator, CalibrationPoint
from app.geospatial.zone_mapping import ZoneManager, PolygonZoneValidator, ZoneDefinition, MapRenderer
from app.geospatial.telemetry import (
    DroneTelemetry,
    GPSCoordinate,
    GimbalOrientation,
    DroneStatus,
    TelemetryBuffer,
    TelemetrySynchronizer,
    MockTelemetryProvider,
)

__all__ = [
    "CoordinateCalibrator",
    "GroundResolutionCalculator",
    "CalibrationPoint",
    "ZoneManager",
    "PolygonZoneValidator",
    "ZoneDefinition",
    "MapRenderer",
    "DroneTelemetry",
    "GPSCoordinate",
    "GimbalOrientation",
    "DroneStatus",
    "TelemetryBuffer",
    "TelemetrySynchronizer",
    "MockTelemetryProvider",
]

