"""
Database Repository Module.
Provides CRUD abstractions for Events, Zones, Drones, Detections, Tracks, Analytics, and Alerts.
"""

from app.database.repositories import (
    BaseRepository,
    EventRepository,
    ZoneRepository,
    DroneRepository,
    DetectionRepository,
    TrackRepository,
    AnalyticsRepository,
    DensityHeatmapRepository,
    RiskRepository,
    AlertRepository,
    OperatorRepository,
    AuditLogRepository,
)

__all__ = [
    "BaseRepository",
    "EventRepository",
    "ZoneRepository",
    "DroneRepository",
    "DetectionRepository",
    "TrackRepository",
    "AnalyticsRepository",
    "DensityHeatmapRepository",
    "RiskRepository",
    "AlertRepository",
    "OperatorRepository",
    "AuditLogRepository",
]

