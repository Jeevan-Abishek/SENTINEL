"""
Database helper alias at project root for direct access:
    from database import get_db, init_db, engine, Base
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.database import (
    Base,
    engine,
    async_session_maker,
    get_db,
    init_db,
    close_db,
    Event,
    Zone,
    Drone,
    DetectionRecord,
    TrackRecord,
    AnalyticsRecord,
    RiskRecord,
    Alert,
)

__all__ = [
    "Base",
    "engine",
    "async_session_maker",
    "get_db",
    "init_db",
    "close_db",
    "Event",
    "Zone",
    "Drone",
    "DetectionRecord",
    "TrackRecord",
    "AnalyticsRecord",
    "RiskRecord",
    "Alert",
]

