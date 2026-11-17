"""
Zone Analyzer Module.
Monitors occupancy ratios and safe capacities within polygonal monitoring zones.
"""

from app.analytics.occupancy import OccupancyMonitor, OccupancyResult
from app.analytics.zone_config import ZoneOccupancyConfig

# Alias matching master architecture
ZoneAnalyzer = OccupancyMonitor

__all__ = ["OccupancyMonitor", "ZoneAnalyzer", "OccupancyResult", "ZoneOccupancyConfig"]

