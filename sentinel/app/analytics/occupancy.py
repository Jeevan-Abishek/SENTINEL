from typing import List, Optional
from loguru import logger

from app.analytics.models import (
    OccupancyResult,
    OccupancyAnalyzer,
    AnalyticsConfig,
)


class OccupancyMonitor:
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.analyzer = OccupancyAnalyzer(config)

    def analyze(self, count: int, capacity: int, timestamp: float, zone_id: str) -> OccupancyResult:
        return self.analyzer.analyze(count, capacity, timestamp, zone_id)

    def is_warning(self, occupancy_ratio: float) -> bool:
        return occupancy_ratio >= self.analyzer.config.occupancy_warning_threshold

    def is_critical(self, occupancy_ratio: float) -> bool:
        return occupancy_ratio >= self.analyzer.config.occupancy_critical_threshold