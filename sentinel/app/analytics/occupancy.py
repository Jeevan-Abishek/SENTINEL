"""
Zone Occupancy & Headroom Monitoring Module for SENTINEL.
Computes occupancy ratios (current_count / capacity), remaining safe headcount headroom,
status classifications (safe, warning, critical), and normalized risk scores.
"""

from typing import Optional, Literal
from loguru import logger

from app.analytics.models import (
    OccupancyResult,
    OccupancyAnalyzer,
    AnalyticsConfig,
)


def calculate_occupancy_ratio(count: int, capacity: int) -> float:
    """Safely compute occupancy ratio, avoiding division by zero if capacity is invalid."""
    if capacity <= 0:
        return 0.0
    return round(float(count) / float(capacity), 4)


def calculate_headroom(count: int, capacity: int) -> int:
    """Compute remaining safe headcount capacity headroom."""
    if capacity <= 0:
        return 0
    return max(0, capacity - count)


def calculate_headroom_ratio(count: int, capacity: int) -> float:
    """Compute remaining safe headroom as a ratio of capacity (1.0 = empty, 0.0 = full)."""
    if capacity <= 0:
        return 1.0
    headroom = calculate_headroom(count, capacity)
    return round(float(headroom) / float(capacity), 4)


def classify_occupancy_status(
    occupancy_ratio: float,
    warning_threshold: float = 0.70,
    critical_threshold: float = 0.90,
) -> Literal["safe", "warning", "critical"]:
    """Classify occupancy ratio into standardized safety tiers."""
    if occupancy_ratio >= critical_threshold:
        return "critical"
    if occupancy_ratio >= warning_threshold:
        return "warning"
    return "safe"


class OccupancyMonitor:
    """
    Real-time zone occupancy and safe headcount capacity monitor.
    Enforces safe headroom tracking and threshold monitoring.
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.config = config
        self.analyzer = OccupancyAnalyzer(config)

    def analyze(
        self,
        count: int,
        capacity: int,
        timestamp: float,
        zone_id: str,
        projected_count_5min: Optional[int] = None,
    ) -> OccupancyResult:
        """Analyze occupancy metrics for the specified zone headcount."""
        return self.analyzer.analyze(
            count=count,
            capacity=capacity,
            timestamp=timestamp,
            zone_id=zone_id,
            projected_count_5min=projected_count_5min,
        )

    def is_warning(self, occupancy_ratio: float) -> bool:
        """Check if occupancy has reached or exceeded warning threshold."""
        return occupancy_ratio >= self.analyzer.config.occupancy_warning_threshold

    def is_critical(self, occupancy_ratio: float) -> bool:
        """Check if occupancy has reached or exceeded critical safety threshold."""
        return occupancy_ratio >= self.analyzer.config.occupancy_critical_threshold

    def get_headroom(self, count: int, capacity: int) -> int:
        """Convenience accessor for remaining capacity headcount."""
        return calculate_headroom(count, capacity)

    def get_normalized_risk_score(self, occupancy_ratio: float) -> float:
        """Normalized 0.0 - 1.0 occupancy score for Risk Engine ingestion."""
        return min(1.0, max(0.0, round(float(occupancy_ratio), 4)))


__all__ = [
    "OccupancyMonitor",
    "OccupancyAnalyzer",
    "calculate_occupancy_ratio",
    "calculate_headroom",
    "calculate_headroom_ratio",
    "classify_occupancy_status",
]
