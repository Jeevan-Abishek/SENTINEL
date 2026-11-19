"""
Choke Point & Bottleneck Detection Module for SENTINEL.
Identifies localized crowd bottlenecks where high density and occupancy overlap with
low movement velocity, high dwell stagnation, or ingress flow compression.
"""

from typing import Optional, Literal, List
from loguru import logger

from app.analytics.models import (
    BottleneckResult,
    BottleneckDetector,
    AnalyticsConfig,
    DensityResult,
    FlowResult,
    MovementResult,
    OccupancyResult,
    AnalyticsContext,
)


def classify_congestion_level(score: float) -> Literal["none", "low", "medium", "high", "critical"]:
    """Categorize bottleneck score into standardized operational congestion tiers."""
    if score < 0.20:
        return "none"
    if score < 0.40:
        return "low"
    if score < 0.60:
        return "medium"
    if score < 0.80:
        return "high"
    return "critical"


def detect_choke_point_condition(
    density: float,
    speed: float,
    stagnation_ratio: float = 0.0,
    density_thresh: float = 2.5,
    speed_thresh: float = 0.5,
) -> bool:
    """Boolean check for dangerous choke point formation."""
    high_density = density >= density_thresh
    low_speed = speed <= speed_thresh or stagnation_ratio >= 0.40
    return high_density and low_speed


class BottleneckDetectorWrapper:
    """
    High-performance choke point and bottleneck detector wrapper.
    Fuses spatial density, directional flow, velocity fields, and stagnation metrics.
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.config = config
        self.detector = BottleneckDetector(config)

    def analyze(
        self,
        density_result: DensityResult,
        flow_result: FlowResult,
        movement_result: MovementResult,
        context: AnalyticsContext,
        occupancy_result: Optional[OccupancyResult] = None,
    ) -> BottleneckResult:
        """Run multidimensional bottleneck analysis across crowd indicators."""
        return self.detector.analyze(
            density_result=density_result,
            flow_result=flow_result,
            movement_result=movement_result,
            context=context,
        )

    def get_normalized_risk_score(self, bottleneck_result: BottleneckResult) -> float:
        """Returns normalized 0.0 - 1.0 bottleneck score for Risk Engine ingestion."""
        return min(1.0, max(0.0, round(bottleneck_result.bottleneck_score, 4)))


__all__ = [
    "BottleneckDetectorWrapper",
    "BottleneckDetector",
    "classify_congestion_level",
    "detect_choke_point_condition",
]
