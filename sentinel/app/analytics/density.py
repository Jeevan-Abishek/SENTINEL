"""
Crowd Density Estimation & Trend Prediction Module for SENTINEL.
Computes ground crowd density (people/m²), spatial heat grid distributions,
density rate-of-change (d(density)/dt), and 5-minute predictive forward trends.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from loguru import logger

from app.analytics.models import (
    DensityResult,
    DensityGrid,
    DensityAnalyzer,
    AnalyticsConfig,
    AnalyticsContext,
)
from app.tracking.models import Track


def calculate_zone_density(count: int, area_sqm: Optional[float] = None, fallback_area: float = 100.0) -> float:
    """Compute physical ground density in people/m² with safe area fallback."""
    area = area_sqm if (area_sqm is not None and area_sqm > 0) else fallback_area
    return round(float(count / max(area, 0.1)), 3)


def calculate_density_rate_of_change(
    history: List[Tuple[float, float]], 
    min_dt_seconds: float = 0.05
) -> float:
    """
    Compute rate-of-change in persons/(m² * min) from a rolling timestamp-density history.
    history: [(timestamp, density), ...]
    """
    if len(history) < 2:
        return 0.0
    t0, d0 = history[0]
    t1, d1 = history[-1]
    dt = t1 - t0
    if dt < min_dt_seconds:
        return 0.0
    rate_per_sec = (d1 - d0) / dt
    return round(float(rate_per_sec * 60.0), 4)


def project_density_trend(
    current_density: float, 
    rate_per_min: float, 
    horizon_minutes: float = 5.0
) -> float:
    """Linear predictive forward horizon projection at t + horizon_minutes."""
    projected = current_density + (rate_per_min * horizon_minutes)
    return max(0.0, round(float(projected), 3))


def estimate_lead_time_to_threshold(
    current_density: float,
    rate_per_min: float,
    threshold: float = 3.0,
    horizon_minutes: float = 5.0,
) -> Optional[float]:
    """Estimate advance minutes until density breaches critical threshold."""
    if current_density >= threshold:
        return 0.0
    if rate_per_min <= 0.01:
        return None
    time_min = (threshold - current_density) / rate_per_min
    if 0.0 < time_min <= horizon_minutes:
        return round(float(time_min), 1)
    return None


class DensityEstimator:
    """
    High-accuracy ground-plane crowd density estimator.
    Seamlessly integrates calibrated ground polygons, homography coordinates,
    and rolling trend-based forward forecasting.
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.config = config
        self.analyzer = DensityAnalyzer(config)

    def estimate(
        self,
        tracks: List[Any],
        context: AnalyticsContext,
        frame_rate: float = 30.0,
        area_sqm: Optional[float] = None,
    ) -> DensityResult:
        """Estimate crowd density for the given context and active tracks."""
        if area_sqm is not None:
            context.area_sqm = area_sqm
        return self.analyzer.compute_density(tracks, context, frame_rate)

    def compute_density(
        self,
        tracks: List[Any],
        context: AnalyticsContext,
        frame_rate: float = 30.0,
        area_sqm: Optional[float] = None,
    ) -> DensityResult:
        """Alias for estimate()."""
        return self.estimate(tracks, context, frame_rate, area_sqm)

    def get_normalized_risk_score(
        self,
        max_density: float,
        warning_thresh: float = 3.0,
        critical_thresh: float = 5.0,
    ) -> float:
        """Returns normalized 0.0 - 1.0 density risk score."""
        if max_density <= 0:
            return 0.0
        score = min(100.0, (max_density ** 1.5) * 8.0)
        return round(score / 100.0, 4)


__all__ = [
    "DensityEstimator",
    "DensityAnalyzer",
    "calculate_zone_density",
    "calculate_density_rate_of_change",
    "project_density_trend",
    "estimate_lead_time_to_threshold",
]
