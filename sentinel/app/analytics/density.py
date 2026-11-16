from typing import List, Optional
from loguru import logger

from app.analytics.models import (
    DensityResult,
    DensityAnalyzer,
    AnalyticsConfig,
    AnalyticsContext,
)
from app.tracking.models import Track


class DensityEstimator:
    def __init__(self, config: Optional["AnalyticsConfig"] = None):
        from app.analytics.models import AnalyticsConfig
        if config is None:
            config = AnalyticsConfig()
        self.analyzer = DensityAnalyzer(config)

    def estimate(self, tracks: List[Track], context: AnalyticsContext, frame_rate: float = 30.0) -> DensityResult:
        return self.analyzer.compute_density(tracks, context, frame_rate)