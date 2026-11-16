from typing import List, Optional
from loguru import logger

from app.analytics.models import (
    MovementResult,
    MovementAnalyzer,
    AnalyticsConfig,
    AnalyticsContext,
)
from app.tracking.models import Track


class MovementAnalyzerWrapper:
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.analyzer = MovementAnalyzer(config)

    def analyze(self, tracks: List[Track], context: AnalyticsContext, frame_rate: float = 30.0) -> MovementResult:
        return self.analyzer.analyze(tracks, context, frame_rate)