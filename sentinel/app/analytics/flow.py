from typing import List, Optional
from loguru import logger

from app.analytics.models import (
    FlowResult,
    FlowAnalyzer,
    AnalyticsConfig,
    AnalyticsContext,
)
from app.tracking.models import Track


class FlowAnalyzerWrapper:
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.analyzer = FlowAnalyzer(config)

    def analyze(
        self, 
        tracks: List[Track], 
        context: AnalyticsContext, 
        frame_rate: float = 30.0,
        entry_lines: Optional[List[dict]] = None,
        exit_lines: Optional[List[dict]] = None
    ) -> FlowResult:
        return self.analyzer.analyze(tracks, context, frame_rate, entry_lines, exit_lines)