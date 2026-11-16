from typing import List, Optional
from loguru import logger

from app.analytics.models import (
    BottleneckResult,
    BottleneckDetector,
    AnalyticsConfig,
    DensityResult,
    FlowResult,
    MovementResult,
    AnalyticsContext,
)


class BottleneckDetectorWrapper:
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.detector = BottleneckDetector(config)

    def analyze(
        self, 
        density_result: DensityResult, 
        flow_result: FlowResult, 
        movement_result: MovementResult,
        context: AnalyticsContext
    ) -> BottleneckResult:
        return self.detector.analyze(density_result, flow_result, movement_result, context)