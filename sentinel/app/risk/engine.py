from typing import Optional
from loguru import logger

from app.risk.models import RiskEngine, RiskConfig, RiskAssessment
from app.analytics.models import (
    DensityResult, OccupancyResult, MovementResult, FlowResult, BottleneckResult
)


class RiskEngineWrapper:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.engine = RiskEngine(config)

    def assess(
        self,
        zone_id: str,
        timestamp: float,
        density_result: Optional[DensityResult] = None,
        occupancy_result: Optional[OccupancyResult] = None,
        movement_result: Optional[MovementResult] = None,
        flow_result: Optional[FlowResult] = None,
        bottleneck_result: Optional[BottleneckResult] = None,
    ) -> RiskAssessment:
        return self.engine.assess(
            zone_id=zone_id,
            timestamp=timestamp,
            density_result=density_result,
            occupancy_result=occupancy_result,
            movement_result=movement_result,
            flow_result=flow_result,
            bottleneck_result=bottleneck_result,
        )