"""
Crowd Flow & Gate Accumulation Dynamics Module for SENTINEL.
Computes directional flow vectors, ingress vs. egress flow rates, net crowd accumulation
(persons/min), counter-flow conflicts, and gate compression ratios.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from loguru import logger

from app.analytics.models import (
    FlowResult,
    FlowAnalyzer,
    AnalyticsConfig,
    AnalyticsContext,
)
from app.tracking.models import Track


def calculate_inflow_outflow_ratio(inflow: float, outflow: float) -> float:
    """Compute ratio of ingress flow to egress flow (> 1.0 indicates ingress surplus)."""
    return round(float(inflow / max(0.1, outflow)), 2)


def calculate_net_accumulation_rate(inflow: float, outflow: float) -> float:
    """Compute net crowd accumulation rate (persons/min)."""
    return round(float(inflow - outflow), 2)


class FlowAnalyzerWrapper:
    """
    High-performance crowd flow analyzer wrapper.
    Evaluates virtual gate lines, rolling accumulation rates, and flow conflict conditions.
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.config = config
        self.analyzer = FlowAnalyzer(config)

    def analyze(
        self,
        tracks: List[Any],
        context: AnalyticsContext,
        frame_rate: float = 30.0,
        entry_lines: Optional[List[dict]] = None,
        exit_lines: Optional[List[dict]] = None,
    ) -> FlowResult:
        """Run directional flow and net accumulation analysis across active tracks."""
        return self.analyzer.analyze(
            tracks=tracks,
            context=context,
            frame_rate=frame_rate,
            entry_lines=entry_lines,
            exit_lines=exit_lines,
        )

    def get_normalized_risk_score(self, flow_result: FlowResult) -> float:
        """Returns normalized 0.0 - 1.0 flow risk score for Risk Engine."""
        return min(1.0, max(0.0, round(flow_result.risk_score / 100.0, 4)))


__all__ = [
    "FlowAnalyzerWrapper",
    "FlowAnalyzer",
    "calculate_inflow_outflow_ratio",
    "calculate_net_accumulation_rate",
]
