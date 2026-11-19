"""
Crowd Movement & Velocity Field Dynamics Module for SENTINEL.
Computes vectorized per-track speed, movement heading angles, crowd stagnation ratios,
sudden velocity deceleration drops, directional convergence, and counter-flow conflicts.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from loguru import logger

from app.analytics.models import (
    MovementResult,
    MovementVector,
    MovementAnalyzer,
    AnalyticsConfig,
    AnalyticsContext,
)
from app.tracking.models import Track


def compute_vector_speed(vx: float, vy: float) -> float:
    """Compute Euclidean velocity magnitude."""
    return float(np.sqrt(vx * vx + vy * vy))


def compute_heading_degrees(vx: float, vy: float) -> float:
    """Compute heading angle in degrees normalized to [0.0, 360.0)."""
    if vx == 0 and vy == 0:
        return 0.0
    rad = np.arctan2(vy, vx)
    deg = float(np.degrees(rad))
    return float((deg + 360.0) % 360.0)


def compute_stagnation_ratio(speeds: List[float], threshold: float = 5.0) -> Tuple[float, int]:
    """
    Compute fraction and count of tracks moving below the stagnation threshold.
    Returns: (stagnation_ratio, stagnant_count)
    """
    if not speeds:
        return 0.0, 0
    stagnant = sum(1 for s in speeds if s < threshold)
    return round(float(stagnant / len(speeds)), 3), stagnant


def compute_directional_convergence(
    centroids: List[Tuple[float, float]], 
    velocities: List[Tuple[float, float]]
) -> float:
    """
    Compute directional convergence index (0.0 to 1.0).
    Measures the degree to which crowd velocity vectors funnel inward toward the group centroid.
    """
    if len(centroids) < 2 or len(centroids) != len(velocities):
        return 0.0

    c_arr = np.array(centroids, dtype=np.float32)
    center = np.mean(c_arr, axis=0)

    inward_alignments = []
    for (cx, cy), (vx, vy) in zip(centroids, velocities):
        dx = center[0] - cx
        dy = center[1] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        v_mag = np.sqrt(vx * vx + vy * vy)

        if dist > 1.0 and v_mag > 0.01:
            # Cosine similarity between velocity vector and vector toward centroid
            cos_phi = (vx * dx + vy * dy) / (v_mag * dist)
            inward_alignments.append(max(0.0, float(cos_phi)))

    if not inward_alignments:
        return 0.0
    return round(float(np.mean(inward_alignments)), 3)


class MovementAnalyzerWrapper:
    """
    High-performance crowd movement analyzer wrapper.
    Evaluates velocity fields, sudden crowd slowing, stagnation, and crossing vectors.
    """

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if config is None:
            config = AnalyticsConfig()
        self.config = config
        self.analyzer = MovementAnalyzer(config)

    def analyze(
        self,
        tracks: List[Any],
        context: AnalyticsContext,
        frame_rate: float = 30.0,
    ) -> MovementResult:
        """Run vectorized movement analysis across active tracks."""
        return self.analyzer.analyze(tracks, context, frame_rate)

    def get_normalized_risk_score(self, movement_result: MovementResult) -> float:
        """Returns normalized 0.0 - 1.0 movement risk score for Risk Engine."""
        return min(1.0, max(0.0, round(movement_result.risk_score / 100.0, 4)))


__all__ = [
    "MovementAnalyzerWrapper",
    "MovementAnalyzer",
    "compute_vector_speed",
    "compute_heading_degrees",
    "compute_stagnation_ratio",
    "compute_directional_convergence",
]
