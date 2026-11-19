"""
Unified 5-Factor Crowd Analytics Engine for SENTINEL.
Orchestrates Density, Occupancy, Movement, Flow, and Bottleneck analyzers,
producing normalized factor scores ([0.0, 1.0] and [0, 100]) aligned directly with the Risk Engine.
"""

from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np
from loguru import logger

from app.analytics.models import (
    AnalyticsConfig,
    AnalyticsContext,
    DensityResult,
    OccupancyResult,
    MovementResult,
    FlowResult,
    BottleneckResult,
    ZoneAnalyticsSummary,
    SceneAnalyticsSummary,
)
from app.analytics.density import DensityEstimator
from app.analytics.occupancy import OccupancyMonitor
from app.analytics.movement import MovementAnalyzerWrapper
from app.analytics.flow import FlowAnalyzerWrapper
from app.analytics.bottleneck import BottleneckDetectorWrapper
from app.analytics.counting import ScenePeopleCounter


class CrowdAnalyticsEngine:
    """
    Core Crowd Analytics Engine for SENTINEL.
    Fuses all five core crowd metrics into synchronized zone and scene summaries.
    """

    @staticmethod
    def _is_point_in_polygon(point: Any, polygon: Any) -> bool:
        if polygon is None or len(polygon) < 3:
            return True
        try:
            poly_np = np.array(polygon, dtype=np.float32)
            pt = (float(point[0]), float(point[1]))
            return bool(cv2.pointPolygonTest(poly_np, pt, False) >= 0)
        except Exception:
            return True

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self.density_estimator = DensityEstimator(self.config)
        self.occupancy_monitor = OccupancyMonitor(self.config)
        self.movement_analyzer = MovementAnalyzerWrapper(self.config)
        self.flow_analyzer = FlowAnalyzerWrapper(self.config)
        self.bottleneck_detector = BottleneckDetectorWrapper(self.config)
        self.people_counter = ScenePeopleCounter()

    def analyze_zone(
        self,
        tracks: Optional[List[Any]] = None,
        context: Optional[AnalyticsContext] = None,
        capacity: int = 100,
        area_sqm: Optional[float] = None,
        frame_rate: float = 30.0,
        entry_lines: Optional[List[dict]] = None,
        exit_lines: Optional[List[dict]] = None,
        zone_id: Optional[str] = None,
        frame_number: int = 0,
        timestamp: float = 0.0,
        zone_config: Optional[Dict[str, Any]] = None,
    ) -> ZoneAnalyticsSummary:
        """
        Execute all 5 core crowd metrics for a single surveillance zone in a single vectorized pass.
        Returns a ZoneAnalyticsSummary with individual metrics and normalized risk factors.
        """
        if tracks is None:
            tracks = []

        poly = None
        if zone_config:
            if isinstance(zone_config, dict):
                poly = zone_config.get("polygon") or zone_config.get("image_polygon")
                capacity = zone_config.get("max_capacity") or zone_config.get("capacity", capacity)
                area_sqm = zone_config.get("area_sqm") or zone_config.get("area_m2", area_sqm)
                if not entry_lines and zone_config.get("entry_line"):
                    el = zone_config["entry_line"]
                    entry_lines = [{"start": el[0], "end": el[1]}] if len(el) == 2 else None
                if not exit_lines and zone_config.get("exit_line"):
                    xl = zone_config["exit_line"]
                    exit_lines = [{"start": xl[0], "end": xl[1]}] if len(xl) == 2 else None
            else:
                poly = getattr(zone_config, "polygon", getattr(zone_config, "image_polygon", None))
                capacity = getattr(zone_config, "max_capacity", getattr(zone_config, "capacity", capacity))
                area_sqm = getattr(zone_config, "area_sqm", getattr(zone_config, "area_m2", area_sqm))

        if context is None:
            context = AnalyticsContext(
                zone_id=zone_id or "default_zone",
                frame_number=frame_number,
                timestamp=timestamp,
                zone_polygon=poly,
                zone_capacity=capacity,
                area_sqm=area_sqm,
            )
        elif poly and not context.zone_polygon:
            context.zone_polygon = poly

        # Filter tracks inside zone if zone polygon is defined
        if context.zone_polygon and len(context.zone_polygon) >= 3:
            zone_tracks = [t for t in tracks if self._is_point_in_polygon(getattr(t, "foot_xy", getattr(t, "center_xy", (0, 0))), context.zone_polygon)]
        else:
            zone_tracks = tracks

        # 1. Density Estimation
        density_res = self.density_estimator.estimate(
            tracks=zone_tracks,
            context=context,
            frame_rate=frame_rate,
            area_sqm=area_sqm,
        )

        # 2. Occupancy Monitoring
        active_count = len([t for t in zone_tracks if getattr(t, "state", "active") == "active"])
        occupancy_res = self.occupancy_monitor.analyze(
            count=active_count,
            capacity=capacity,
            timestamp=context.timestamp,
            zone_id=context.zone_id,
            projected_count_5min=density_res.projected_occupancy_5min,
        )

        # 3. Movement & Velocity Field Dynamics
        movement_res = self.movement_analyzer.analyze(
            tracks=zone_tracks,
            context=context,
            frame_rate=frame_rate,
        )

        # 4. Flow & Net Accumulation
        flow_res = self.flow_analyzer.analyze(
            tracks=zone_tracks,
            context=context,
            frame_rate=frame_rate,
            entry_lines=entry_lines,
            exit_lines=exit_lines,
        )

        # 5. Bottleneck & Choke Point Detection
        bottleneck_res = self.bottleneck_detector.analyze(
            density_result=density_res,
            flow_result=flow_res,
            movement_result=movement_res,
            context=context,
            occupancy_result=occupancy_res,
        )

        # Compute normalized 0.0 - 1.0 factor scores
        d_score = density_res.normalized_score
        occ_score = occupancy_res.normalized_score
        m_score = movement_res.normalized_score
        f_score = flow_res.normalized_score
        b_score = bottleneck_res.normalized_score

        # Composite 5-Factor Risk Weighting (aligned with Risk Engine: 0.30, 0.20, 0.20, 0.15, 0.15)
        composite_score = round(
            (0.30 * d_score + 0.20 * occ_score + 0.20 * m_score + 0.15 * f_score + 0.15 * b_score) * 100.0,
            2,
        )

        if composite_score >= 70.0:
            risk_level = "critical"
        elif composite_score >= 40.0:
            risk_level = "warning"
        else:
            risk_level = "safe"

        return ZoneAnalyticsSummary(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            density=density_res,
            occupancy=occupancy_res,
            movement=movement_res,
            flow=flow_res,
            bottleneck=bottleneck_res,
            density_score_0_1=d_score,
            occupancy_score_0_1=occ_score,
            movement_score_0_1=m_score,
            flow_score_0_1=f_score,
            bottleneck_score_0_1=b_score,
            composite_risk_score=composite_score,
            risk_level=risk_level,
        )

    def analyze_scene(
        self,
        tracks: List[Any],
        frame_number: int,
        timestamp: float,
        zones: Optional[Dict[str, Any]] = None,
        frame_rate: float = 30.0,
    ) -> SceneAnalyticsSummary:
        """
        Execute crowd analytics across all calibrated zones and global scene.
        """
        active_tracks = [t for t in tracks if getattr(t, "state", "active") == "active"]
        val_tracks = [t for t in active_tracks if getattr(t, "is_validated", False)]

        zone_summaries: Dict[str, ZoneAnalyticsSummary] = {}
        densities = []
        risk_scores = []
        highest_zone = None
        max_risk = -1.0

        if zones:
            for zid, zdef in zones.items():
                if isinstance(zdef, dict):
                    poly = zdef.get("image_polygon")
                    cap = zdef.get("max_capacity") or zdef.get("capacity", 100)
                    area = zdef.get("area_sqm") or zdef.get("area_m2")
                    entry_lines = [{"start": zdef["entry_line"][0], "end": zdef["entry_line"][1]}] if zdef.get("entry_line") else None
                    exit_lines = [{"start": zdef["exit_line"][0], "end": zdef["exit_line"][1]}] if zdef.get("exit_line") else None
                else:
                    poly = getattr(zdef, "image_polygon", None)
                    cap = getattr(zdef, "max_capacity", getattr(zdef, "capacity", 100))
                    area = getattr(zdef, "area_sqm", getattr(zdef, "area_m2", None))
                    el = getattr(zdef, "entry_line", None)
                    xl = getattr(zdef, "exit_line", None)
                    entry_lines = [{"start": el[0], "end": el[1]}] if el and len(el) == 2 else None
                    exit_lines = [{"start": xl[0], "end": xl[1]}] if xl and len(xl) == 2 else None

                ctx = AnalyticsContext(
                    zone_id=zid,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    zone_polygon=poly,
                    zone_capacity=cap,
                    area_sqm=area,
                )
                summary = self.analyze_zone(
                    tracks=active_tracks,
                    context=ctx,
                    capacity=cap,
                    area_sqm=area,
                    frame_rate=frame_rate,
                    entry_lines=entry_lines,
                    exit_lines=exit_lines,
                )
                zone_summaries[zid] = summary
                densities.append(summary.density.avg_density)
                risk_scores.append(summary.composite_risk_score)
                if summary.composite_risk_score > max_risk:
                    max_risk = summary.composite_risk_score
                    highest_zone = zid
        else:
            # Default global scene zone
            ctx = AnalyticsContext(
                zone_id="scene_global",
                frame_number=frame_number,
                timestamp=timestamp,
            )
            summary = self.analyze_zone(
                tracks=active_tracks,
                context=ctx,
                capacity=100,
                frame_rate=frame_rate,
            )
            zone_summaries["scene_global"] = summary
            densities.append(summary.density.avg_density)
            risk_scores.append(summary.composite_risk_score)
            max_risk = summary.composite_risk_score
            highest_zone = "scene_global"

        scene_density_avg = round(float(np.mean(densities)), 3) if densities else 0.0
        scene_max_density = round(float(np.max(densities)), 3) if densities else 0.0
        scene_risk_score = round(float(np.max(risk_scores)), 2) if risk_scores else 0.0

        if scene_risk_score >= 70.0:
            scene_risk_level = "critical"
        elif scene_risk_score >= 40.0:
            scene_risk_level = "warning"
        else:
            scene_risk_level = "safe"

        return SceneAnalyticsSummary(
            timestamp=timestamp,
            frame_number=frame_number,
            total_active_tracks=len(active_tracks),
            total_validated_tracks=len(val_tracks),
            zone_summaries=zone_summaries,
            scene_density_avg=scene_density_avg,
            scene_max_density=scene_max_density,
            scene_risk_score=scene_risk_score,
            scene_risk_level=scene_risk_level,
            highest_risk_zone=highest_zone,
        )

    @staticmethod
    def to_risk_factors(
        summary: Optional[ZoneAnalyticsSummary] = None,
        density_res: Optional[DensityResult] = None,
        occupancy_res: Optional[OccupancyResult] = None,
        movement_res: Optional[MovementResult] = None,
        flow_res: Optional[FlowResult] = None,
        bottleneck_res: Optional[BottleneckResult] = None,
    ) -> Dict[str, Any]:
        """Extract normalized factor dictionary and factor list ready for Risk Engine ingestion."""
        if summary is not None:
            d = summary.density
            o = summary.occupancy
            m = summary.movement
            f = summary.flow
            b = summary.bottleneck
        else:
            d = density_res
            o = occupancy_res
            m = movement_res
            f = flow_res
            b = bottleneck_res

        d_score = float(getattr(d, "risk_score", 0.0))
        o_score = float(getattr(o, "risk_score", 0.0))
        m_score = float(getattr(m, "risk_score", 0.0))
        f_score = float(getattr(f, "risk_score", 0.0))
        b_score = float(getattr(b, "risk_score", 0.0))
        overall = round(d_score * 0.30 + o_score * 0.20 + m_score * 0.20 + f_score * 0.15 + b_score * 0.15, 2)

        factor_list = [
            {"name": "density", "value": getattr(d, "max_density", 0.0), "weight": 0.30, "score": d_score, "threshold_warning": 3.0, "threshold_critical": 5.0},
            {"name": "occupancy", "value": getattr(o, "occupancy_ratio", 0.0), "weight": 0.20, "score": o_score, "threshold_warning": 0.70, "threshold_critical": 0.90},
            {"name": "movement", "value": getattr(m, "avg_speed", 0.0), "weight": 0.20, "score": m_score, "threshold_warning": 1.0, "threshold_critical": 0.5},
            {"name": "flow", "value": getattr(f, "flow_rate", 0.0), "weight": 0.15, "score": f_score, "threshold_warning": 5.0, "threshold_critical": 10.0},
            {"name": "bottleneck", "value": getattr(b, "bottleneck_score", 0.0), "weight": 0.15, "score": b_score, "threshold_warning": 0.50, "threshold_critical": 0.80},
        ]

        return {
            "density": d_score,
            "occupancy": o_score,
            "movement": m_score,
            "flow": f_score,
            "bottleneck": b_score,
            "overall_risk": overall,
            "factors": factor_list,
        }


__all__ = ["CrowdAnalyticsEngine"]
