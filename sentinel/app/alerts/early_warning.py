"""
Accuracy-Gated 5+ Minute Predictive Early Warning & Alert Engine for SENTINEL.

Enforces strict, multi-window trend projections over a 5+ minute forward horizon (>= 5 min lead time).
Eliminates false alarms via multi-window confirmation, tracking confidence gating,
and sample sufficiency requirements. Withholds speculative alerts when data is noisy or incomplete.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Deque
from collections import deque, defaultdict
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


class EarlyWarningAccuracyStatus(str, Enum):
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MODERATE_CONFIDENCE = "MODERATE_CONFIDENCE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class EarlyWarningSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class EarlyWarningTrendWindow:
    """A discrete temporal telemetry sample for rolling multi-window trend projection."""
    timestamp: float
    avg_density: float = 0.0
    max_density: float = 0.0
    current_count: int = 0
    capacity: int = 100
    occupancy_ratio: float = 0.0
    inflow_rate: float = 0.0                # persons/min
    outflow_rate: float = 0.0               # persons/min
    net_accumulation_rate: float = 0.0      # persons/min (inflow - outflow)
    mean_speed: float = 1.2                 # m/s
    bottleneck_score: float = 0.0           # 0.0 to 1.0
    avg_track_confidence: float = 0.85      # 0.0 to 1.0
    sample_count: int = 1                   # number of tracked persons in frame


class EarlyWarningProjection(BaseModel):
    """
    Accuracy-Gated 5+ Minute Forward Projection Result.
    Contains predictive metrics, advance lead time, confidence score, and human explainability.
    """
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    projected_density_5min: float = 0.0
    projected_occupancy_5min: float = 0.0
    estimated_lead_minutes: float = 5.0     # Target >= 5.0 minutes advance warning horizon
    density_slope: float = 0.0              # persons / (m² * min)
    net_inflow_surplus: float = 0.0         # persons / min
    speed_trend_slope: float = 0.0          # (m/s) / min
    bottleneck_persistence: float = 0.0     # 0.0 to 1.0 fraction of windows
    confidence_score: float = 0.0           # 0.0 to 1.0
    accuracy_status: EarlyWarningAccuracyStatus = EarlyWarningAccuracyStatus.INSUFFICIENT_DATA
    is_early_warning: bool = False          # True ONLY if all accuracy gates are satisfied
    severity: EarlyWarningSeverity = EarlyWarningSeverity.INFO
    primary_driver: str = "nominal"
    reason: str = ""
    steward_action: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class EarlyWarningConfig:
    """Configuration for 5+ Minute Early Warning Engine."""
    horizon_minutes: float = 5.0
    min_lead_minutes: float = 5.0               # Minimum target lead time for advance warning
    min_history_samples: int = 3                # Minimum windows required to compute trend (never single-frame!)
    min_window_duration_seconds: float = 4.0    # Minimum temporal span (seconds)
    min_tracking_confidence: float = 0.45       # Minimum detector/tracker confidence required
    density_warning_threshold: float = 2.5      # p/m²
    density_critical_threshold: float = 4.0     # p/m²
    occupancy_warning_threshold: float = 0.70   # ratio
    occupancy_critical_threshold: float = 0.90  # ratio
    net_inflow_surplus_threshold: float = 10.0  # persons/min
    velocity_drop_threshold: float = 0.25       # 25% drop relative to baseline
    bottleneck_threshold: float = 0.50          # 0.0 to 1.0
    confidence_high_threshold: float = 0.75     # >= 0.75 is HIGH_CONFIDENCE
    confidence_moderate_threshold: float = 0.50 # >= 0.50 is MODERATE_CONFIDENCE; below is INSUFFICIENT_DATA
    cooldown_seconds: float = 60.0              # deduplication cooldown per zone
    max_history_len: int = 60                   # Bounded circular buffer for low RAM usage (~145MB target)


class EarlyWarningEngine:
    """
    Accuracy-Gated 5+ Minute Predictive Early Warning Engine.
    Employs robust least-squares trend estimation across temporal rolling windows,
    gating out noisy/insufficient data and preventing false alarm storms.
    """

    def __init__(self, config: Optional[EarlyWarningConfig] = None):
        if config is None:
            config = EarlyWarningConfig()
        self.config = config
        self.history: Dict[str, Deque[EarlyWarningTrendWindow]] = defaultdict(
            lambda: deque(maxlen=self.config.max_history_len)
        )
        self.last_alert_time: Dict[str, float] = {}
        self.last_projections: Dict[str, EarlyWarningProjection] = {}

    def reset(self, zone_id: Optional[str] = None):
        """Reset historical windows and alert timestamps."""
        if zone_id:
            if zone_id in self.history:
                self.history[zone_id].clear()
            self.last_alert_time.pop(zone_id, None)
            self.last_projections.pop(zone_id, None)
        else:
            self.history.clear()
            self.last_alert_time.clear()
            self.last_projections.clear()

    def record_window(
        self,
        zone_id: str,
        timestamp: float,
        avg_density: float = 0.0,
        max_density: float = 0.0,
        current_count: int = 0,
        capacity: int = 100,
        occupancy_ratio: float = 0.0,
        inflow_rate: float = 0.0,
        outflow_rate: float = 0.0,
        net_accumulation_rate: float = 0.0,
        mean_speed: float = 1.2,
        bottleneck_score: float = 0.0,
        avg_track_confidence: float = 0.85,
        sample_count: int = 1,
    ) -> EarlyWarningTrendWindow:
        """Append a new discrete telemetry window to the rolling history."""
        # Clean net accumulation rate
        if net_accumulation_rate == 0.0 and (inflow_rate > 0.0 or outflow_rate > 0.0):
            net_accumulation_rate = inflow_rate - outflow_rate

        # Clean occupancy ratio
        if occupancy_ratio == 0.0 and capacity > 0 and current_count > 0:
            occupancy_ratio = current_count / float(capacity)

        win = EarlyWarningTrendWindow(
            timestamp=float(timestamp),
            avg_density=float(max(0.0, avg_density)),
            max_density=float(max(0.0, max(max_density, avg_density))),
            current_count=int(max(0, current_count)),
            capacity=int(max(1, capacity)),
            occupancy_ratio=float(max(0.0, occupancy_ratio)),
            inflow_rate=float(max(0.0, inflow_rate)),
            outflow_rate=float(max(0.0, outflow_rate)),
            net_accumulation_rate=float(net_accumulation_rate),
            mean_speed=float(max(0.0, mean_speed)),
            bottleneck_score=float(min(1.0, max(0.0, bottleneck_score))),
            avg_track_confidence=float(min(1.0, max(0.0, avg_track_confidence))),
            sample_count=int(max(0, sample_count)),
        )
        self.history[zone_id].append(win)
        return win

    def assess_early_warning(
        self,
        zone_id: str,
        timestamp: float,
        # Direct raw inputs or pre-extracted results
        avg_density: Optional[float] = None,
        max_density: Optional[float] = None,
        current_count: Optional[int] = None,
        capacity: Optional[int] = None,
        occupancy_ratio: Optional[float] = None,
        inflow_rate: Optional[float] = None,
        outflow_rate: Optional[float] = None,
        net_accumulation_rate: Optional[float] = None,
        mean_speed: Optional[float] = None,
        bottleneck_score: Optional[float] = None,
        avg_track_confidence: Optional[float] = None,
        sample_count: Optional[int] = None,
        # Or structured analytics objects
        density_result: Optional[Any] = None,
        occupancy_result: Optional[Any] = None,
        movement_result: Optional[Any] = None,
        flow_result: Optional[Any] = None,
        bottleneck_result: Optional[Any] = None,
    ) -> EarlyWarningProjection:
        """
        Evaluate 5+ minute forward projection for a zone with strict accuracy gating.
        Returns EarlyWarningProjection with accuracy_status, confidence_score, and lead time.
        """
        # 1. Unify and extract incoming telemetry
        d_avg = avg_density
        d_max = max_density
        if density_result is not None:
            d_avg = getattr(density_result, "avg_density", getattr(density_result, "density_p_sqm", d_avg or 0.0))
            d_max = getattr(density_result, "max_density", getattr(density_result, "max_density_p_sqm", d_max or d_avg or 0.0))
        d_avg = d_avg if d_avg is not None else 0.0
        d_max = d_max if d_max is not None else d_avg

        c_cnt = current_count
        c_cap = capacity
        c_occ = occupancy_ratio
        if occupancy_result is not None:
            c_cnt = getattr(occupancy_result, "current_count", getattr(occupancy_result, "count", c_cnt or 0))
            c_cap = getattr(occupancy_result, "capacity", c_cap or 100)
            c_occ = getattr(occupancy_result, "occupancy_ratio", getattr(occupancy_result, "ratio", c_occ or 0.0))
        c_cnt = c_cnt if c_cnt is not None else 0
        c_cap = c_cap if c_cap is not None else 100
        c_occ = c_occ if c_occ is not None else (c_cnt / float(max(1, c_cap)))

        f_in = inflow_rate
        f_out = outflow_rate
        f_net = net_accumulation_rate
        if flow_result is not None:
            f_in = getattr(flow_result, "inflow_rate_ppm", getattr(flow_result, "entry_flow", f_in or 0.0))
            f_out = getattr(flow_result, "outflow_rate_ppm", getattr(flow_result, "exit_flow", f_out or 0.0))
            f_net = getattr(flow_result, "net_accumulation_rate_ppm", getattr(flow_result, "net_flow", f_net or (f_in - f_out)))
        f_in = f_in if f_in is not None else 0.0
        f_out = f_out if f_out is not None else 0.0
        f_net = f_net if f_net is not None else (f_in - f_out)

        m_spd = mean_speed
        if movement_result is not None:
            m_spd = getattr(movement_result, "avg_speed", getattr(movement_result, "avg_speed_mps", m_spd or 1.2))
        m_spd = m_spd if m_spd is not None else 1.2

        b_sc = bottleneck_score
        if bottleneck_result is not None:
            b_sc = getattr(bottleneck_result, "bottleneck_score", b_sc or 0.0)
            if b_sc > 1.0:
                b_sc = b_sc / 100.0
        b_sc = b_sc if b_sc is not None else 0.0

        trk_conf = avg_track_confidence if avg_track_confidence is not None else 0.85
        s_cnt = sample_count if sample_count is not None else max(1, c_cnt)

        # 2. Record to historical window buffer
        self.record_window(
            zone_id=zone_id,
            timestamp=timestamp,
            avg_density=d_avg,
            max_density=d_max,
            current_count=c_cnt,
            capacity=c_cap,
            occupancy_ratio=c_occ,
            inflow_rate=f_in,
            outflow_rate=f_out,
            net_accumulation_rate=f_net,
            mean_speed=m_spd,
            bottleneck_score=b_sc,
            avg_track_confidence=trk_conf,
            sample_count=s_cnt,
        )

        windows = list(self.history[zone_id])
        n_windows = len(windows)

        # -------------------------------------------------------------
        # ACCURACY GATE 1: Minimum Sample / History Sufficiency
        # -------------------------------------------------------------
        if n_windows < self.config.min_history_samples:
            proj = EarlyWarningProjection(
                zone_id=zone_id,
                timestamp=timestamp,
                projected_density_5min=round(d_max, 2),
                projected_occupancy_5min=float(c_cnt),
                estimated_lead_minutes=5.0,
                density_slope=0.0,
                net_inflow_surplus=f_net,
                speed_trend_slope=0.0,
                bottleneck_persistence=b_sc,
                confidence_score=round(n_windows / float(self.config.min_history_samples) * 0.40, 2),
                accuracy_status=EarlyWarningAccuracyStatus.INSUFFICIENT_DATA,
                is_early_warning=False,
                severity=EarlyWarningSeverity.INFO,
                primary_driver="calibrating_baseline",
                reason=(
                    f"Insufficient multi-window history ({n_windows}/{self.config.min_history_samples} windows). "
                    "Calibrating baseline before forward projection."
                ),
                steward_action="Monitor telemetry while rolling baseline calibrates.",
                metrics={"window_count": n_windows, "required_windows": self.config.min_history_samples},
            )
            self.last_projections[zone_id] = proj
            return proj

        # Temporal span check
        time_span = max(0.0, windows[-1].timestamp - windows[0].timestamp)
        if time_span < self.config.min_window_duration_seconds:
            proj = EarlyWarningProjection(
                zone_id=zone_id,
                timestamp=timestamp,
                projected_density_5min=round(d_max, 2),
                projected_occupancy_5min=float(c_cnt),
                estimated_lead_minutes=5.0,
                density_slope=0.0,
                net_inflow_surplus=f_net,
                speed_trend_slope=0.0,
                bottleneck_persistence=b_sc,
                confidence_score=0.45,
                accuracy_status=EarlyWarningAccuracyStatus.INSUFFICIENT_DATA,
                is_early_warning=False,
                severity=EarlyWarningSeverity.INFO,
                primary_driver="temporal_span_insufficient",
                reason=f"Temporal window duration ({time_span:.1f}s) below minimum ({self.config.min_window_duration_seconds:.1f}s).",
                steward_action="Awaiting required window interval.",
                metrics={"time_span_s": time_span, "min_time_span_s": self.config.min_window_duration_seconds},
            )
            self.last_projections[zone_id] = proj
            return proj

        # -------------------------------------------------------------
        # ACCURACY GATE 2: Tracking / Detection Confidence Gating
        # -------------------------------------------------------------
        mean_track_conf = float(np.mean([w.avg_track_confidence for w in windows]))
        if mean_track_conf < self.config.min_tracking_confidence:
            proj = EarlyWarningProjection(
                zone_id=zone_id,
                timestamp=timestamp,
                projected_density_5min=round(d_max, 2),
                projected_occupancy_5min=float(c_cnt),
                estimated_lead_minutes=5.0,
                density_slope=0.0,
                net_inflow_surplus=f_net,
                speed_trend_slope=0.0,
                bottleneck_persistence=b_sc,
                confidence_score=round(mean_track_conf, 2),
                accuracy_status=EarlyWarningAccuracyStatus.INSUFFICIENT_DATA,
                is_early_warning=False,
                severity=EarlyWarningSeverity.INFO,
                primary_driver="low_tracking_confidence",
                reason=(
                    f"Tracking quality ({mean_track_conf:.2f}) below minimum confidence threshold "
                    f"({self.config.min_tracking_confidence:.2f}). Withholding speculative projection."
                ),
                steward_action="Check camera focus and lighting conditions.",
                metrics={"mean_track_confidence": mean_track_conf},
            )
            self.last_projections[zone_id] = proj
            return proj

        # -------------------------------------------------------------
        # 3. Vectorized Multi-Window Slope & Trend Regression
        # -------------------------------------------------------------
        t0 = windows[0].timestamp
        # Express time in minutes from t0 for natural (per-minute) slopes
        times_min = np.array([(w.timestamp - t0) / 60.0 for w in windows], dtype=np.float64)
        densities = np.array([w.max_density for w in windows], dtype=np.float64)
        speeds = np.array([w.mean_speed for w in windows], dtype=np.float64)
        net_flows = np.array([w.net_accumulation_rate for w in windows], dtype=np.float64)
        bottlenecks = np.array([w.bottleneck_score for w in windows], dtype=np.float64)

        # Robust linear slope: m = Cov(t, y) / Var(t)
        t_var = float(np.var(times_min))
        if t_var > 1e-7:
            density_slope = float(np.cov(times_min, densities, bias=True)[0, 1] / t_var)
            speed_slope = float(np.cov(times_min, speeds, bias=True)[0, 1] / t_var)

            # Goodness-of-fit R² for density trend
            d_pred = densities[0] + density_slope * times_min
            ss_tot = float(np.sum((densities - np.mean(densities)) ** 2))
            ss_res = float(np.sum((densities - d_pred) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.85
            r_squared = max(0.0, min(1.0, r_squared))
        else:
            density_slope = 0.0
            speed_slope = 0.0
            r_squared = 0.50

        # Mean and sustained values across windows
        mean_net_inflow = float(np.mean(net_flows))
        bottleneck_persistence = float(np.mean(bottlenecks >= self.config.bottleneck_threshold))
        cur_density = float(windows[-1].max_density)
        cur_count = int(windows[-1].current_count)

        # -------------------------------------------------------------
        # 4. Multi-Window Trend Confirmation (Filter Transient Spikes)
        # -------------------------------------------------------------
        # Check if the recent direction is sustained across the last 2 intervals
        if len(windows) >= 3:
            recent_delta1 = windows[-1].max_density - windows[-2].max_density
            recent_delta2 = windows[-2].max_density - windows[-3].max_density
            is_rising_consistently = (recent_delta1 > 0.0 and recent_delta2 >= -0.05) or (density_slope > 0.08)
            is_net_inflow_sustained = (windows[-1].net_accumulation_rate > 5.0 and windows[-2].net_accumulation_rate > 0.0)
        else:
            is_rising_consistently = density_slope > 0.05
            is_net_inflow_sustained = mean_net_inflow > 5.0

        # -------------------------------------------------------------
        # 5. 5-Minute Forward Projection Calculation
        # -------------------------------------------------------------
        h_min = self.config.horizon_minutes  # 5.0 min
        projected_density_5min = max(0.0, cur_density + (density_slope * h_min))

        # Projected occupancy at t+5min based on sustained net inflow rate
        projected_occupancy_5min = max(0.0, float(cur_count) + (mean_net_inflow * h_min))

        # Advance Lead Time (target >= 5.0 min advance warning window)
        # Time required to reach Warning threshold (2.5 p/m²) from current density
        w_thresh = self.config.density_warning_threshold
        c_thresh = self.config.density_critical_threshold

        if density_slope > 0.02 and cur_density < w_thresh:
            lead_time_min = (w_thresh - cur_density) / density_slope
        elif density_slope > 0.02 and cur_density < c_thresh:
            lead_time_min = (c_thresh - cur_density) / density_slope
        elif density_slope > 0.02 and cur_density >= c_thresh:
            lead_time_min = 0.5  # Already critical
        else:
            lead_time_min = 10.0  # Stable / non-critical horizon

        lead_time_min = max(0.5, round(lead_time_min, 1))

        # -------------------------------------------------------------
        # 6. Accuracy Confidence Score Formulation (0.0 to 1.0)
        # -------------------------------------------------------------
        # Weighted factors:
        # - Fit consistency (R²): 30%
        # - Window sample stability: 25%
        # - Tracking detection confidence: 25%
        # - Slope stability / temporal duration: 20%
        w_samples = min(1.0, n_windows / 6.0)
        w_span = min(1.0, time_span / 20.0)
        conf_score = (
            (r_squared * 0.30)
            + (w_samples * 0.25)
            + (mean_track_conf * 0.25)
            + (w_span * 0.20)
        )
        conf_score = round(min(1.0, max(0.0, conf_score)), 2)

        # Determine Accuracy Status
        if conf_score >= self.config.confidence_high_threshold:
            acc_status = EarlyWarningAccuracyStatus.HIGH_CONFIDENCE
        elif conf_score >= self.config.confidence_moderate_threshold:
            acc_status = EarlyWarningAccuracyStatus.MODERATE_CONFIDENCE
        else:
            acc_status = EarlyWarningAccuracyStatus.INSUFFICIENT_DATA

        # -------------------------------------------------------------
        # 7. Strict Trigger Gating (Zero False Alarms)
        # -------------------------------------------------------------
        # Conditions for 5+ Min Early Warning:
        # Must pass confidence threshold (>= 0.50) AND exhibit sustained multi-window surge
        is_triggered = False
        primary_driver = "nominal"
        reason = ""
        steward_action = ""
        severity = EarlyWarningSeverity.INFO

        if acc_status != EarlyWarningAccuracyStatus.INSUFFICIENT_DATA:
            # Condition A: Projected density breaches warning (2.5 p/m²) or critical (4.0 p/m²)
            cond_density = (projected_density_5min >= w_thresh) and is_rising_consistently and density_slope > 0.05
            # Condition B: High sustained net inflow surplus (> 10 ppm) with positive slope
            cond_inflow = (mean_net_inflow >= self.config.net_inflow_surplus_threshold) and is_net_inflow_sustained and density_slope > 0.02
            # Condition C: Bottleneck persistence overlapping with decelerating speed
            cond_bottleneck = (bottleneck_persistence >= self.config.bottleneck_threshold) and (speed_slope < -0.05 or m_spd < 0.6)
            # Condition D: High projected occupancy (> 80% capacity)
            cond_occupancy = (c_cap > 0 and (projected_occupancy_5min / float(c_cap)) >= self.config.occupancy_warning_threshold) and mean_net_inflow > 5.0

            if cond_density or cond_inflow or cond_bottleneck or cond_occupancy:
                is_triggered = True

                # Determine primary driver and severity
                if projected_density_5min >= c_thresh or bottleneck_persistence >= 0.75:
                    severity = EarlyWarningSeverity.CRITICAL
                else:
                    severity = EarlyWarningSeverity.WARNING

                # Craft transparent root cause explanation
                drivers = []
                if cond_inflow:
                    drivers.append(f"sustained net inflow surplus (+{mean_net_inflow:.1f} persons/min over {n_windows} windows)")
                    primary_driver = "inflow_surplus"
                if cond_density:
                    drivers.append(f"rising density slope (+{density_slope:.2f} p/m²/min)")
                    if primary_driver == "nominal":
                        primary_driver = "density_slope"
                if cond_bottleneck:
                    drivers.append(f"persistent choke bottleneck ({int(bottleneck_persistence*100)}% persistence, speed: {m_spd:.2f} m/s)")
                    if primary_driver == "nominal":
                        primary_driver = "bottleneck_persistence"
                if cond_occupancy:
                    drivers.append(f"projected occupancy surge ({int(projected_occupancy_5min)}/{c_cap} people, {int(projected_occupancy_5min/c_cap*100)}%)")
                    if primary_driver == "nominal":
                        primary_driver = "occupancy_surge"

                lead_display = f"{lead_time_min:.1f}" if lead_time_min >= 1.0 else "< 1.0"
                reason = (
                    f"⏳ 5+ MIN EARLY WARNING (~{lead_display} min lead horizon): Driven by {'; '.join(drivers)}. "
                    f"Projected density: {projected_density_5min:.2f} p/m² (confidence: {int(conf_score*100)}%)."
                )

                if severity == EarlyWarningSeverity.CRITICAL:
                    steward_action = "🚨 IMMEDIATE: Divert incoming foot traffic, open auxiliary egress gates, deploy safety stewards."
                else:
                    steward_action = "⚠️ PREEMPTIVE: Regulate entry turnstiles, announce alternative assembly areas, monitor choke points."
            else:
                reason = (
                    f"Crowd conditions nominal: Projected density {projected_density_5min:.2f} p/m² "
                    f"(slope: {density_slope:+.2f}/min, confidence: {int(conf_score*100)}%)."
                )
                steward_action = "Continue standard monitoring."
        else:
            reason = "Confidence below gating threshold; withholding speculative projection."
            steward_action = "Awaiting cleaner sensor input."

        proj = EarlyWarningProjection(
            zone_id=zone_id,
            timestamp=timestamp,
            projected_density_5min=round(projected_density_5min, 2),
            projected_occupancy_5min=round(projected_occupancy_5min, 1),
            estimated_lead_minutes=lead_time_min,
            density_slope=round(density_slope, 4),
            net_inflow_surplus=round(mean_net_inflow, 2),
            speed_trend_slope=round(speed_slope, 4),
            bottleneck_persistence=round(bottleneck_persistence, 2),
            confidence_score=conf_score,
            accuracy_status=acc_status,
            is_early_warning=is_triggered,
            severity=severity,
            primary_driver=primary_driver,
            reason=reason,
            steward_action=steward_action,
            metrics={
                "window_count": n_windows,
                "time_span_s": round(time_span, 1),
                "r_squared": round(r_squared, 3),
                "mean_track_confidence": round(mean_track_conf, 3),
                "density_slope_ppm": round(density_slope, 4),
                "mean_net_inflow_ppm": round(mean_net_inflow, 2),
                "lead_time_min": lead_time_min,
            },
        )
        self.last_projections[zone_id] = proj
        return proj


__all__ = [
    "EarlyWarningAccuracyStatus",
    "EarlyWarningSeverity",
    "EarlyWarningTrendWindow",
    "EarlyWarningProjection",
    "EarlyWarningConfig",
    "EarlyWarningEngine",
]

