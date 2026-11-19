from typing import Optional, List, Literal, Tuple, Dict, Any, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


class CountingResult(BaseModel):
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    current_count: int
    entries: int
    exits: int
    unique_tracks: int
    net_count: int = 0
    validated_count: Optional[int] = None
    inflow_rate_ppm: float = 0.0
    outflow_rate_ppm: float = 0.0
    net_flow_rate_ppm: float = 0.0
    zone_counts: dict[str, int] = Field(default_factory=dict)
    zone_densities: dict[str, float] = Field(default_factory=dict)


class SceneCountReport(BaseModel):
    """
    Comprehensive multi-zone crowd counting report for an entire scene/frame.
    Synchronizes with Appearance Re-ID, Second-Stage Validation, and Crowd Analytics.
    """
    model_config = {"extra": "ignore"}

    timestamp: float = 0.0
    frame_number: int = 0
    total_active_count: int = 0
    total_validated_count: int = 0
    total_tentative_count: int = 0
    rejected_count: int = 0
    reid_matches: int = 0

    # Zone breakdowns
    zone_counts: dict[str, int] = Field(default_factory=dict)
    zone_densities: dict[str, float] = Field(default_factory=dict)
    zone_occupancies: dict[str, float] = Field(default_factory=dict)
    zone_status: dict[str, str] = Field(default_factory=dict)  # safe, warning, critical

    # Ingress / Egress / Net Flow rates (persons per minute)
    total_entries: int = 0
    total_exits: int = 0
    net_flow: int = 0
    inflow_rate_ppm: float = 0.0
    outflow_rate_ppm: float = 0.0
    net_flow_rate_ppm: float = 0.0

    # Track ID sets for auditability
    active_track_ids: list[int] = Field(default_factory=list)
    validated_track_ids: list[int] = Field(default_factory=list)


class DensityGrid(BaseModel):
    grid_data: List[List[float]]
    grid_width: int
    grid_height: int
    cell_size_meters: float
    max_density: float


class DensityResult(BaseModel):
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    grid: Optional[DensityGrid] = None
    avg_density: float = 0.0
    max_density: float = 0.0
    density_rate_of_change: float = 0.0          # d(density)/dt in persons/(m² * min)
    projected_density_5min: float = 0.0          # Trend-based predictive estimate at t+5min (p/m²)
    projected_occupancy_5min: float = 0.0        # Trend-based predictive occupancy count at t+5min
    prediction_horizon_min: float = 5.0          # 5-minute prediction horizon
    lead_time_to_critical_min: Optional[float] = None  # Estimated minutes until warning/critical breach
    trend_label: str = "stable"                  # "rising", "stable", "falling"
    calibrated_area_sqm: Optional[float] = None  # Calibrated area in m²
    risk_score: float = 0.0                      # 0 - 100 risk score
    normalized_score: float = 0.0                # 0.0 - 1.0 normalized score

    # Aliases for flexible API consumption
    density_p_sqm: Optional[float] = None
    max_density_p_sqm: Optional[float] = None
    rate_of_change_ppm: Optional[float] = None
    projected_density_5m: Optional[float] = None
    lead_time_min: Optional[float] = None
    area_sqm: Optional[float] = None
    status: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.density_p_sqm is None:
            self.density_p_sqm = self.avg_density
        if self.max_density_p_sqm is None:
            self.max_density_p_sqm = self.max_density
        if self.rate_of_change_ppm is None:
            self.rate_of_change_ppm = self.density_rate_of_change
        if self.projected_density_5m is None:
            self.projected_density_5m = self.projected_density_5min
        if self.lead_time_min is None:
            self.lead_time_min = self.lead_time_to_critical_min
        if self.area_sqm is None:
            self.area_sqm = self.calibrated_area_sqm
        if self.status is None:
            self.status = "critical" if self.max_density >= 5.0 else ("warning" if self.max_density >= 3.0 else "safe")


class OccupancyResult(BaseModel):
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    capacity: int
    current_count: int
    occupancy_ratio: float
    status: Literal["safe", "warning", "critical"]
    headroom: int = 0                            # Remaining safe headcount
    headroom_ratio: float = 1.0                  # headroom / capacity
    projected_occupancy_5min: Optional[int] = None
    risk_score: float = 0.0                      # 0 - 100 risk score
    normalized_score: float = 0.0                # 0.0 - 1.0 normalized score


class MovementVector(BaseModel):
    track_id: int
    centroid: List[float]
    velocity: List[float]
    speed: float
    direction: float


class MovementResult(BaseModel):
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    movements: List[MovementVector] = []
    avg_speed: float = 0.0
    avg_direction: Optional[float] = None
    dominant_direction: Optional[float] = None
    counter_flow_index: float = 0.0              # 0.0 to 1.0 (proportion of tracks in counter-flow)
    directional_convergence: float = 0.0         # 0.0 to 1.0 (funneling/convergence toward choke/centroid)
    velocity_drop_ratio: float = 0.0             # 0.0 to 1.0 (slowdown relative to baseline speed)
    speed_trend: str = "stable"                  # "increasing", "stable", "slowing"
    crossing_flow_detected: bool = False         # True when crossing-flow conflicts exist
    stagnation_ratio: float = 0.0                # Fraction of tracks with speed < threshold (0.0 to 1.0)
    low_movement_count: int = 0                  # Count of stagnant/low-movement tracks
    risk_score: float = 0.0                      # 0 - 100 risk score
    normalized_score: float = 0.0                # 0.0 - 1.0 normalized score

    # Aliases for flexible API consumption
    avg_speed_mps: Optional[float] = None
    avg_heading_deg: Optional[float] = None
    sudden_drop_detected: Optional[bool] = None

    def model_post_init(self, __context: Any) -> None:
        if self.avg_speed_mps is None:
            self.avg_speed_mps = self.avg_speed
        if self.avg_heading_deg is None:
            self.avg_heading_deg = self.avg_direction
        if self.sudden_drop_detected is None:
            self.sudden_drop_detected = self.velocity_drop_ratio >= 0.30


class FlowResult(BaseModel):
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    flow_rate: float = 0.0
    flow_direction: Optional[float] = None
    entry_flow: float = 0.0
    exit_flow: float = 0.0
    net_flow: float = 0.0
    net_accumulation_rate_ppm: float = 0.0       # Net crowd accumulation in persons/min
    inflow_outflow_ratio: float = 1.0            # Ingress vs egress flow ratio
    flow_conflict_detected: bool = False         # Inflow surplus with counter-flow condition
    counter_flow_conflict_score: float = 0.0     # 0.0 to 1.0 counter-flow conflict score
    crossing_flow_conflict_score: float = 0.0    # 0.0 to 1.0 crossing-flow conflict score
    risk_score: float = 0.0                      # 0 - 100 risk score
    normalized_score: float = 0.0                # 0.0 - 1.0 normalized score

    # Aliases for flexible API consumption
    inflow_rate_ppm: Optional[float] = None
    outflow_rate_ppm: Optional[float] = None
    net_accumulation_ppm: Optional[float] = None

    def model_post_init(self, __context: Any) -> None:
        if self.inflow_rate_ppm is None:
            self.inflow_rate_ppm = self.entry_flow
        if self.outflow_rate_ppm is None:
            self.outflow_rate_ppm = self.exit_flow
        if self.net_accumulation_ppm is None:
            self.net_accumulation_ppm = self.net_accumulation_rate_ppm


class BottleneckResult(BaseModel):
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    bottleneck_score: float = 0.0
    is_bottleneck: bool = False
    congestion_level: Literal["none", "low", "medium", "high", "critical"] = "none"
    affected_area: Optional[List[float]] = None
    projected_bottleneck_5min: bool = False      # Whether bottleneck is projected within 5 mins
    early_warning_lead_min: Optional[float] = None  # Advance lead time in minutes
    stagnation_overlap: bool = False             # High density overlapping with high stagnation
    risk_score: float = 0.0                      # 0 - 100 risk score
    normalized_score: float = 0.0                # 0.0 - 1.0 normalized score

    # Aliases for flexible API consumption
    is_choke_point: Optional[bool] = None
    lead_time_min: Optional[float] = None
    early_warning_active: Optional[bool] = None

    def model_post_init(self, __context: Any) -> None:
        if self.is_choke_point is None:
            self.is_choke_point = self.is_bottleneck
        if self.lead_time_min is None:
            self.lead_time_min = self.early_warning_lead_min
        if self.early_warning_active is None:
            self.early_warning_active = self.projected_bottleneck_5min or (self.early_warning_lead_min is not None and self.early_warning_lead_min <= 5.0)


class ZoneAnalyticsSummary(BaseModel):
    """Unified five-factor crowd analytics summary for a single zone."""
    model_config = {"extra": "ignore"}

    zone_id: str
    timestamp: float
    density: DensityResult
    occupancy: OccupancyResult
    movement: MovementResult
    flow: FlowResult
    bottleneck: BottleneckResult

    # Normalized individual factor scores (0.0 to 1.0 and 0 to 100)
    density_score_0_1: float = 0.0
    occupancy_score_0_1: float = 0.0
    movement_score_0_1: float = 0.0
    flow_score_0_1: float = 0.0
    bottleneck_score_0_1: float = 0.0

    # Risk Engine aligned composite risk score (0 to 100)
    composite_risk_score: float = 0.0
    risk_level: Literal["safe", "warning", "critical"] = "safe"


class SceneAnalyticsSummary(BaseModel):
    """Unified multi-zone crowd analytics summary across entire scene."""
    model_config = {"extra": "ignore"}

    timestamp: float
    frame_number: int
    total_active_tracks: int
    total_validated_tracks: int
    zone_summaries: dict[str, ZoneAnalyticsSummary] = Field(default_factory=dict)
    scene_density_avg: float = 0.0
    scene_max_density: float = 0.0
    scene_risk_score: float = 0.0
    scene_risk_level: Literal["safe", "warning", "critical"] = "safe"
    highest_risk_zone: Optional[str] = None

    @property
    def zones(self) -> dict[str, ZoneAnalyticsSummary]:
        return self.zone_summaries

    @property
    def risk_level(self) -> str:
        return self.scene_risk_level.upper()

    @property
    def risk_score(self) -> float:
        return self.scene_risk_score

    @property
    def density(self) -> DensityResult:
        if self.highest_risk_zone and self.highest_risk_zone in self.zone_summaries:
            return self.zone_summaries[self.highest_risk_zone].density
        if self.zone_summaries:
            return next(iter(self.zone_summaries.values())).density
        return DensityResult(zone_id="scene_global", timestamp=self.timestamp)

    @property
    def occupancy(self) -> OccupancyResult:
        if self.highest_risk_zone and self.highest_risk_zone in self.zone_summaries:
            return self.zone_summaries[self.highest_risk_zone].occupancy
        if self.zone_summaries:
            return next(iter(self.zone_summaries.values())).occupancy
        return OccupancyResult(zone_id="scene_global", timestamp=self.timestamp, capacity=100, current_count=self.total_active_tracks, occupancy_ratio=0.0, status="safe")

    @property
    def movement(self) -> MovementResult:
        if self.highest_risk_zone and self.highest_risk_zone in self.zone_summaries:
            return self.zone_summaries[self.highest_risk_zone].movement
        if self.zone_summaries:
            return next(iter(self.zone_summaries.values())).movement
        return MovementResult(zone_id="scene_global", timestamp=self.timestamp)

    @property
    def flow(self) -> FlowResult:
        if self.highest_risk_zone and self.highest_risk_zone in self.zone_summaries:
            return self.zone_summaries[self.highest_risk_zone].flow
        if self.zone_summaries:
            return next(iter(self.zone_summaries.values())).flow
        return FlowResult(zone_id="scene_global", timestamp=self.timestamp)

    @property
    def bottleneck(self) -> BottleneckResult:
        if self.highest_risk_zone and self.highest_risk_zone in self.zone_summaries:
            return self.zone_summaries[self.highest_risk_zone].bottleneck
        if self.zone_summaries:
            return next(iter(self.zone_summaries.values())).bottleneck
        return BottleneckResult(zone_id="scene_global", timestamp=self.timestamp)

    @property
    def risk_factors(self) -> dict[str, float]:
        d = self.density.risk_score
        o = self.occupancy.risk_score
        m = self.movement.risk_score
        f = self.flow.risk_score
        b = self.bottleneck.risk_score
        return {
            "density": d,
            "occupancy": o,
            "movement": m,
            "flow": f,
            "bottleneck": b,
            "overall_risk": self.scene_risk_score,
        }


class AnalyticsContext(BaseModel):
    model_config = {"extra": "ignore"}

    zone_id: str
    frame_number: int
    timestamp: float
    zone_polygon: Optional[List[List[float]]] = None
    zone_capacity: int = 0
    area_sqm: Optional[float] = None
    scale_meters_per_pixel: Optional[float] = None
    homography_matrix: Optional[List[List[float]]] = None
    cell_size_meters: float = 1.0


@dataclass
class AnalyticsConfig:
    density_grid_size: float = 1.0
    occupancy_warning_threshold: float = 0.7
    occupancy_critical_threshold: float = 0.9
    flow_window_seconds: int = 60
    bottleneck_persistence_frames: int = 30
    min_tracks_for_flow: int = 5
    speed_smoothing_alpha: float = 0.3
    direction_bins: int = 8
    prediction_horizon_minutes: float = 5.0
    counter_flow_angle_threshold_deg: float = 120.0
    crossing_flow_angle_threshold_deg: float = 60.0
    velocity_drop_threshold: float = 0.30
    stagnation_speed_threshold_m_s: float = 0.20
    stagnation_speed_threshold_px_s: float = 5.0
    warning_density: float = 3.0
    critical_density: float = 5.0
    warning_occupancy: float = 0.70
    critical_occupancy: float = 0.90
    choke_density_threshold: float = 2.5
    stagnation_speed_threshold: float = 0.30
    bottleneck_window_seconds: float = 15.0


class TrackState:
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.centroid_history: List[List[float]] = []
        self.velocity_history: List[List[float]] = []
        self.zone_id: Optional[str] = None
        self.first_seen_frame: int = 0
        self.last_seen_frame: int = 0
        self.total_distance: float = 0.0
        self.is_counted_entry: bool = False
        self.is_counted_exit: bool = False

    def add_position(self, centroid: List[float], frame_number: int):
        if self.centroid_history:
            prev = np.array(self.centroid_history[-1])
            curr = np.array(centroid)
            self.total_distance += float(np.linalg.norm(curr - prev))
        self.centroid_history.append(centroid)
        self.last_seen_frame = frame_number

    def get_velocity(self, frame_rate: float = 30.0) -> List[float]:
        if len(self.centroid_history) < 2:
            return [0.0, 0.0]
        prev = np.array(self.centroid_history[-2])
        curr = np.array(self.centroid_history[-1])
        velocity = (curr - prev) * frame_rate
        return velocity.tolist()

    def get_speed(self, frame_rate: float = 30.0) -> float:
        vel = self.get_velocity(frame_rate)
        return float(np.linalg.norm(vel))

    def get_direction(self, frame_rate: float = 30.0) -> float:
        vel = self.get_velocity(frame_rate)
        if vel[0] == 0 and vel[1] == 0:
            return 0.0
        return float(np.arctan2(vel[1], vel[0]))


class ZoneCounter:
    def __init__(
        self,
        zone_id: str,
        zone_polygon: Optional[List[List[float]]] = None,
        capacity: int = 100,
        area_m2: Optional[float] = None,
    ):
        self.zone_id = zone_id
        self.zone_polygon = np.array(zone_polygon) if zone_polygon else None
        self._path = None
        if self.zone_polygon is not None and len(self.zone_polygon) >= 3:
            try:
                from matplotlib.path import Path
                self._path = Path(self.zone_polygon)
            except Exception:
                self._path = None
        self.capacity = capacity
        self.area_m2 = area_m2
        self.current_count = 0
        self.entries = 0
        self.exits = 0
        self.unique_tracks: set = set()
        self.track_states: dict[int, TrackState] = {}
        self.validated_count: int = 0

    def is_inside(self, point: Union[List[float], Tuple[float, float], np.ndarray]) -> bool:
        if self.zone_polygon is None:
            return True
        if self._path is None:
            try:
                from matplotlib.path import Path
                self._path = Path(self.zone_polygon)
            except Exception:
                return True
        pt = (float(point[0]), float(point[1]))
        return bool(self._path.contains_point(pt))

    def update(self, tracks: List, frame_number: int, frame_rate: float = 30.0, only_validated: bool = False):
        current_inside = set()
        validated_inside = set()
        
        for track in tracks:
            state_val = getattr(track, "state", "active")
            if state_val not in ("active", "validated"):
                continue
            is_val = getattr(track, "is_validated", False)
            if only_validated and not is_val:
                continue

            # Ground footprint contact point preferred, fallback to center_xy / centroid
            if hasattr(track, "foot_xy") and track.foot_xy is not None:
                pos = track.foot_xy
            elif hasattr(track, "center_xy") and track.center_xy is not None:
                pos = track.center_xy
            elif hasattr(track, "centroid") and track.centroid is not None:
                pos = track.centroid
            elif hasattr(track, "bbox") and track.bbox is not None:
                b = track.bbox
                pos = [(b[0] + b[2]) / 2.0, b[3]]
            else:
                continue
            
            track_id = getattr(track, "track_id", None)
            if track_id is None:
                continue

            is_inside = self.is_inside(pos)
            
            if is_inside:
                current_inside.add(track_id)
                self.unique_tracks.add(track_id)
                if is_val:
                    validated_inside.add(track_id)
            
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(track_id)
                self.track_states[track_id].first_seen_frame = frame_number
            
            state = self.track_states[track_id]
            was_inside = state.zone_id == self.zone_id
            
            if is_inside and not was_inside:
                state.zone_id = self.zone_id
                if not state.is_counted_entry:
                    self.entries += 1
                    state.is_counted_entry = True
            elif not is_inside and was_inside:
                state.zone_id = None
                if not state.is_counted_exit:
                    self.exits += 1
                    state.is_counted_exit = True
            
            centroid_val = getattr(track, "centroid", pos)
            if isinstance(centroid_val, tuple):
                centroid_val = list(centroid_val)
            state.add_position(centroid_val, frame_number)
        
        self.current_count = len(current_inside)
        self.validated_count = len(validated_inside)
        
        for track_id in list(self.track_states.keys()):
            st = self.track_states[track_id]
            if frame_number - st.last_seen_frame > 300:
                del self.track_states[track_id]

    def get_counting_result(self, timestamp: float) -> CountingResult:
        net = self.entries - self.exits
        area = self.area_m2 or 100.0
        dens = round(self.current_count / area, 3) if area > 0 else 0.0
        return CountingResult(
            zone_id=self.zone_id,
            timestamp=timestamp,
            current_count=self.current_count,
            entries=self.entries,
            exits=self.exits,
            unique_tracks=len(self.unique_tracks),
            net_count=net,
            validated_count=self.validated_count,
            zone_counts={self.zone_id: self.current_count},
            zone_densities={self.zone_id: dens},
        )


class DensityAnalyzer:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.density_history: dict[str, List[tuple]] = {}

    def compute_density(
        self, 
        tracks: List, 
        context: AnalyticsContext,
        frame_rate: float = 30.0
    ) -> DensityResult:
        active_tracks = [t for t in tracks if getattr(t, "state", "active") == "active"]
        count = len(active_tracks)

        if context.zone_polygon is None and context.area_sqm is None:
            return self._compute_simple_density(tracks, context)
        
        # If explicit area_sqm is provided in context, use it for calibrated density
        calibrated_area = getattr(context, "area_sqm", None)
        if calibrated_area and calibrated_area > 0:
            real_density = count / calibrated_area
        else:
            real_density = None

        if context.zone_polygon is not None:
            grid = self._create_grid(context)
            density_grid = self._map_tracks_to_grid(tracks, grid, context)
            valid_cells = density_grid[density_grid > 0]
            avg_density = real_density if real_density is not None else (float(np.mean(valid_cells)) if len(valid_cells) > 0 else 0.0)
            max_density = float(np.max(density_grid)) if density_grid.size > 0 else avg_density
            g_data = density_grid.tolist()
            gw = grid["width"]
            gh = grid["height"]
        else:
            avg_density = real_density if real_density is not None else (count / 100.0)
            max_density = avg_density
            g_data = [[avg_density]]
            gw = 1
            gh = 1

        pred = self._compute_predictive_metrics(context.zone_id, context.timestamp, avg_density, count)
        risk_score = min(100.0, float(max_density ** 1.5 * 8.0))

        return DensityResult(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            grid=DensityGrid(
                grid_data=g_data,
                grid_width=gw,
                grid_height=gh,
                cell_size_meters=self.config.density_grid_size,
                max_density=max_density,
            ),
            avg_density=avg_density,
            max_density=max_density,
            density_rate_of_change=pred["density_rate_of_change"],
            projected_density_5min=pred["projected_density_5min"],
            projected_occupancy_5min=pred["projected_occupancy_5min"],
            prediction_horizon_min=pred["prediction_horizon_min"],
            lead_time_to_critical_min=pred["lead_time_to_critical_min"],
            trend_label=pred["trend_label"],
            calibrated_area_sqm=calibrated_area,
            risk_score=round(risk_score, 2),
            normalized_score=round(min(1.0, risk_score / 100.0), 4),
        )

    def _compute_simple_density(self, tracks: List, context: AnalyticsContext) -> DensityResult:
        active_tracks = [t for t in tracks if getattr(t, "state", "active") == "active"]
        area_sqm = getattr(context, "area_sqm", None) or 100.0
        count = len(active_tracks)
        density = count / area_sqm if area_sqm > 0 else 0.0
        
        pred = self._compute_predictive_metrics(context.zone_id, context.timestamp, density, count)
        risk_score = min(100.0, float(density ** 1.5 * 8.0))

        return DensityResult(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            grid=DensityGrid(
                grid_data=[[density]],
                grid_width=1,
                grid_height=1,
                cell_size_meters=self.config.density_grid_size,
                max_density=density,
            ),
            avg_density=density,
            max_density=density,
            density_rate_of_change=pred["density_rate_of_change"],
            projected_density_5min=pred["projected_density_5min"],
            projected_occupancy_5min=pred["projected_occupancy_5min"],
            prediction_horizon_min=pred["prediction_horizon_min"],
            lead_time_to_critical_min=pred["lead_time_to_critical_min"],
            trend_label=pred["trend_label"],
            calibrated_area_sqm=area_sqm,
            risk_score=round(risk_score, 2),
            normalized_score=round(min(1.0, risk_score / 100.0), 4),
        )

    def _compute_predictive_metrics(self, zone_id: str, timestamp: float, avg_density: float, count: int) -> dict:
        if zone_id not in self.density_history:
            self.density_history[zone_id] = []
        
        history = self.density_history[zone_id]
        history.append((timestamp, avg_density, count))
        if len(history) > 60:
            history.pop(0)
        
        if len(history) < 2:
            rate_per_min = 0.0
            count_rate_ppm = 0.0
        else:
            oldest_time, oldest_density, oldest_count = history[0]
            dt = timestamp - oldest_time
            if dt > 0.01:
                rate_per_min = ((avg_density - oldest_density) / dt) * 60.0
                count_rate_ppm = ((count - oldest_count) / dt) * 60.0
            else:
                rate_per_min = 0.0
                count_rate_ppm = 0.0
        
        horizon = getattr(self.config, "prediction_horizon_minutes", 5.0)
        proj_density = max(0.0, round(avg_density + (rate_per_min * horizon), 3))
        proj_occupancy = max(0, int(round(count + (count_rate_ppm * horizon))))
        
        if rate_per_min > 0.05:
            trend_label = "rising"
        elif rate_per_min < -0.05:
            trend_label = "falling"
        else:
            trend_label = "stable"
        
        # Lead time calculation toward warning threshold (3.0 p/m²)
        warn_threshold = 3.0
        lead_time = None
        if avg_density >= warn_threshold:
            lead_time = 0.0
        elif rate_per_min > 0:
            needed = warn_threshold - avg_density
            minutes = needed / rate_per_min
            if 0 < minutes <= horizon:
                lead_time = round(minutes, 1)
        
        return {
            "density_rate_of_change": round(rate_per_min, 4),
            "projected_density_5min": proj_density,
            "projected_occupancy_5min": proj_occupancy,
            "prediction_horizon_min": horizon,
            "lead_time_to_critical_min": lead_time,
            "trend_label": trend_label,
        }

    def _create_grid(self, context: AnalyticsContext) -> dict:
        polygon = np.array(context.zone_polygon)
        min_x, min_y = np.min(polygon[:, 0]), np.min(polygon[:, 1])
        max_x, max_y = np.max(polygon[:, 0]), np.max(polygon[:, 1])
        
        width_m = max_x - min_x
        height_m = max_y - min_y
        
        grid_width = max(1, int(width_m / self.config.density_grid_size))
        grid_height = max(1, int(height_m / self.config.density_grid_size))
        
        return {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "width": grid_width,
            "height": grid_height,
            "cell_size": self.config.density_grid_size,
        }

    def _map_tracks_to_grid(self, tracks: List, grid: dict, context: AnalyticsContext) -> np.ndarray:
        density_grid = np.zeros((grid["height"], grid["width"]))
        
        if context.homography_matrix is not None:
            H = np.array(context.homography_matrix)
        else:
            H = None
        
        for track in tracks:
            if track.state != "active":
                continue
            
            centroid = np.array(track.centroid)
            if H is not None:
                centroid_h = np.array([centroid[0], centroid[1], 1.0])
                mapped = H @ centroid_h
                centroid = np.array([mapped[0] / mapped[2], mapped[1] / mapped[2]])
            
            col = int((centroid[0] - grid["min_x"]) / grid["cell_size"])
            row = int((centroid[1] - grid["min_y"]) / grid["cell_size"])
            
            if 0 <= row < grid["height"] and 0 <= col < grid["width"]:
                density_grid[row, col] += 1
        
        return density_grid


class OccupancyAnalyzer:
    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def analyze(
        self,
        count: int,
        capacity: int,
        timestamp: float,
        zone_id: str,
        projected_count_5min: Optional[int] = None,
    ) -> OccupancyResult:
        if capacity <= 0:
            occupancy_ratio = 0.0
            headroom = 0
            headroom_ratio = 0.0
        else:
            occupancy_ratio = count / capacity
            headroom = max(0, capacity - count)
            headroom_ratio = max(0.0, float(headroom) / float(capacity))
        
        if occupancy_ratio >= self.config.occupancy_critical_threshold:
            status = "critical"
        elif occupancy_ratio >= self.config.occupancy_warning_threshold:
            status = "warning"
        else:
            status = "safe"
        
        risk_score = min(100.0, occupancy_ratio * 100.0)
        normalized_score = min(1.0, occupancy_ratio)

        return OccupancyResult(
            zone_id=zone_id,
            timestamp=timestamp,
            capacity=capacity,
            current_count=count,
            occupancy_ratio=occupancy_ratio,
            status=status,
            headroom=headroom,
            headroom_ratio=round(headroom_ratio, 4),
            projected_occupancy_5min=projected_count_5min,
            risk_score=round(risk_score, 2),
            normalized_score=round(normalized_score, 4),
        )


class MovementAnalyzer:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.track_speeds: dict[int, List[float]] = {}
        self.track_directions: dict[int, List[float]] = {}
        self.zone_speed_history: dict[str, List[float]] = {}

    def analyze(self, tracks: List, context: AnalyticsContext, frame_rate: float = 30.0) -> MovementResult:
        movements = []
        
        for track in tracks:
            if getattr(track, "state", "active") != "active":
                continue
            
            centroid_hist = getattr(track, "centroid_history", [])
            if len(centroid_hist) >= 2:
                if hasattr(track, "get_velocity"):
                    velocity = track.get_velocity(frame_rate)
                    speed = track.get_speed(frame_rate)
                    direction = track.get_direction(frame_rate)
                else:
                    prev = np.array(centroid_hist[-2])
                    curr = np.array(centroid_hist[-1])
                    vel = (curr - prev) * frame_rate
                    velocity = vel.tolist()
                    speed = float(np.linalg.norm(vel))
                    direction = float(np.arctan2(vel[1], vel[0])) if (vel[0] != 0 or vel[1] != 0) else 0.0
            elif hasattr(track, "speed") or hasattr(track, "velocity"):
                vel = getattr(track, "velocity", (0.0, 0.0))
                velocity = list(vel) if isinstance(vel, (list, tuple)) else [0.0, 0.0]
                speed = float(getattr(track, "speed", np.linalg.norm(velocity)))
                direction = float(getattr(track, "heading", 0.0))
                if direction == 0.0 and (velocity[0] != 0 or velocity[1] != 0):
                    direction = float(np.arctan2(velocity[1], velocity[0]))
            else:
                continue
            
            if track.track_id not in self.track_speeds:
                self.track_speeds[track.track_id] = []
            if track.track_id not in self.track_directions:
                self.track_directions[track.track_id] = []
            
            self.track_speeds[track.track_id].append(speed)
            self.track_directions[track.track_id].append(direction)
            
            if len(self.track_speeds[track.track_id]) > 10:
                self.track_speeds[track.track_id].pop(0)
            if len(self.track_directions[track.track_id]) > 10:
                self.track_directions[track.track_id].pop(0)
            
            smoothed_speed = np.mean(self.track_speeds[track.track_id])
            smoothed_direction = np.mean(self.track_directions[track.track_id])
            
            movements.append(MovementVector(
                track_id=track.track_id,
                centroid=track.centroid,
                velocity=velocity,
                speed=smoothed_speed,
                direction=smoothed_direction,
            ))
        
        avg_speed = float(np.mean([m.speed for m in movements])) if movements else 0.0
        
        directions = [m.direction for m in movements if m.speed > 0.1]
        avg_direction = float(np.mean(directions)) if directions else None
        
        dominant_direction = self._compute_dominant_direction(movements) if movements else None

        # Counter-flow and crossing-flow conflict metrics
        counter_flow_index = 0.0
        crossing_flow_detected = False
        if dominant_direction is not None and movements:
            moving_vectors = [m for m in movements if m.speed > 0.1]
            counter_weights = 0.0
            crossing_weights = 0.0
            total_weights = sum(m.speed for m in moving_vectors)
            counter_thresh = np.radians(getattr(self.config, "counter_flow_angle_threshold_deg", 120.0))
            crossing_thresh = np.radians(getattr(self.config, "crossing_flow_angle_threshold_deg", 60.0))

            for m in moving_vectors:
                diff = abs(float(np.arctan2(np.sin(m.direction - dominant_direction), np.cos(m.direction - dominant_direction))))
                if diff >= counter_thresh:
                    counter_weights += m.speed
                elif diff >= crossing_thresh:
                    crossing_weights += m.speed

            if total_weights > 0:
                counter_flow_index = float(round(counter_weights / total_weights, 3))
                crossing_flow_detected = (crossing_weights / total_weights) > 0.15 or counter_flow_index > 0.10

        # Directional convergence (funneling toward center of mass or choke point)
        directional_convergence = 0.0
        if len(movements) >= 2:
            coords = np.array([m.centroid for m in movements])
            center = np.mean(coords, axis=0)
            inward_alignments = []
            for m in movements:
                p = np.array(m.centroid)
                d = center - p
                dist = float(np.linalg.norm(d))
                v = np.array(m.velocity)
                v_norm = float(np.linalg.norm(v))
                if dist > 1.0 and v_norm > 0.01:
                    u = d / dist
                    cos_phi = float(np.dot(v / v_norm, u))
                    inward_alignments.append(max(0.0, cos_phi))
            directional_convergence = float(round(np.mean(inward_alignments), 3)) if inward_alignments else 0.0

        # Velocity deceleration / drop ratio and speed trend
        zone_id = context.zone_id
        if zone_id not in self.zone_speed_history:
            self.zone_speed_history[zone_id] = []
        sp_hist = self.zone_speed_history[zone_id]
        sp_hist.append(avg_speed)
        if len(sp_hist) > 30:
            sp_hist.pop(0)

        if len(sp_hist) >= 5:
            baseline = float(np.mean(sp_hist[:max(3, len(sp_hist) // 2)]))
        else:
            baseline = avg_speed

        if baseline > 0.1:
            velocity_drop_ratio = float(round(max(0.0, min(1.0, 1.0 - (avg_speed / baseline))), 3))
        else:
            velocity_drop_ratio = 0.0

        if velocity_drop_ratio >= getattr(self.config, "velocity_drop_threshold", 0.30):
            speed_trend = "slowing"
        elif baseline > 0.1 and avg_speed > baseline * 1.25:
            speed_trend = "increasing"
        else:
            speed_trend = "stable"

        # Stagnation / low-movement ratio
        if getattr(context, "scale_meters_per_pixel", None):
            stagnation_thresh = getattr(self.config, "stagnation_speed_threshold_m_s", 0.20)
        else:
            stagnation_thresh = getattr(self.config, "stagnation_speed_threshold_px_s", 5.0)
        stagnant_count = sum(1 for m in movements if m.speed < stagnation_thresh)
        stagnation_ratio = float(round(stagnant_count / len(movements), 3)) if movements else 0.0

        if avg_speed > 0:
            base_score = min(100.0, 100.0 / (avg_speed + 0.5))
        else:
            base_score = 100.0 if movements else 0.0
        
        mult = 1.0 + (0.35 * stagnation_ratio) + (0.35 * counter_flow_index) + (0.30 * directional_convergence)
        movement_risk_score = min(100.0, round(base_score * mult, 2))
        normalized_movement = min(1.0, round(movement_risk_score / 100.0, 4))

        return MovementResult(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            movements=movements,
            avg_speed=avg_speed,
            avg_direction=avg_direction,
            dominant_direction=dominant_direction,
            counter_flow_index=counter_flow_index,
            directional_convergence=directional_convergence,
            velocity_drop_ratio=velocity_drop_ratio,
            speed_trend=speed_trend,
            crossing_flow_detected=crossing_flow_detected,
            stagnation_ratio=stagnation_ratio,
            low_movement_count=stagnant_count,
            risk_score=movement_risk_score,
            normalized_score=normalized_movement,
        )

    def _compute_dominant_direction(self, movements: List[MovementVector]) -> Optional[float]:
        if not movements:
            return None
        
        hist, bin_edges = np.histogram(
            [m.direction for m in movements],
            bins=self.config.direction_bins,
            range=(-np.pi, np.pi),
            weights=[m.speed for m in movements],
        )
        
        if np.sum(hist) == 0:
            return None
        
        dominant_bin = np.argmax(hist)
        return float((bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2)


class FlowAnalyzer:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.track_positions: dict[int, List[tuple]] = {}
        self.zone_count_history: dict[str, List[tuple]] = {}
        self.window_size = config.flow_window_seconds

    def analyze(
        self, 
        tracks: List, 
        context: AnalyticsContext, 
        frame_rate: float = 30.0,
        entry_lines: Optional[List[dict]] = None,
        exit_lines: Optional[List[dict]] = None
    ) -> FlowResult:
        current_time = context.timestamp
        
        for track in tracks:
            if getattr(track, "state", "active") != "active":
                continue
            
            if track.track_id not in self.track_positions:
                # Initialize from track's centroid_history if available
                self.track_positions[track.track_id] = []
                if hasattr(track, 'centroid_history') and track.centroid_history:
                    # Estimate timestamps for historical positions
                    dt = 1.0 / frame_rate
                    for i, centroid in enumerate(track.centroid_history):
                        hist_time = current_time - (len(track.centroid_history) - 1 - i) * dt
                        self.track_positions[track.track_id].append({
                            "time": hist_time,
                            "centroid": centroid,
                        })
            
            # Add current position
            self.track_positions[track.track_id].append({
                "time": current_time,
                "centroid": track.centroid,
            })
        
        entry_flow = 0.0
        exit_flow = 0.0
        
        if entry_lines:
            entry_flow = self._compute_line_flow(entry_lines, "entry", frame_rate)
        if exit_lines:
            exit_flow = self._compute_line_flow(exit_lines, "exit", frame_rate)
        
        net_flow = entry_flow - exit_flow
        
        directions = []
        speeds = []
        for track in tracks:
            if getattr(track, "state", "active") != "active" or len(getattr(track, "centroid_history", [])) < 2:
                continue
            speed = track.get_speed(frame_rate)
            if speed > 0.1:
                speeds.append(speed)
                directions.append(track.get_direction(frame_rate))
        
        flow_rate = (entry_flow + exit_flow) / 2.0 if (entry_flow + exit_flow) > 0 else 0.0
        flow_direction = float(np.mean(directions)) if directions else None

        # Track rolling count history for net crowd accumulation (persons/min)
        zone_id = context.zone_id
        if zone_id not in self.zone_count_history:
            self.zone_count_history[zone_id] = []
        
        active_count = len([t for t in tracks if getattr(t, "state", "active") == "active"])
        self.zone_count_history[zone_id].append((current_time, active_count))
        if len(self.zone_count_history[zone_id]) > 60:
            self.zone_count_history[zone_id].pop(0)

        if (entry_flow + exit_flow) > 0:
            net_accumulation_rate_ppm = float(round(net_flow, 2))
            inflow_outflow_ratio = float(round(entry_flow / max(0.1, exit_flow), 2))
        else:
            cnt_hist = self.zone_count_history[zone_id]
            if len(cnt_hist) >= 2:
                t0, c0 = cnt_hist[0]
                dt = current_time - t0
                rate = ((active_count - c0) / dt) * 60.0 if dt > 0.05 else 0.0
            else:
                rate = 0.0
            net_accumulation_rate_ppm = float(round(rate, 2))
            inflow_outflow_ratio = float(round(1.0 + (rate / max(1.0, active_count)), 2))

        # Directional opposition and crossing-flow scores across moving tracks
        dir_opp_score = 0.0
        crossing_score = 0.0
        if len(directions) >= 2:
            dir_rads = np.array(directions)
            diff_matrix = np.abs(np.arctan2(
                np.sin(dir_rads[:, None] - dir_rads[None, :]),
                np.cos(dir_rads[:, None] - dir_rads[None, :])
            ))
            counter_thresh = np.radians(getattr(self.config, "counter_flow_angle_threshold_deg", 120.0))
            crossing_thresh = np.radians(getattr(self.config, "crossing_flow_angle_threshold_deg", 60.0))

            n_pairs = len(directions) * (len(directions) - 1)
            if n_pairs > 0:
                opp_pairs = int(np.sum(diff_matrix >= counter_thresh))
                cross_pairs = int(np.sum((diff_matrix >= crossing_thresh) & (diff_matrix < counter_thresh)))
                dir_opp_score = float(round(opp_pairs / n_pairs, 3))
                crossing_score = float(round(cross_pairs / n_pairs, 3))

        flow_conflict_detected = (
            net_accumulation_rate_ppm > 15.0
            or (inflow_outflow_ratio > 2.0 and flow_rate > 5.0)
            or dir_opp_score > 0.15
        )
        accum_cf = max(0.0, net_accumulation_rate_ppm / 25.0) * (1.0 if flow_conflict_detected else 0.5)
        counter_flow_conflict_score = min(1.0, round(max(dir_opp_score, accum_cf), 3))
        crossing_flow_conflict_score = min(1.0, round(max(crossing_score, flow_rate / 15.0), 3)) if (crossing_score > 0 or flow_rate > 0) else 0.0
        flow_risk_score = min(100.0, round(flow_rate * 5.0 + max(0.0, net_accumulation_rate_ppm) * 1.5 + dir_opp_score * 25.0, 2))
        normalized_flow = min(1.0, round(flow_risk_score / 100.0, 4))

        return FlowResult(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            flow_rate=flow_rate,
            flow_direction=flow_direction,
            entry_flow=entry_flow,
            exit_flow=exit_flow,
            net_flow=net_flow,
            net_accumulation_rate_ppm=net_accumulation_rate_ppm,
            inflow_outflow_ratio=inflow_outflow_ratio,
            flow_conflict_detected=flow_conflict_detected,
            counter_flow_conflict_score=counter_flow_conflict_score,
            crossing_flow_conflict_score=crossing_flow_conflict_score,
            risk_score=flow_risk_score,
            normalized_score=normalized_flow,
        )

    def _compute_line_flow(self, lines: List[dict], line_type: str, frame_rate: float) -> float:
        count = 0
        for track_id, positions in self.track_positions.items():
            if len(positions) < 2:
                continue
            
            for line in lines:
                p1 = np.array(line["start"])
                p2 = np.array(line["end"])
                
                for i in range(1, len(positions)):
                    prev_pos = np.array(positions[i-1]["centroid"])
                    curr_pos = np.array(positions[i]["centroid"])
                    
                    if self._line_crossed(prev_pos, curr_pos, p1, p2):
                        count += 1
                        break
        
        return count / max(self.window_size / 60.0, 1.0)

    def _line_crossed(self, p1: np.ndarray, p2: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> bool:
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        
        return ccw(p1, line_start, line_end) != ccw(p2, line_start, line_end) and \
               ccw(p1, p2, line_start) != ccw(p1, p2, line_end)


class BottleneckDetector:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.congestion_history: dict[str, List[float]] = {}

    def analyze(
        self, 
        density_result: DensityResult, 
        flow_result: FlowResult, 
        movement_result: MovementResult,
        context: AnalyticsContext
    ) -> BottleneckResult:
        zone_id = context.zone_id
        
        density_score = min(density_result.max_density / 5.0, 1.0)
        
        flow_score = 0.0
        if flow_result.flow_rate > 0:
            flow_score = min(flow_result.flow_rate / 10.0, 1.0)
        
        movement_score = 0.0
        if movement_result.avg_speed > 0:
            movement_score = min(1.0 / (movement_result.avg_speed + 0.1), 1.0)
        
        bottleneck_score = (density_score * 0.5 + flow_score * 0.3 + movement_score * 0.2)
        
        if zone_id not in self.congestion_history:
            self.congestion_history[zone_id] = []
        
        self.congestion_history[zone_id].append(bottleneck_score)
        
        if len(self.congestion_history[zone_id]) > self.config.bottleneck_persistence_frames:
            self.congestion_history[zone_id].pop(0)
        
        avg_score = np.mean(self.congestion_history[zone_id])
        is_bottleneck = avg_score > 0.5 and len(self.congestion_history[zone_id]) >= self.config.bottleneck_persistence_frames // 2
        
        if avg_score < 0.2:
            congestion_level = "none"
        elif avg_score < 0.4:
            congestion_level = "low"
        elif avg_score < 0.6:
            congestion_level = "medium"
        elif avg_score < 0.8:
            congestion_level = "high"
        else:
            congestion_level = "critical"
        
        affected_area = None
        if is_bottleneck and density_result.grid.grid_data:
            grid = np.array(density_result.grid.grid_data)
            max_idx = np.unravel_index(np.argmax(grid), grid.shape)
            affected_area = [float(max_idx[1]), float(max_idx[0])]

        # Advance 5-minute lead predictive bottleneck projection
        proj_dens = getattr(density_result, "projected_density_5min", 0.0)
        vel_drop = getattr(movement_result, "velocity_drop_ratio", 0.0)
        net_flow_acc = getattr(flow_result, "net_accumulation_rate_ppm", 0.0)
        lead_time = getattr(density_result, "lead_time_to_critical_min", None)

        projected_bottleneck_5min = bool(
            (proj_dens >= 2.8 and vel_drop > 0.20)
            or (net_flow_acc > 12.0 and vel_drop > 0.25)
            or (proj_dens >= 3.5)
        )
        early_warning_lead_min = lead_time if lead_time is not None else (4.5 if projected_bottleneck_5min else None)

        stagnation_ratio = getattr(movement_result, "stagnation_ratio", 0.0)
        stagnation_overlap = bool(
            density_result.max_density >= 2.5 and stagnation_ratio >= 0.40
        )
        if stagnation_overlap:
            bottleneck_score = min(1.0, bottleneck_score * 1.25)

        bottleneck_risk_score = min(100.0, round(bottleneck_score * 100.0, 2))
        normalized_bottleneck = min(1.0, round(bottleneck_score, 4))

        return BottleneckResult(
            zone_id=zone_id,
            timestamp=context.timestamp,
            bottleneck_score=float(bottleneck_score),
            is_bottleneck=is_bottleneck,
            congestion_level=congestion_level,
            affected_area=affected_area,
            projected_bottleneck_5min=projected_bottleneck_5min,
            early_warning_lead_min=early_warning_lead_min,
            stagnation_overlap=stagnation_overlap,
            risk_score=bottleneck_risk_score,
            normalized_score=normalized_bottleneck,
        )