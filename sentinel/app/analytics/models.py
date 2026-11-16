from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


class CountingResult(BaseModel):
    zone_id: str
    timestamp: float
    current_count: int
    entries: int
    exits: int
    unique_tracks: int


class DensityGrid(BaseModel):
    grid_data: List[List[float]]
    grid_width: int
    grid_height: int
    cell_size_meters: float
    max_density: float


class DensityResult(BaseModel):
    zone_id: str
    timestamp: float
    grid: DensityGrid
    avg_density: float
    max_density: float


class OccupancyResult(BaseModel):
    zone_id: str
    timestamp: float
    capacity: int
    current_count: int
    occupancy_ratio: float
    status: Literal["safe", "warning", "critical"]


class MovementVector(BaseModel):
    track_id: int
    centroid: List[float]
    velocity: List[float]
    speed: float
    direction: float


class MovementResult(BaseModel):
    zone_id: str
    timestamp: float
    movements: List[MovementVector]
    avg_speed: float
    avg_direction: Optional[float] = None
    dominant_direction: Optional[float] = None


class FlowResult(BaseModel):
    zone_id: str
    timestamp: float
    flow_rate: float
    flow_direction: Optional[float] = None
    entry_flow: float
    exit_flow: float
    net_flow: float


class BottleneckResult(BaseModel):
    zone_id: str
    timestamp: float
    bottleneck_score: float
    is_bottleneck: bool
    congestion_level: Literal["none", "low", "medium", "high", "critical"]
    affected_area: Optional[List[float]] = None


class AnalyticsContext(BaseModel):
    zone_id: str
    frame_number: int
    timestamp: float
    zone_polygon: Optional[List[List[float]]] = None
    zone_capacity: int = 0
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
    def __init__(self, zone_id: str, zone_polygon: Optional[List[List[float]]] = None):
        self.zone_id = zone_id
        self.zone_polygon = np.array(zone_polygon) if zone_polygon else None
        self.current_count = 0
        self.entries = 0
        self.exits = 0
        self.unique_tracks: set = set()
        self.track_states: dict[int, TrackState] = {}

    def is_inside(self, centroid: List[float]) -> bool:
        if self.zone_polygon is None:
            return True
        from matplotlib.path import Path
        path = Path(self.zone_polygon)
        return path.contains_point(centroid)

    def update(self, tracks: List, frame_number: int, frame_rate: float = 30.0):
        current_inside = set()
        
        for track in tracks:
            if track.state != "active":
                continue
            
            centroid = track.centroid
            track_id = track.track_id
            is_inside = self.is_inside(centroid)
            
            if is_inside:
                current_inside.add(track_id)
                self.unique_tracks.add(track_id)
            
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
            
            state.add_position(centroid, frame_number)
        
        self.current_count = len(current_inside)
        
        for track_id in list(self.track_states.keys()):
            state = self.track_states[track_id]
            if frame_number - state.last_seen_frame > 300:
                del self.track_states[track_id]

    def get_counting_result(self, timestamp: float) -> CountingResult:
        return CountingResult(
            zone_id=self.zone_id,
            timestamp=timestamp,
            current_count=self.current_count,
            entries=self.entries,
            exits=self.exits,
            unique_tracks=len(self.unique_tracks),
        )


class DensityAnalyzer:
    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def compute_density(
        self, 
        tracks: List, 
        context: AnalyticsContext,
        frame_rate: float = 30.0
    ) -> DensityResult:
        if context.zone_polygon is None:
            return self._compute_simple_density(tracks, context)
        
        grid = self._create_grid(context)
        density_grid = self._map_tracks_to_grid(tracks, grid, context)
        
        valid_cells = density_grid[density_grid > 0]
        avg_density = float(np.mean(valid_cells)) if len(valid_cells) > 0 else 0.0
        max_density = float(np.max(density_grid)) if density_grid.size > 0 else 0.0
        
        return DensityResult(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            grid=DensityGrid(
                grid_data=density_grid.tolist(),
                grid_width=grid["width"],
                grid_height=grid["height"],
                cell_size_meters=self.config.density_grid_size,
                max_density=max_density,
            ),
            avg_density=avg_density,
            max_density=max_density,
        )

    def _compute_simple_density(self, tracks: List, context: AnalyticsContext) -> DensityResult:
        active_tracks = [t for t in tracks if t.state == "active"]
        area_sqm = 100.0
        count = len(active_tracks)
        density = count / area_sqm if area_sqm > 0 else 0.0
        
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
        )

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

    def analyze(self, count: int, capacity: int, timestamp: float, zone_id: str) -> OccupancyResult:
        if capacity <= 0:
            occupancy_ratio = 0.0
        else:
            occupancy_ratio = count / capacity
        
        if occupancy_ratio >= self.config.occupancy_critical_threshold:
            status = "critical"
        elif occupancy_ratio >= self.config.occupancy_warning_threshold:
            status = "warning"
        else:
            status = "safe"
        
        return OccupancyResult(
            zone_id=zone_id,
            timestamp=timestamp,
            capacity=capacity,
            current_count=count,
            occupancy_ratio=occupancy_ratio,
            status=status,
        )


class MovementAnalyzer:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.track_speeds: dict[int, List[float]] = {}
        self.track_directions: dict[int, List[float]] = {}

    def analyze(self, tracks: List, context: AnalyticsContext, frame_rate: float = 30.0) -> MovementResult:
        movements = []
        
        for track in tracks:
            if track.state != "active":
                continue
            
            if len(track.centroid_history) < 2:
                continue
            
            velocity = track.get_velocity(frame_rate)
            speed = track.get_speed(frame_rate)
            direction = track.get_direction(frame_rate)
            
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
        
        return MovementResult(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            movements=movements,
            avg_speed=avg_speed,
            avg_direction=avg_direction,
            dominant_direction=dominant_direction,
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
            if track.state != "active":
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
            if track.state != "active" or len(track.centroid_history) < 2:
                continue
            speed = track.get_speed(frame_rate)
            if speed > 0.1:
                speeds.append(speed)
                directions.append(track.get_direction(frame_rate))
        
        flow_rate = (entry_flow + exit_flow) / 2.0 if (entry_flow + exit_flow) > 0 else 0.0
        flow_direction = float(np.mean(directions)) if directions else None
        
        return FlowResult(
            zone_id=context.zone_id,
            timestamp=context.timestamp,
            flow_rate=flow_rate,
            flow_direction=flow_direction,
            entry_flow=entry_flow,
            exit_flow=exit_flow,
            net_flow=net_flow,
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
        
        return BottleneckResult(
            zone_id=zone_id,
            timestamp=context.timestamp,
            bottleneck_score=float(bottleneck_score),
            is_bottleneck=is_bottleneck,
            congestion_level=congestion_level,
            affected_area=affected_area,
        )