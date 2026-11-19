"""
People Count & Tracking Module for SENTINEL.
Provides real-time multi-zone crowd counting, virtual gate line-crossing detection,
anti-flicker debouncing, and synchronization with Appearance Re-ID and Second-Stage Validation.
"""

from typing import List, Optional, Dict, Any, Tuple, Union
import numpy as np
from loguru import logger

from app.analytics.models import (
    CountingResult,
    SceneCountReport,
    ZoneCounter,
    AnalyticsContext,
)
from app.tracking.models import Track


class LineCrossCounter:
    """
    Virtual gate or boundary line-crossing counter with anti-flicker debouncing.
    Detects trajectory segment intersection and calculates directional flow (inflow vs outflow).
    """

    def __init__(
        self,
        line_id: str,
        line_coords: Union[List[List[float]], List[Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]]],
        name: Optional[str] = None,
        cooldown_frames: int = 25,
        flow_window_seconds: float = 60.0,
    ):
        self.line_id = line_id
        self.name = name or line_id
        coords = [tuple(p) for p in line_coords]
        if len(coords) != 2:
            raise ValueError(f"Line coords must contain exactly 2 endpoints, got {len(coords)}")
        self.p1: Tuple[float, float] = (float(coords[0][0]), float(coords[0][1]))
        self.p2: Tuple[float, float] = (float(coords[1][0]), float(coords[1][1]))
        self.cooldown_frames = max(1, int(cooldown_frames))
        self.flow_window_seconds = max(1.0, float(flow_window_seconds))

        self.total_inflow: int = 0
        self.total_outflow: int = 0

        # Debounce tracking: track_id -> last_frame_crossed
        self._last_crossed: Dict[int, int] = {}
        # Position tracking: track_id -> (last_x, last_y, last_frame)
        self._prev_positions: Dict[int, Tuple[float, float, int]] = {}
        # Timestamps of crossing events for rolling rate calculation: [(timestamp, direction), ...]
        self._event_history: List[Tuple[float, str]] = []

    @staticmethod
    def _ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
        """Counter-clockwise test for 3 2D points."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @staticmethod
    def _segments_intersect(
        A: Tuple[float, float],
        B: Tuple[float, float],
        C: Tuple[float, float],
        D: Tuple[float, float],
    ) -> bool:
        """Check if 2D line segment AB intersects line segment CD."""
        ccw1 = LineCrossCounter._ccw(A, C, D) != LineCrossCounter._ccw(B, C, D)
        ccw2 = LineCrossCounter._ccw(A, B, C) != LineCrossCounter._ccw(A, B, D)
        return ccw1 and ccw2

    def _cross_product_orientation(self, P: Tuple[float, float]) -> float:
        """Signed 2D cross product of vector p1->p2 with vector p1->P."""
        return (self.p2[0] - self.p1[0]) * (P[1] - self.p1[1]) - (self.p2[1] - self.p1[1]) * (P[0] - self.p1[0])

    def update(
        self,
        tracks: List[Any],
        frame_number: int,
        timestamp: float,
        only_validated: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate tracks for line crossings on this frame.
        Applies anti-flicker cooldown to debounce boundary oscillations.
        """
        inflow_events: List[int] = []
        outflow_events: List[int] = []

        # Prune event history older than rolling flow window
        cutoff_time = timestamp - self.flow_window_seconds
        self._event_history = [(ts, d) for ts, d in self._event_history if ts >= cutoff_time]

        for track in tracks:
            state_val = getattr(track, "state", "active")
            if state_val not in ("active", "validated"):
                continue
            if only_validated and not getattr(track, "is_validated", False):
                continue

            track_id = getattr(track, "track_id", None)
            if track_id is None:
                continue

            # Ground contact footprint point preferred for tilted camera accuracy
            if hasattr(track, "foot_xy") and track.foot_xy is not None:
                curr_pos = (float(track.foot_xy[0]), float(track.foot_xy[1]))
            elif hasattr(track, "center_xy") and track.center_xy is not None:
                curr_pos = (float(track.center_xy[0]), float(track.center_xy[1]))
            elif hasattr(track, "centroid") and track.centroid is not None:
                curr_pos = (float(track.centroid[0]), float(track.centroid[1]))
            elif hasattr(track, "bbox") and track.bbox is not None:
                b = track.bbox
                curr_pos = ((float(b[0]) + float(b[2])) / 2.0, float(b[3]))
            else:
                continue

            if track_id in self._prev_positions:
                prev_x, prev_y, prev_frame = self._prev_positions[track_id]
                prev_pos = (prev_x, prev_y)

                last_cross_f = self._last_crossed.get(track_id, -9999)
                # Require frame gap <= 30 and outside debounce cooldown
                if (frame_number - prev_frame <= 30) and (frame_number - last_cross_f >= self.cooldown_frames):
                    if self._segments_intersect(self.p1, self.p2, prev_pos, curr_pos):
                        prev_cross = self._cross_product_orientation(prev_pos)
                        curr_cross = self._cross_product_orientation(curr_pos)

                        if prev_cross < 0 and curr_cross >= 0:
                            # Inflow (entry)
                            self.total_inflow += 1
                            self._last_crossed[track_id] = frame_number
                            self._event_history.append((timestamp, "inflow"))
                            inflow_events.append(track_id)
                        elif prev_cross > 0 and curr_cross <= 0:
                            # Outflow (exit)
                            self.total_outflow += 1
                            self._last_crossed[track_id] = frame_number
                            self._event_history.append((timestamp, "outflow"))
                            outflow_events.append(track_id)

            self._prev_positions[track_id] = (curr_pos[0], curr_pos[1], frame_number)

        # Cleanup tracks inactive for > 300 frames (~10s at 30fps)
        for tid in list(self._prev_positions.keys()):
            if frame_number - self._prev_positions[tid][2] > 300:
                del self._prev_positions[tid]
                if tid in self._last_crossed:
                    del self._last_crossed[tid]

        return {
            "line_id": self.line_id,
            "inflow_events": inflow_events,
            "outflow_events": outflow_events,
            "total_inflow": self.total_inflow,
            "total_outflow": self.total_outflow,
            "net_flow": self.total_inflow - self.total_outflow,
            "inflow_rate_ppm": self.get_inflow_rate_ppm(),
            "outflow_rate_ppm": self.get_outflow_rate_ppm(),
            "net_flow_rate_ppm": self.get_net_flow_rate_ppm(),
        }

    def get_inflow_rate_ppm(self) -> float:
        """Inflow rate in persons per minute over the rolling time window."""
        inflow_count = sum(1 for _, d in self._event_history if d == "inflow")
        window_min = max(0.1, self.flow_window_seconds / 60.0)
        return round(float(inflow_count / window_min), 2)

    def get_outflow_rate_ppm(self) -> float:
        """Outflow rate in persons per minute over the rolling time window."""
        outflow_count = sum(1 for _, d in self._event_history if d == "outflow")
        window_min = max(0.1, self.flow_window_seconds / 60.0)
        return round(float(outflow_count / window_min), 2)

    def get_net_flow_rate_ppm(self) -> float:
        """Net flow rate in persons per minute (inflow_rate - outflow_rate)."""
        return round(self.get_inflow_rate_ppm() - self.get_outflow_rate_ppm(), 2)

    def reset(self):
        """Reset counters and anti-flicker states."""
        self.total_inflow = 0
        self.total_outflow = 0
        self._last_crossed.clear()
        self._prev_positions.clear()
        self._event_history.clear()


class ScenePeopleCounter:
    """
    Multi-zone crowd counting and track aggregation engine.
    Synchronizes Appearance Re-ID, Second-Stage Validation, and calibrated zones
    into unified real-time scene metrics.
    """

    def __init__(
        self,
        zones: Optional[Dict[str, ZoneCounter]] = None,
        lines: Optional[Dict[str, LineCrossCounter]] = None,
        default_capacity: int = 100,
        flow_window_seconds: float = 60.0,
    ):
        self.zone_counters: Dict[str, ZoneCounter] = zones or {}
        self.line_counters: Dict[str, LineCrossCounter] = lines or {}
        self.default_capacity = default_capacity
        self.flow_window_seconds = flow_window_seconds

        self.active_tracks: set[int] = set()
        self.validated_tracks: set[int] = set()
        self.tentative_tracks: set[int] = set()
        self.total_unique_seen: set[int] = set()
        self.latest_report: Optional[SceneCountReport] = None

    @classmethod
    def from_zone_manager(cls, zone_manager: Any, flow_window_seconds: float = 60.0) -> "ScenePeopleCounter":
        """Instantiate counters directly from a ZoneManager or dictionary of zones."""
        zones = {}
        lines = {}
        zone_dict = getattr(zone_manager, "zones", zone_manager if isinstance(zone_manager, dict) else {})

        for zid, zdef in zone_dict.items():
            if isinstance(zdef, dict):
                poly = zdef.get("image_polygon")
                cap = zdef.get("max_capacity") or zdef.get("capacity", 100)
                area = zdef.get("area_sqm") or zdef.get("area_m2")
                name = zdef.get("name", zid)
                entry_line = zdef.get("entry_line")
                exit_line = zdef.get("exit_line")
            else:
                poly = getattr(zdef, "image_polygon", None)
                cap = getattr(zdef, "max_capacity", getattr(zdef, "capacity", 100))
                area = getattr(zdef, "area_sqm", getattr(zdef, "area_m2", None))
                name = getattr(zdef, "name", zid)
                entry_line = getattr(zdef, "entry_line", None)
                exit_line = getattr(zdef, "exit_line", None)

            zones[zid] = ZoneCounter(zone_id=zid, zone_polygon=poly, capacity=cap, area_m2=area)

            if entry_line and len(entry_line) == 2:
                lines[f"{zid}_entry"] = LineCrossCounter(f"{zid}_entry", entry_line, name=f"{name} Entry")
            if exit_line and len(exit_line) == 2:
                lines[f"{zid}_exit"] = LineCrossCounter(f"{zid}_exit", exit_line, name=f"{name} Exit")

        return cls(zones=zones, lines=lines, flow_window_seconds=flow_window_seconds)

    def register_zone(
        self,
        zone_id: str,
        polygon: Optional[List[List[float]]] = None,
        capacity: int = 100,
        area_m2: Optional[float] = None,
    ) -> ZoneCounter:
        """Register a new calibrated polygon zone."""
        zc = ZoneCounter(zone_id=zone_id, zone_polygon=polygon, capacity=capacity, area_m2=area_m2)
        self.zone_counters[zone_id] = zc
        return zc

    def register_line(
        self,
        line_id: str,
        line_coords: Union[List[List[float]], List[Tuple[float, float]], Tuple[Tuple[float, float], Tuple[float, float]]],
        name: Optional[str] = None,
        cooldown_frames: int = 25,
    ) -> LineCrossCounter:
        """Register a new virtual entry/exit line counter."""
        lc = LineCrossCounter(
            line_id=line_id,
            line_coords=line_coords,
            name=name,
            cooldown_frames=cooldown_frames,
            flow_window_seconds=self.flow_window_seconds,
        )
        self.line_counters[line_id] = lc
        return lc

    def update(
        self,
        tracks: Union[List[Any], Any],
        frame_number: int = 0,
        timestamp: float = 0.0,
        frame_rate: float = 30.0,
        rejected_count: int = 0,
        reid_matches: int = 0,
        only_validated_for_zones: bool = False,
    ) -> SceneCountReport:
        """
        Process tracks for current frame, updating all zone and line counters.
        Generates a comprehensive SceneCountReport.
        """
        track_list: List[Any] = []
        if hasattr(tracks, "tracks"):
            track_list = tracks.tracks
            frame_number = getattr(tracks, "frame_number", frame_number)
            timestamp = getattr(tracks, "timestamp", timestamp)
            rejected_count = getattr(tracks, "rejected_count", rejected_count)
            reid_matches = getattr(tracks, "reid_matches", reid_matches)
        elif isinstance(tracks, list):
            track_list = tracks

        active_tids = []
        validated_tids = []
        tentative_tids = []

        for t in track_list:
            state_val = getattr(t, "state", "active")
            if state_val not in ("active", "validated"):
                continue
            tid = getattr(t, "track_id", None)
            if tid is None:
                continue
            active_tids.append(tid)
            self.total_unique_seen.add(tid)
            if getattr(t, "is_validated", False):
                validated_tids.append(tid)
            else:
                tentative_tids.append(tid)

        self.active_tracks = set(active_tids)
        self.validated_tracks = set(validated_tids)
        self.tentative_tracks = set(tentative_tids)

        # Update Zone Counters
        zone_counts: Dict[str, int] = {}
        zone_densities: Dict[str, float] = {}
        zone_occupancies: Dict[str, float] = {}
        zone_status: Dict[str, str] = {}
        total_entries = 0
        total_exits = 0

        if not self.zone_counters:
            self.register_zone("scene_global", None, capacity=self.default_capacity)

        for zid, zcounter in self.zone_counters.items():
            zcounter.update(
                track_list,
                frame_number=frame_number,
                frame_rate=frame_rate,
                only_validated=only_validated_for_zones,
            )
            count = zcounter.current_count
            zone_counts[zid] = count
            total_entries += zcounter.entries
            total_exits += zcounter.exits

            area = getattr(zcounter, "area_m2", None) or 100.0
            density = round(count / area, 3) if area > 0 else 0.0
            zone_densities[zid] = density

            cap = getattr(zcounter, "capacity", 100) or 100
            occ = round(count / float(cap), 3) if cap > 0 else 0.0
            zone_occupancies[zid] = occ

            if occ >= 0.85 or density >= 3.0:
                zone_status[zid] = "critical"
            elif occ >= 0.65 or density >= 1.5:
                zone_status[zid] = "warning"
            else:
                zone_status[zid] = "safe"

        # Update Line Counters
        line_inflow = 0
        line_outflow = 0
        inflow_rates = []
        outflow_rates = []

        for lid, lcounter in self.line_counters.items():
            l_res = lcounter.update(
                track_list,
                frame_number=frame_number,
                timestamp=timestamp,
                only_validated=only_validated_for_zones,
            )
            line_inflow += l_res["total_inflow"]
            line_outflow += l_res["total_outflow"]
            inflow_rates.append(l_res["inflow_rate_ppm"])
            outflow_rates.append(l_res["outflow_rate_ppm"])

        if self.line_counters:
            inflow_rate_ppm = round(float(np.sum(inflow_rates)), 2)
            outflow_rate_ppm = round(float(np.sum(outflow_rates)), 2)
            net_flow_rate_ppm = round(inflow_rate_ppm - outflow_rate_ppm, 2)
            agg_entries = line_inflow
            agg_exits = line_outflow
        else:
            agg_entries = total_entries
            agg_exits = total_exits
            elapsed_min = max(0.1, timestamp / 60.0) if timestamp > 0 else 1.0
            inflow_rate_ppm = round(float(total_entries / elapsed_min), 2)
            outflow_rate_ppm = round(float(total_exits / elapsed_min), 2)
            net_flow_rate_ppm = round(inflow_rate_ppm - outflow_rate_ppm, 2)

        report = SceneCountReport(
            timestamp=timestamp,
            frame_number=frame_number,
            total_active_count=len(self.active_tracks),
            total_validated_count=len(self.validated_tracks),
            total_tentative_count=len(self.tentative_tracks),
            rejected_count=rejected_count,
            reid_matches=reid_matches,
            zone_counts=zone_counts,
            zone_densities=zone_densities,
            zone_occupancies=zone_occupancies,
            zone_status=zone_status,
            total_entries=agg_entries,
            total_exits=agg_exits,
            net_flow=agg_entries - agg_exits,
            inflow_rate_ppm=inflow_rate_ppm,
            outflow_rate_ppm=outflow_rate_ppm,
            net_flow_rate_ppm=net_flow_rate_ppm,
            active_track_ids=active_tids,
            validated_track_ids=validated_tids,
        )
        self.latest_report = report
        return report

    def get_latest_report(self) -> Optional[SceneCountReport]:
        return self.latest_report

    def get_zone_count(self, zone_id: str) -> int:
        if zone_id in self.zone_counters:
            return self.zone_counters[zone_id].current_count
        return 0

    def get_total_count(self, only_validated: bool = True) -> int:
        return len(self.validated_tracks) if only_validated else len(self.active_tracks)

    def reset(self):
        """Reset scene counters and zone states."""
        self.active_tracks.clear()
        self.validated_tracks.clear()
        self.tentative_tracks.clear()
        self.total_unique_seen.clear()
        self.latest_report = None
        for zc in self.zone_counters.values():
            zc.current_count = 0
            zc.entries = 0
            zc.exits = 0
            zc.unique_tracks.clear()
            zc.track_states.clear()
            zc.validated_count = 0
        for lc in self.line_counters.values():
            lc.reset()


class PeopleCounter:
    """
    Backward-compatible single-zone or scene People Counter.
    Wraps ZoneCounter and seamlessly supports both single-zone updates and multi-zone queries.
    """

    def __init__(
        self,
        zone_id: str = "default_zone",
        zone_polygon: Optional[List[List[float]]] = None,
        capacity: int = 100,
        area_m2: Optional[float] = None,
    ):
        self.counter = ZoneCounter(zone_id, zone_polygon, capacity=capacity, area_m2=area_m2)
        self.zone_id = zone_id

    def update(
        self,
        tracks: List[Any],
        context: Optional[AnalyticsContext] = None,
        frame_number: int = 0,
        timestamp: float = 0.0,
        frame_rate: float = 30.0,
        only_validated: bool = False,
    ) -> CountingResult:
        if context is not None:
            f_num = context.frame_number
            ts = context.timestamp
        else:
            f_num = frame_number
            ts = timestamp
        self.counter.update(tracks, f_num, frame_rate=frame_rate, only_validated=only_validated)
        return self.counter.get_counting_result(ts)

    def get_current_count(self) -> int:
        return self.counter.current_count

    def get_total_entries(self) -> int:
        return self.counter.entries

    def get_total_exits(self) -> int:
        return self.counter.exits

    def get_unique_tracks(self) -> int:
        return len(self.counter.unique_tracks)

    def get_validated_count(self) -> int:
        return self.counter.validated_count

    def reset(self):
        poly = self.counter.zone_polygon.tolist() if self.counter.zone_polygon is not None else None
        self.counter = ZoneCounter(self.zone_id, poly, capacity=self.counter.capacity, area_m2=self.counter.area_m2)
