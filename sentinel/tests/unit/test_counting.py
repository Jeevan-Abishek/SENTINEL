"""
Unit tests for People Count & Tracking module (Phase 10).
Verifies active count, zone-wise counts, foot-contact polygon containment,
virtual gate line-crossing, anti-flicker debouncing, and ScenePeopleCounter aggregation.
"""

import pytest
import numpy as np
import time

from app.analytics.models import (
    CountingResult,
    SceneCountReport,
    ZoneCounter,
    AnalyticsContext,
)
from app.analytics.counting import (
    PeopleCounter,
    LineCrossCounter,
    ScenePeopleCounter,
)
from app.tracking.models import Track, TrackResult


class DummyTrack:
    """Mock Track conforming to Track / TrackedPerson interface."""
    def __init__(
        self,
        track_id: int,
        bbox: list[float],
        state: str = "active",
        is_validated: bool = True,
        centroid_history: list[list[float]] = None,
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.state = state
        self.is_validated = is_validated
        self.centroid_history = centroid_history or []

    @property
    def centroid(self) -> list[float]:
        return [(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0]

    @property
    def center_xy(self) -> tuple[float, float]:
        c = self.centroid
        return (float(c[0]), float(c[1]))

    @property
    def foot_xy(self) -> tuple[float, float]:
        return (float(self.center_xy[0]), float(self.bbox[3]))


class TestEnhancedZoneCounter:
    """Tests for ZoneCounter foot-contact containment, precompiled path caching, and validation filtering."""

    def test_zone_counter_foot_xy_ground_contact(self):
        # Polygon covers ground region y in [90, 150], x in [0, 100]
        # A person with bbox [40, 20, 60, 100] has centroid at (50, 60) [OUTSIDE] but foot at (50, 100) [INSIDE]
        polygon = [[0, 90], [100, 90], [100, 150], [0, 150]]
        zc = ZoneCounter(zone_id="gate_a", zone_polygon=polygon, capacity=50, area_m2=100.0)

        track = DummyTrack(1, [40, 20, 60, 100])
        zc.update([track], frame_number=1)

        assert zc.current_count == 1
        assert zc.entries == 1
        assert zc.is_inside(track.foot_xy) is True

    def test_zone_counter_only_validated_filter(self):
        polygon = [[0, 0], [200, 0], [200, 200], [0, 200]]
        zc = ZoneCounter(zone_id="assembly", zone_polygon=polygon)

        t1_validated = DummyTrack(1, [20, 20, 40, 60], is_validated=True)
        t2_tentative = DummyTrack(2, [80, 80, 100, 120], is_validated=False)

        # When only_validated=True, tentative track 2 is ignored
        zc.update([t1_validated, t2_tentative], frame_number=1, only_validated=True)
        assert zc.current_count == 1
        assert 1 in zc.unique_tracks
        assert 2 not in zc.unique_tracks

        # When only_validated=False, both active tracks are counted
        zc.update([t1_validated, t2_tentative], frame_number=2, only_validated=False)
        assert zc.current_count == 2
        assert 2 in zc.unique_tracks

    def test_zone_counter_precompiled_path_performance(self):
        polygon = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
        zc = ZoneCounter(zone_id="large_zone", zone_polygon=polygon)

        # 10,000 containment checks should finish in under 100ms with cached path
        t0 = time.perf_counter()
        for i in range(5000):
            zc.is_inside((500.0, 500.0))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 200.0, f"Containment checks took {elapsed_ms:.1f}ms, expected < 200ms"

    def test_zone_counter_stale_track_pruning(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        zc = ZoneCounter(zone_id="zone_1", zone_polygon=polygon)

        t1 = DummyTrack(1, [10, 10, 30, 30])
        zc.update([t1], frame_number=1)
        assert 1 in zc.track_states

        # Advance 350 frames with no tracks present
        zc.update([], frame_number=355)
        assert zc.current_count == 0
        assert 1 not in zc.track_states


class TestLineCrossCounter:
    """Tests for virtual gate line crossing, direction resolution, anti-flicker debouncing, and rate calculation."""

    def test_line_cross_inflow_and_outflow(self):
        # Horizontal line from (0, 100) to (200, 100)
        # Inflow: moving from y < 100 to y > 100
        # Outflow: moving from y > 100 to y < 100
        line = [[0, 100], [200, 100]]
        lc = LineCrossCounter("gate_north", line, cooldown_frames=20)

        # Frame 1: Person at (100, 80) [Side 1]
        t1 = DummyTrack(1, [90, 60, 110, 80])
        res1 = lc.update([t1], frame_number=1, timestamp=0.033)
        assert res1["total_inflow"] == 0
        assert res1["total_outflow"] == 0

        # Frame 2: Person moves across line to (100, 120) [Side 2] -> INFLOW!
        t1_next = DummyTrack(1, [90, 100, 110, 120])
        res2 = lc.update([t1_next], frame_number=2, timestamp=0.066)
        assert res2["total_inflow"] == 1
        assert res2["total_outflow"] == 0
        assert 1 in res2["inflow_events"]

    def test_line_cross_anti_flicker_debouncing(self):
        # Line from (0, 100) to (200, 100), cooldown = 25 frames
        line = [[0, 100], [200, 100]]
        lc = LineCrossCounter("gate_a", line, cooldown_frames=25)

        # Frame 1: Person at (100, 95)
        lc.update([DummyTrack(1, [90, 75, 110, 95])], frame_number=1, timestamp=0.033)
        # Frame 2: Person crosses to (100, 105) -> Inflow (1)
        r2 = lc.update([DummyTrack(1, [90, 85, 110, 105])], frame_number=2, timestamp=0.066)
        assert r2["total_inflow"] == 1

        # Frame 3: Sensor noise / jitter pushes person back to (100, 95) within cooldown!
        # Should be debounced and NOT increment outflow or duplicate entry!
        r3 = lc.update([DummyTrack(1, [90, 75, 110, 95])], frame_number=3, timestamp=0.10)
        assert r3["total_outflow"] == 0
        assert len(r3["outflow_events"]) == 0

        # Frame 4: Person jitters forward to (100, 105) again within cooldown!
        # Should be debounced and NOT increment inflow!
        r4 = lc.update([DummyTrack(1, [90, 85, 110, 105])], frame_number=4, timestamp=0.133)
        assert r4["total_inflow"] == 1
        assert len(r4["inflow_events"]) == 0

        # Frame 30: 28 frames later (> cooldown of 25), person intentionally crosses back to (100, 95)
        r30 = lc.update([DummyTrack(1, [90, 75, 110, 95])], frame_number=30, timestamp=1.0)
        assert r30["total_outflow"] == 1
        assert 1 in r30["outflow_events"]

    def test_line_cross_parallel_movement_no_crossing(self):
        # Line from (0, 100) to (200, 100)
        line = [[0, 100], [200, 100]]
        lc = LineCrossCounter("gate_a", line)

        # Track moves horizontally from (50, 50) to (150, 50) without crossing y=100
        lc.update([DummyTrack(1, [40, 30, 60, 50])], frame_number=1, timestamp=0.033)
        r2 = lc.update([DummyTrack(1, [140, 30, 160, 50])], frame_number=2, timestamp=0.066)
        assert r2["total_inflow"] == 0
        assert r2["total_outflow"] == 0

    def test_line_cross_rolling_ppm_rates(self):
        line = [[0, 50], [100, 50]]
        lc = LineCrossCounter("gate_b", line, flow_window_seconds=60.0, cooldown_frames=5)

        # Simulate 6 people crossing inflow across 10 seconds
        for i in range(6):
            tid = i + 10
            lc.update([DummyTrack(tid, [40, 20, 60, 40])], frame_number=i * 5 + 1, timestamp=float(i))
            lc.update([DummyTrack(tid, [40, 55, 60, 75])], frame_number=i * 5 + 2, timestamp=float(i) + 0.1)

        # 6 entries in 60s window = 6.0 persons/minute
        assert lc.total_inflow == 6
        assert lc.get_inflow_rate_ppm() == 6.0
        assert lc.get_net_flow_rate_ppm() == 6.0


class TestScenePeopleCounter:
    """Tests for unified ScenePeopleCounter multi-zone aggregation and SceneCountReport generation."""

    def test_scene_people_counter_multi_zone(self):
        counter = ScenePeopleCounter()
        counter.register_zone("north_ingress", [[0, 0], [100, 0], [100, 100], [0, 100]], capacity=20, area_m2=50.0)
        counter.register_zone("central_plaza", [[200, 0], [400, 0], [400, 200], [200, 200]], capacity=100, area_m2=200.0)
        counter.register_line("north_gate", [[0, 100], [100, 100]])

        # Create tracks
        t1 = DummyTrack(1, [20, 20, 40, 60], is_validated=True)    # in north_ingress
        t2 = DummyTrack(2, [220, 20, 240, 60], is_validated=True)  # in central_plaza
        t3 = DummyTrack(3, [250, 20, 270, 60], is_validated=False) # tentative in central_plaza
        t4 = DummyTrack(4, [500, 500, 520, 540], is_validated=True) # outside both zones

        report = counter.update(
            [t1, t2, t3, t4],
            frame_number=1,
            timestamp=0.033,
            rejected_count=2,
            reid_matches=1,
        )

        assert isinstance(report, SceneCountReport)
        assert report.total_active_count == 4
        assert report.total_validated_count == 3
        assert report.total_tentative_count == 1
        assert report.rejected_count == 2
        assert report.reid_matches == 1

        # Check zone breakdown
        assert report.zone_counts["north_ingress"] == 1
        assert report.zone_counts["central_plaza"] == 2
        assert report.zone_densities["north_ingress"] == round(1 / 50.0, 3)
        assert report.zone_densities["central_plaza"] == round(2 / 200.0, 3)
        assert report.zone_status["north_ingress"] == "safe"
        assert report.zone_status["central_plaza"] == "safe"

    def test_scene_people_counter_from_zone_manager(self):
        class MockZoneDef:
            def __init__(self, zid, name, poly, cap, area, entry_l=None, exit_l=None):
                self.zone_id = zid
                self.name = name
                self.image_polygon = poly
                self.max_capacity = cap
                self.area_sqm = area
                self.entry_line = entry_l
                self.exit_line = exit_l

        class MockZoneManager:
            def __init__(self):
                self.zones = {
                    "z1": MockZoneDef("z1", "Ingress", [[0, 0], [100, 0], [100, 100], [0, 100]], 50, 80.0, entry_l=[[0, 50], [100, 50]]),
                    "z2": MockZoneDef("z2", "Egress", [[200, 0], [300, 0], [300, 100], [200, 100]], 30, 40.0, exit_l=[[200, 50], [300, 50]]),
                }

        zm = MockZoneManager()
        counter = ScenePeopleCounter.from_zone_manager(zm)

        assert "z1" in counter.zone_counters
        assert "z2" in counter.zone_counters
        assert "z1_entry" in counter.line_counters
        assert "z2_exit" in counter.line_counters

    def test_scene_people_counter_with_track_result_object(self):
        t1 = Track(track_id=1, bbox=[10, 10, 30, 50], confidence=0.85, is_validated=True)
        t2 = Track(track_id=2, bbox=[20, 20, 40, 60], confidence=0.75, is_validated=False)
        track_result = TrackResult(
            tracks=[t1, t2],
            frame_number=10,
            timestamp=0.333,
            processing_time_ms=12.5,
            num_detections=2,
            rejected_count=1,
            reid_matches=1,
        )

        counter = ScenePeopleCounter()
        report = counter.update(track_result)

        assert report.frame_number == 10
        assert report.total_active_count == 2
        assert report.total_validated_count == 1
        assert report.total_tentative_count == 1
        assert report.rejected_count == 1
        assert report.reid_matches == 1

    def test_scene_people_counter_empty_frame(self):
        counter = ScenePeopleCounter()
        report = counter.update([], frame_number=1, timestamp=0.033)

        assert report.total_active_count == 0
        assert report.total_validated_count == 0
        assert report.net_flow == 0
        assert report.inflow_rate_ppm == 0.0

    def test_scene_people_counter_reset(self):
        counter = ScenePeopleCounter()
        counter.register_zone("z1", [[0, 0], [100, 0], [100, 100], [0, 100]])
        t = DummyTrack(1, [10, 10, 30, 50], is_validated=True)
        counter.update([t], frame_number=1, timestamp=0.033)

        assert counter.get_total_count() == 1
        counter.reset()
        assert counter.get_total_count() == 0
        assert counter.latest_report is None


class TestPeopleCounterBackwardCompatibility:
    """Verifies that the PeopleCounter wrapper remains 100% backward-compatible."""

    def test_wrapper_creation_and_properties(self):
        pc = PeopleCounter("turnstile_zone", [[0, 0], [100, 0], [100, 100], [0, 100]], capacity=40)
        assert pc.zone_id == "turnstile_zone"
        assert pc.get_current_count() == 0
        assert pc.get_total_entries() == 0
        assert pc.get_total_exits() == 0
        assert pc.get_unique_tracks() == 0

    def test_wrapper_update_with_context(self):
        pc = PeopleCounter("turnstile_zone")
        ctx = AnalyticsContext(zone_id="turnstile_zone", frame_number=5, timestamp=0.166)
        t = DummyTrack(1, [10, 10, 30, 30])
        res = pc.update([t], context=ctx)

        assert isinstance(res, CountingResult)
        assert res.zone_id == "turnstile_zone"
        assert res.current_count == 1
        assert res.entries == 1
        assert pc.get_current_count() == 1
        assert pc.get_unique_tracks() == 1

