"""
Unit tests for SENTINEL Analytics Module
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.analytics.models import (
    CountingResult,
    DensityResult,
    OccupancyResult,
    MovementResult,
    FlowResult,
    BottleneckResult,
    AnalyticsContext,
    AnalyticsConfig,
    TrackState,
    ZoneCounter,
    DensityAnalyzer,
    OccupancyAnalyzer,
    MovementAnalyzer,
    FlowAnalyzer,
    BottleneckDetector,
)
from app.analytics.counting import PeopleCounter
from app.analytics.density import DensityEstimator
from app.analytics.occupancy import OccupancyMonitor
from app.analytics.movement import MovementAnalyzerWrapper
from app.analytics.flow import FlowAnalyzerWrapper
from app.analytics.bottleneck import BottleneckDetectorWrapper


class MockTrack:
    def __init__(self, track_id, bbox, state="active", centroid_history=None):
        self.track_id = track_id
        self.bbox = bbox
        self.state = state
        self._centroid_history = centroid_history or []
    
    @property
    def centroid(self):
        x1, y1, x2, y2 = self.bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    @property
    def centroid_history(self):
        return self._centroid_history
    
    def get_velocity(self, frame_rate=30.0):
        if len(self._centroid_history) < 2:
            return [0.0, 0.0]
        prev = np.array(self._centroid_history[-2])
        curr = np.array(self._centroid_history[-1])
        velocity = (curr - prev) * frame_rate
        return velocity.tolist()
    
    def get_speed(self, frame_rate=30.0):
        vel = self.get_velocity(frame_rate)
        return float(np.linalg.norm(vel))
    
    def get_direction(self, frame_rate=30.0):
        vel = self.get_velocity(frame_rate)
        if vel[0] == 0 and vel[1] == 0:
            return 0.0
        return float(np.arctan2(vel[1], vel[0]))


class TestAnalyticsModels:
    def test_counting_result_creation(self):
        result = CountingResult(
            zone_id="zone-1",
            timestamp=1.0,
            current_count=10,
            entries=5,
            exits=3,
            unique_tracks=12,
        )
        assert result.zone_id == "zone-1"
        assert result.current_count == 10

    def test_density_result_creation(self):
        from app.analytics.models import DensityGrid
        grid = DensityGrid(
            grid_data=[[1.0, 2.0], [3.0, 4.0]],
            grid_width=2,
            grid_height=2,
            cell_size_meters=1.0,
            max_density=4.0,
        )
        result = DensityResult(
            zone_id="zone-1",
            timestamp=1.0,
            grid=grid,
            avg_density=2.5,
            max_density=4.0,
        )
        assert result.zone_id == "zone-1"
        assert result.max_density == 4.0

    def test_occupancy_result_creation(self):
        result = OccupancyResult(
            zone_id="zone-1",
            timestamp=1.0,
            capacity=100,
            current_count=70,
            occupancy_ratio=0.7,
            status="warning",
        )
        assert result.status == "warning"
        assert result.occupancy_ratio == 0.7

    def test_movement_result_creation(self):
        from app.analytics.models import MovementVector
        mv = MovementVector(
            track_id=1,
            centroid=[100, 100],
            velocity=[1.0, 2.0],
            speed=2.236,
            direction=1.107,
        )
        result = MovementResult(
            zone_id="zone-1",
            timestamp=1.0,
            movements=[mv],
            avg_speed=2.236,
        )
        assert len(result.movements) == 1
        assert result.avg_speed == 2.236

    def test_flow_result_creation(self):
        result = FlowResult(
            zone_id="zone-1",
            timestamp=1.0,
            flow_rate=5.0,
            entry_flow=3.0,
            exit_flow=2.0,
            net_flow=1.0,
        )
        assert result.net_flow == 1.0

    def test_bottleneck_result_creation(self):
        result = BottleneckResult(
            zone_id="zone-1",
            timestamp=1.0,
            bottleneck_score=0.8,
            is_bottleneck=True,
            congestion_level="high",
        )
        assert result.is_bottleneck is True
        assert result.congestion_level == "high"

    def test_analytics_context_creation(self):
        ctx = AnalyticsContext(
            zone_id="zone-1",
            frame_number=100,
            timestamp=3.33,
            zone_polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
            zone_capacity=500,
        )
        assert ctx.zone_id == "zone-1"
        assert ctx.zone_capacity == 500


class TestAnalyticsConfig:
    def test_default_config(self):
        config = AnalyticsConfig()
        assert config.density_grid_size == 1.0
        assert config.occupancy_warning_threshold == 0.7
        assert config.occupancy_critical_threshold == 0.9
        assert config.flow_window_seconds == 60

    def test_custom_config(self):
        config = AnalyticsConfig(
            density_grid_size=2.0,
            occupancy_warning_threshold=0.8,
        )
        assert config.density_grid_size == 2.0
        assert config.occupancy_warning_threshold == 0.8


class TestTrackState:
    def test_track_state_creation(self):
        state = TrackState(track_id=1)
        assert state.track_id == 1
        assert len(state.centroid_history) == 0

    def test_add_position(self):
        state = TrackState(track_id=1)
        state.add_position([100, 100], 1)
        state.add_position([110, 110], 2)
        assert len(state.centroid_history) == 2
        assert state.total_distance > 0

    def test_get_velocity(self):
        state = TrackState(track_id=1)
        state.add_position([100, 100], 1)
        state.add_position([110, 110], 2)
        vel = state.get_velocity(30.0)
        assert vel[0] > 0
        assert vel[1] > 0

    def test_get_speed(self):
        state = TrackState(track_id=1)
        state.add_position([100, 100], 1)
        state.add_position([110, 110], 2)
        speed = state.get_speed(30.0)
        assert speed > 0


class TestZoneCounter:
    def test_zone_counter_creation(self):
        counter = ZoneCounter("zone-1")
        assert counter.zone_id == "zone-1"
        assert counter.current_count == 0

    def test_zone_counter_with_polygon(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        counter = ZoneCounter("zone-1", polygon)
        assert counter.zone_polygon is not None
        assert counter.is_inside([50, 50]) is True
        assert counter.is_inside([150, 150]) is False

    def test_zone_counter_update(self):
        counter = ZoneCounter("zone-1")
        tracks = [
            MockTrack(1, [10, 10, 20, 20]),
            MockTrack(2, [30, 30, 40, 40]),
        ]
        counter.update(tracks, frame_number=1, frame_rate=30.0)
        assert counter.current_count == 2
        assert counter.entries == 2

    def test_zone_counter_exit(self):
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        counter = ZoneCounter("zone-1", polygon)
        tracks = [MockTrack(1, [10, 10, 20, 20])]
        counter.update(tracks, frame_number=1, frame_rate=30.0)
        assert counter.entries == 1
        
        tracks_out = [MockTrack(1, [200, 200, 210, 210])]
        counter.update(tracks_out, frame_number=2, frame_rate=30.0)
        assert counter.exits == 1
        assert counter.current_count == 0

    def test_get_counting_result(self):
        counter = ZoneCounter("zone-1")
        tracks = [MockTrack(1, [10, 10, 20, 20])]
        counter.update(tracks, frame_number=1, frame_rate=30.0)
        result = counter.get_counting_result(0.033)
        assert isinstance(result, CountingResult)
        assert result.current_count == 1


class TestDensityAnalyzer:
    def test_simple_density(self):
        config = AnalyticsConfig()
        analyzer = DensityAnalyzer(config)
        tracks = [MockTrack(i, [i*10, i*10, i*10+10, i*10+10]) for i in range(5)]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = analyzer.compute_density(tracks, ctx)
        assert isinstance(result, DensityResult)
        assert result.avg_density > 0

    def test_density_with_polygon(self):
        config = AnalyticsConfig(density_grid_size=10.0)
        analyzer = DensityAnalyzer(config)
        polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
        tracks = [MockTrack(i, [i*20, i*20, i*20+10, i*20+10]) for i in range(3)]
        ctx = AnalyticsContext(
            zone_id="zone-1", 
            frame_number=1, 
            timestamp=0.033,
            zone_polygon=polygon,
        )
        result = analyzer.compute_density(tracks, ctx)
        assert isinstance(result, DensityResult)
        assert result.grid.grid_width > 0
        assert result.grid.grid_height > 0


class TestOccupancyAnalyzer:
    def test_safe_occupancy(self):
        config = AnalyticsConfig()
        analyzer = OccupancyAnalyzer(config)
        result = analyzer.analyze(30, 100, 1.0, "zone-1")
        assert result.status == "safe"
        assert result.occupancy_ratio == 0.3

    def test_warning_occupancy(self):
        config = AnalyticsConfig()
        analyzer = OccupancyAnalyzer(config)
        result = analyzer.analyze(75, 100, 1.0, "zone-1")
        assert result.status == "warning"
        assert result.occupancy_ratio == 0.75

    def test_critical_occupancy(self):
        config = AnalyticsConfig()
        analyzer = OccupancyAnalyzer(config)
        result = analyzer.analyze(95, 100, 1.0, "zone-1")
        assert result.status == "critical"
        assert result.occupancy_ratio == 0.95

    def test_zero_capacity(self):
        config = AnalyticsConfig()
        analyzer = OccupancyAnalyzer(config)
        result = analyzer.analyze(10, 0, 1.0, "zone-1")
        assert result.status == "safe"
        assert result.occupancy_ratio == 0.0


class TestMovementAnalyzer:
    def test_movement_analysis(self):
        config = AnalyticsConfig()
        analyzer = MovementAnalyzer(config)
        tracks = [
            MockTrack(1, [10, 10, 20, 20], centroid_history=[[15, 15], [16, 16], [17, 17]]),
            MockTrack(2, [30, 30, 40, 40], centroid_history=[[35, 35], [36, 36]]),
        ]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = analyzer.analyze(tracks, ctx)
        assert isinstance(result, MovementResult)
        assert len(result.movements) == 2
        assert result.avg_speed > 0

    def test_dominant_direction(self):
        config = AnalyticsConfig()
        analyzer = MovementAnalyzer(config)
        tracks = [
            MockTrack(1, [10, 10, 20, 20], centroid_history=[[15, 15], [25, 15]]),
            MockTrack(2, [30, 30, 40, 40], centroid_history=[[35, 35], [45, 35]]),
        ]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = analyzer.analyze(tracks, ctx)
        assert result.dominant_direction is not None


class TestFlowAnalyzer:
    def test_flow_analysis(self):
        config = AnalyticsConfig()
        analyzer = FlowAnalyzer(config)
        tracks = [
            MockTrack(1, [10, 10, 20, 20], centroid_history=[[15, 15], [16, 16]]),
            MockTrack(2, [30, 30, 40, 40], centroid_history=[[35, 35], [36, 36]]),
        ]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = analyzer.analyze(tracks, ctx)
        assert isinstance(result, FlowResult)
        assert result.flow_rate >= 0

    def test_flow_with_lines(self):
        config = AnalyticsConfig()
        analyzer = FlowAnalyzer(config)
        tracks = [
            MockTrack(1, [10, 10, 20, 20], centroid_history=[[5, 5], [15, 5], [25, 5]]),
        ]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        entry_lines = [{"start": [10, 0], "end": [10, 20]}]
        result = analyzer.analyze(tracks, ctx, entry_lines=entry_lines)
        assert result.entry_flow > 0


class TestBottleneckDetector:
    def test_no_bottleneck(self):
        config = AnalyticsConfig()
        detector = BottleneckDetector(config)
        
        from app.analytics.models import DensityGrid
        density = DensityResult(
            zone_id="zone-1",
            timestamp=1.0,
            grid=DensityGrid(grid_data=[[0.5]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=0.5),
            avg_density=0.5,
            max_density=0.5,
        )
        flow = FlowResult(zone_id="zone-1", timestamp=1.0, flow_rate=1.0, entry_flow=0.5, exit_flow=0.5, net_flow=0.0)
        movement = MovementResult(zone_id="zone-1", timestamp=1.0, movements=[], avg_speed=2.0)
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=1.0)
        
        result = detector.analyze(density, flow, movement, ctx)
        assert isinstance(result, BottleneckResult)
        assert result.congestion_level in ["none", "low"]

    def test_bottleneck_detection(self):
        config = AnalyticsConfig(bottleneck_persistence_frames=5)
        detector = BottleneckDetector(config)
        
        from app.analytics.models import DensityGrid
        density = DensityResult(
            zone_id="zone-1",
            timestamp=1.0,
            grid=DensityGrid(grid_data=[[5.0]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=5.0),
            avg_density=5.0,
            max_density=5.0,
        )
        flow = FlowResult(zone_id="zone-1", timestamp=1.0, flow_rate=10.0, entry_flow=5.0, exit_flow=5.0, net_flow=0.0)
        movement = MovementResult(zone_id="zone-1", timestamp=1.0, movements=[], avg_speed=0.1)
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=1.0)
        
        for _ in range(5):
            result = detector.analyze(density, flow, movement, ctx)
        
        assert result.is_bottleneck is True
        assert result.congestion_level in ["medium", "high", "critical"]


class TestPeopleCounter:
    def test_counter_creation(self):
        counter = PeopleCounter("zone-1")
        assert counter.zone_id == "zone-1"

    def test_counter_update(self):
        counter = PeopleCounter("zone-1")
        tracks = [MockTrack(1, [10, 10, 20, 20])]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = counter.update(tracks, ctx)
        assert isinstance(result, CountingResult)
        assert result.current_count == 1


class TestDensityEstimator:
    def test_estimator_creation(self):
        estimator = DensityEstimator()
        assert estimator.analyzer is not None

    def test_estimate(self):
        estimator = DensityEstimator()
        tracks = [MockTrack(i, [i*10, i*10, i*10+10, i*10+10]) for i in range(3)]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = estimator.estimate(tracks, ctx)
        assert isinstance(result, DensityResult)


class TestOccupancyMonitor:
    def test_monitor_creation(self):
        monitor = OccupancyMonitor()
        assert monitor.analyzer is not None

    def test_analyze(self):
        monitor = OccupancyMonitor()
        result = monitor.analyze(50, 100, 1.0, "zone-1")
        assert isinstance(result, OccupancyResult)

    def test_thresholds(self):
        monitor = OccupancyMonitor()
        assert monitor.is_warning(0.8) is True
        assert monitor.is_warning(0.5) is False
        assert monitor.is_critical(0.95) is True
        assert monitor.is_critical(0.5) is False


class TestMovementAnalyzerWrapper:
    def test_wrapper_creation(self):
        wrapper = MovementAnalyzerWrapper()
        assert wrapper.analyzer is not None

    def test_analyze(self):
        wrapper = MovementAnalyzerWrapper()
        tracks = [MockTrack(1, [10, 10, 20, 20], centroid_history=[[15, 15], [16, 16]])]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = wrapper.analyze(tracks, ctx)
        assert isinstance(result, MovementResult)


class TestFlowAnalyzerWrapper:
    def test_wrapper_creation(self):
        wrapper = FlowAnalyzerWrapper()
        assert wrapper.analyzer is not None

    def test_analyze(self):
        wrapper = FlowAnalyzerWrapper()
        tracks = [MockTrack(1, [10, 10, 20, 20], centroid_history=[[15, 15], [16, 16]])]
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=0.033)
        result = wrapper.analyze(tracks, ctx)
        assert isinstance(result, FlowResult)


class TestBottleneckDetectorWrapper:
    def test_wrapper_creation(self):
        wrapper = BottleneckDetectorWrapper()
        assert wrapper.detector is not None

    def test_analyze(self):
        wrapper = BottleneckDetectorWrapper()
        from app.analytics.models import DensityGrid
        density = DensityResult(
            zone_id="zone-1",
            timestamp=1.0,
            grid=DensityGrid(grid_data=[[1.0]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=1.0),
            avg_density=1.0,
            max_density=1.0,
        )
        flow = FlowResult(zone_id="zone-1", timestamp=1.0, flow_rate=1.0, entry_flow=0.5, exit_flow=0.5, net_flow=0.0)
        movement = MovementResult(zone_id="zone-1", timestamp=1.0, movements=[], avg_speed=1.0)
        ctx = AnalyticsContext(zone_id="zone-1", frame_number=1, timestamp=1.0)
        result = wrapper.analyze(density, flow, movement, ctx)
        assert isinstance(result, BottleneckResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])