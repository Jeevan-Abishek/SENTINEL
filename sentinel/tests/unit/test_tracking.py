"""
Unit tests for SENTINEL Tracking Module
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.tracking.models import Track, TrackResult, TrackerConfig, TrackingContext, TrackingError, TrackerInitError
from app.tracking.bytetrack import KalmanBoxTracker, ByteTrack, BYTETracker
from app.tracking.tracker import Tracker, create_tracker_wrapper
from app.detection.models import Detection, DetectionResult


class TestTrackModel:
    def test_track_creation(self):
        track = Track(
            track_id=1,
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
        assert track.track_id == 1
        assert track.x1 == 100
        assert track.y1 == 100
        assert track.x2 == 200
        assert track.y2 == 200
        assert track.width == 100
        assert track.height == 100
        assert track.center_x == 150
        assert track.center_y == 150
        assert track.centroid == [150, 150]
        assert track.area == 10000

    def test_track_to_xywh(self):
        track = Track(track_id=1, bbox=[100, 100, 200, 200], confidence=0.9)
        xywh = track.to_xywh()
        assert xywh == [100, 100, 100, 100]


class TestTrackResult:
    def test_track_result_creation(self):
        tracks = [
            Track(track_id=1, bbox=[100, 100, 200, 200], confidence=0.9, state="active"),
            Track(track_id=2, bbox=[300, 300, 400, 400], confidence=0.8, state="lost"),
        ]
        result = TrackResult(tracks=tracks, frame_number=1, timestamp=0.033)
        assert result.num_active_tracks == 1
        assert result.num_lost_tracks == 1
        assert len(result.active_tracks) == 1
        assert len(result.lost_tracks) == 1
        assert result.active_tracks[0].track_id == 1
        assert result.lost_tracks[0].track_id == 2


class TestTrackerConfig:
    def test_default_config(self):
        config = TrackerConfig()
        assert config.tracker_type == "bytetrack"
        assert config.track_thresh == 0.5
        assert config.track_buffer == 30
        assert config.match_thresh == 0.8

    def test_custom_config(self):
        config = TrackerConfig(
            tracker_type="botsort",
            track_thresh=0.6,
            track_buffer=50,
            reid_enabled=True,
        )
        assert config.tracker_type == "botsort"
        assert config.track_thresh == 0.6
        assert config.track_buffer == 50
        assert config.reid_enabled is True


class TestTrackingContext:
    def test_context_creation(self):
        ctx = TrackingContext(frame_number=100, timestamp=3.33, original_frame_shape=(720, 1280))
        assert ctx.frame_number == 100
        assert ctx.timestamp == 3.33
        assert ctx.original_frame_shape == (720, 1280)


class TestTrackingErrors:
    def test_tracking_error_inheritance(self):
        assert issubclass(TrackerInitError, TrackingError)


class TestKalmanBoxTracker:
    def test_kalman_tracker_creation(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        assert tracker.id >= 0
        assert tracker.hits == 1
        assert tracker.age == 0
        assert tracker.state == "active"

    def test_kalman_tracker_predict(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        predicted = tracker.predict()
        assert tracker.age == 1
        assert tracker.time_since_update == 1
        assert predicted.shape == (1, 4)

    def test_kalman_tracker_update(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        new_bbox = np.array([110, 110, 210, 210], dtype=np.float32)
        tracker.update(new_bbox)
        assert tracker.time_since_update == 0
        assert tracker.hits == 2
        assert tracker.hit_streak == 2

    def test_kalman_tracker_get_state(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        tracker = KalmanBoxTracker(bbox)
        state = tracker.get_state()
        assert state.shape == (4,)
        assert np.allclose(state, bbox, atol=1.0)

    def test_kalman_tracker_unique_ids(self):
        bbox = np.array([100, 100, 200, 200], dtype=np.float32)
        t1 = KalmanBoxTracker(bbox)
        t2 = KalmanBoxTracker(bbox)
        t3 = KalmanBoxTracker(bbox, track_id=999)
        assert t1.id != t2.id
        assert t3.id == 999


class TestByteTrack:
    def test_bytetrack_creation(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)
        assert tracker.frame_id == 0
        assert len(tracker.tracked_tracks) == 0

    def test_bytetrack_update_empty(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)
        detections = np.empty((0, 4))
        scores = np.empty((0,))
        active = tracker.update(detections, scores)
        assert len(active) == 0

    def test_bytetrack_update_single_detection(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)
        detections = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        active = tracker.update(detections, scores)
        assert len(active) == 1
        assert active[0].hits == 1
        assert active[0].state == "active"

    def test_bytetrack_track_persistence(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)
        detections = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        # First frame
        active1 = tracker.update(detections, scores)
        track_id = active1[0].id

        # Second frame - slightly moved
        detections2 = np.array([[105, 105, 205, 205]], dtype=np.float32)
        scores2 = np.array([0.9], dtype=np.float32)
        active2 = tracker.update(detections2, scores2)

        assert len(active2) == 1
        assert active2[0].id == track_id
        assert active2[0].hits == 2

    def test_bytetrack_track_loss_and_recovery(self):
        config = TrackerConfig(track_buffer=3)
        tracker = ByteTrack(config)
        detections = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        # Create track
        tracker.update(detections, scores)
        track_id = tracker.tracked_tracks[0].id

        # Miss 2 frames (within buffer)
        for _ in range(2):
            tracker.update(np.empty((0, 4)), np.empty((0,)))

        # Track should be in lost
        assert len(tracker.lost_tracks) == 1
        assert tracker.lost_tracks[0].id == track_id

        # Reappear
        detections2 = np.array([[110, 110, 210, 210]], dtype=np.float32)
        scores2 = np.array([0.9], dtype=np.float32)
        active = tracker.update(detections2, scores2)

        # Track should be recovered
        assert len(active) == 1
        assert active[0].id == track_id
        assert active[0].state == "active"

    def test_bytetrack_track_removal(self):
        config = TrackerConfig(track_buffer=2)
        tracker = ByteTrack(config)
        detections = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)

        tracker.update(detections, scores)

        # Miss 3 frames (exceeds buffer)
        for _ in range(3):
            tracker.update(np.empty((0, 4)), np.empty((0,)))

        # Track should be removed
        assert len(tracker.tracked_tracks) == 0
        assert len(tracker.lost_tracks) == 0
        assert len(tracker.removed_tracks) == 1

    def test_bytetrack_multiple_tracks(self):
        config = TrackerConfig()
        tracker = ByteTrack(config)

        # Two detections
        detections = np.array([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)

        active = tracker.update(detections, scores)
        assert len(active) == 2
        assert active[0].id != active[1].id


class TestBYTETracker:
    def test_bytetracker_initialize(self):
        config = TrackerConfig()
        tracker = BYTETracker(config)
        result = tracker.initialize()
        assert result is True
        assert tracker._is_initialized is True

    def test_bytetracker_update(self):
        config = TrackerConfig()
        tracker = BYTETracker(config)
        tracker.initialize()

        detections = np.array([[100, 100, 200, 200]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        active = tracker.update(detections, scores)
        assert len(active) == 1


class TestTrackerWrapper:
    def test_tracker_creation(self):
        config = TrackerConfig()
        tracker = Tracker(config)
        assert tracker.config == config

    def test_tracker_update(self):
        config = TrackerConfig()
        tracker = Tracker(config)
        tracker.initialize()

        detections = [
            Detection(bbox=[100, 100, 200, 200], confidence=0.9),
            Detection(bbox=[300, 300, 400, 400], confidence=0.8),
        ]
        det_result = DetectionResult(detections=detections, frame_number=1, timestamp=0.033)
        context = TrackingContext(frame_number=1, timestamp=0.033, original_frame_shape=(720, 1280))

        result = tracker.update(det_result, context)

        assert isinstance(result, TrackResult)
        assert result.frame_number == 1
        assert result.timestamp == 0.033
        assert result.processing_time_ms > 0
        assert result.num_detections == 2
        assert len(result.tracks) == 2

    def test_tracker_track_persistence(self):
        config = TrackerConfig()
        tracker = Tracker(config)
        tracker.initialize()

        # Frame 1
        detections1 = [Detection(bbox=[100, 100, 200, 200], confidence=0.9)]
        det_result1 = DetectionResult(detections=detections1, frame_number=1, timestamp=0.033)
        context1 = TrackingContext(frame_number=1, timestamp=0.033, original_frame_shape=(720, 1280))
        result1 = tracker.update(det_result1, context1)
        track_id = result1.tracks[0].track_id

        # Frame 2 - moved slightly
        detections2 = [Detection(bbox=[105, 105, 205, 205], confidence=0.9)]
        det_result2 = DetectionResult(detections=detections2, frame_number=2, timestamp=0.066)
        context2 = TrackingContext(frame_number=2, timestamp=0.066, original_frame_shape=(720, 1280))
        result2 = tracker.update(det_result2, context2)

        assert len(result2.tracks) == 1
        assert result2.tracks[0].track_id == track_id
        assert result2.tracks[0].hits == 2

    def test_tracker_reset(self):
        config = TrackerConfig()
        tracker = Tracker(config)
        tracker.initialize()

        detections = [Detection(bbox=[100, 100, 200, 200], confidence=0.9)]
        det_result = DetectionResult(detections=detections, frame_number=1, timestamp=0.033)
        context = TrackingContext(frame_number=1, timestamp=0.033, original_frame_shape=(720, 1280))
        tracker.update(det_result, context)

        tracker.reset()

        # Should be able to track again with new IDs
        detections2 = [Detection(bbox=[100, 100, 200, 200], confidence=0.9)]
        det_result2 = DetectionResult(detections=detections2, frame_number=2, timestamp=0.066)
        context2 = TrackingContext(frame_number=2, timestamp=0.066, original_frame_shape=(720, 1280))
        result2 = tracker.update(det_result2, context2)

        # After reset, track IDs should start from 0 or 1 again
        assert len(result2.tracks) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])