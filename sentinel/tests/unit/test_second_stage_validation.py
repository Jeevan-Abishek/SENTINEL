"""
Unit tests for SENTINEL Second-Stage Scanning (Tracking & Validation) Module
Verifies:
  1. TrackedPerson and TrackState data contracts, center_xy, foot_xy, velocity, speed, heading_deg.
  2. SecondStageValidator: min_box_area, aspect-ratio bounds, temporal min_hits, motion outlier gating.
  3. Lightweight pure-NumPy fallback tracker behavior.
  4. Full Tracker integration with second-stage validation.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.tracking.models import (
    Track,
    TrackedPerson,
    TrackState,
    TrackResult,
    TrackerConfig,
    TrackingContext,
)
from app.tracking.validator import (
    SecondStageValidator,
    ValidationConfig,
    ValidationReport,
)
from app.tracking.bytetrack import (
    LightweightBoxTracker,
    LightweightFallbackTracker,
    ByteTrack,
)
from app.tracking.tracker import Tracker
from app.detection.models import Detection, DetectionResult


class TestTrackedPersonModel:
    def test_tracked_person_creation_and_alias(self):
        person = TrackedPerson(
            track_id=1,
            bbox=[100.0, 100.0, 160.0, 220.0],
            confidence=0.88,
            class_id=0,
            class_name="person",
            age=5,
            hits=4,
            time_since_update=0,
            state="active",
        )
        assert person.track_id == 1
        assert isinstance(person, Track)
        assert person.is_validated is False
        assert person.validation_reason is None

    def test_center_xy_and_foot_xy(self):
        person = TrackedPerson(
            track_id=10,
            bbox=[100.0, 200.0, 200.0, 400.0],
            confidence=0.92,
        )
        # Center: ((100+200)/2, (200+400)/2) = (150.0, 300.0)
        assert person.center_xy == (150.0, 300.0)
        # Foot point for ground contact: (center_x, y2) = (150.0, 400.0)
        assert person.foot_xy == (150.0, 400.0)

    def test_aspect_ratio_and_area(self):
        person = TrackedPerson(
            track_id=5,
            bbox=[10.0, 20.0, 50.0, 120.0],
            confidence=0.85,
        )
        # Width: 40.0, Height: 100.0
        assert person.width == 40.0
        assert person.height == 100.0
        assert person.area == 4000.0
        assert person.aspect_ratio == pytest.approx(0.40, rel=1e-3)

    def test_velocity_speed_and_heading(self):
        person = TrackedPerson(
            track_id=7,
            bbox=[100.0, 100.0, 140.0, 200.0],
            confidence=0.90,
            centroid_history=[[100.0, 100.0], [130.0, 140.0]],  # dx = +30, dy = +40
        )
        # With frame_rate = 30: vx = 30*30 = 900, vy = 40*30 = 1200
        vx, vy = person.velocity_xy
        assert vx == pytest.approx(900.0, rel=1e-2)
        assert vy == pytest.approx(1200.0, rel=1e-2)
        # Speed: sqrt(900^2 + 1200^2) = 1500 px/sec
        assert person.speed == pytest.approx(1500.0, rel=1e-2)
        # Heading: arctan2(40, 30) = 53.13 degrees
        assert person.heading_deg == pytest.approx(53.13, abs=0.5)

    def test_track_state_enum(self):
        assert TrackState.ACTIVE == "active"
        assert TrackState.TENTATIVE == "tentative"
        assert TrackState.LOST == "lost"
        assert TrackState.REMOVED == "removed"

        person = TrackedPerson(
            track_id=1,
            bbox=[10.0, 10.0, 30.0, 50.0],
            confidence=0.9,
            state=TrackState.ACTIVE,
        )
        assert person.state == "active"
        assert person.state == TrackState.ACTIVE


class TestSecondStageValidator:
    @pytest.fixture
    def validator(self):
        cfg = ValidationConfig(
            min_hits=3,
            min_box_area=20.0,
            max_box_area=100000.0,
            min_aspect_ratio=0.15,
            max_aspect_ratio=3.0,
            max_velocity_jump=100.0,
            min_confidence=0.30,
            enabled=True,
        )
        return SecondStageValidator(cfg)

    def test_validate_passes_normal_person_track(self, validator):
        track = TrackedPerson(
            track_id=1,
            bbox=[100.0, 100.0, 150.0, 220.0],  # w=50, h=120 -> aspect = 0.416, area = 6000
            confidence=0.85,
            hits=4,
            centroid_history=[[125.0, 160.0], [127.0, 162.0]],  # jump = 2.8 px
            state="active",
        )
        is_valid, reason = validator.validate_track(track)
        assert is_valid is True
        assert reason == "validated"

    def test_reject_tiny_noise_box(self, validator):
        tiny_track = TrackedPerson(
            track_id=2,
            bbox=[100.0, 100.0, 103.0, 104.0],  # area = 12 px^2 < 20 px^2
            confidence=0.75,
            hits=5,
            state="active",
        )
        is_valid, reason = validator.validate_track(tiny_track)
        assert is_valid is False
        assert "min_box_area" in reason

    def test_reject_huge_box(self, validator):
        huge_track = TrackedPerson(
            track_id=3,
            bbox=[0.0, 0.0, 500.0, 500.0],  # area = 250,000 px^2 > 100,000
            confidence=0.80,
            hits=5,
            state="active",
        )
        is_valid, reason = validator.validate_track(huge_track)
        assert is_valid is False
        assert "max_box_area" in reason

    def test_reject_extreme_aspect_ratios(self, validator):
        # Too wide (e.g. railing or curb: w=150, h=20 -> w/h = 7.5 > 3.0)
        wide_track = TrackedPerson(
            track_id=4,
            bbox=[10.0, 10.0, 160.0, 30.0],
            confidence=0.70,
            hits=4,
            state="active",
        )
        is_valid_wide, reason_wide = validator.validate_track(wide_track)
        assert is_valid_wide is False
        assert "max_aspect_ratio" in reason_wide

        # Too thin/tall (e.g. vertical pole: w=5, h=100 -> w/h = 0.05 < 0.15)
        tall_track = TrackedPerson(
            track_id=5,
            bbox=[10.0, 10.0, 15.0, 110.0],
            confidence=0.70,
            hits=4,
            state="active",
        )
        is_valid_tall, reason_tall = validator.validate_track(tall_track)
        assert is_valid_tall is False
        assert "min_aspect_ratio" in reason_tall

    def test_temporal_confirmation_min_hits(self, validator):
        # 1 hit: unconfirmed candidate
        track_1hit = TrackedPerson(
            track_id=6,
            bbox=[50.0, 50.0, 90.0, 130.0],
            confidence=0.90,
            hits=1,
            state="active",
        )
        is_valid1, reason1 = validator.validate_track(track_1hit)
        assert is_valid1 is False
        assert "insufficient_hits" in reason1

        # 3 hits: confirmed
        track_3hit = TrackedPerson(
            track_id=6,
            bbox=[50.0, 50.0, 90.0, 130.0],
            confidence=0.90,
            hits=3,
            state="active",
        )
        is_valid3, reason3 = validator.validate_track(track_3hit)
        assert is_valid3 is True
        assert reason3 == "validated"

    def test_motion_consistency_outlier_gating(self, validator):
        # Sudden 200px jump in one frame (max allowed = 100px)
        teleport_track = TrackedPerson(
            track_id=7,
            bbox=[300.0, 300.0, 340.0, 380.0],
            confidence=0.88,
            hits=5,
            centroid_history=[[100.0, 100.0], [320.0, 340.0]],  # displacement = sqrt(220^2 + 240^2) > 300px
            state="active",
        )
        is_valid, reason = validator.validate_track(teleport_track)
        assert is_valid is False
        assert "velocity_jump_outlier" in reason

    def test_validate_tracks_batch_report(self, validator):
        t1 = TrackedPerson(track_id=1, bbox=[50.0, 50.0, 90.0, 130.0], confidence=0.90, hits=4, state="active")
        t2 = TrackedPerson(track_id=2, bbox=[10.0, 10.0, 12.0, 13.0], confidence=0.70, hits=4, state="active")  # tiny
        t3 = TrackedPerson(track_id=3, bbox=[20.0, 20.0, 60.0, 100.0], confidence=0.90, hits=1, state="active")  # 1 hit
        t4 = TrackedPerson(track_id=4, bbox=[30.0, 30.0, 70.0, 110.0], confidence=0.85, hits=5, state="lost")    # inactive

        report: ValidationReport = validator.validate_tracks([t1, t2, t3, t4])
        assert report.total_input == 3  # t4 was inactive
        assert report.validated_count == 1
        assert report.rejected_count == 2
        assert len(report.validated_tracks) == 1
        assert report.validated_tracks[0].track_id == 1
        assert 2 in report.validation_reasons
        assert 3 in report.validation_reasons
        assert "min_box_area" in report.rejection_breakdown
        assert "insufficient_hits" in report.rejection_breakdown

    def test_validation_disabled_allows_all(self):
        val_off = SecondStageValidator(ValidationConfig(enabled=False))
        tiny_track = TrackedPerson(track_id=1, bbox=[1.0, 1.0, 2.0, 2.0], confidence=0.1, hits=1, state="active")
        is_valid, reason = val_off.validate_track(tiny_track)
        assert is_valid is True
        assert reason == "validation_disabled"


class TestLightweightFallbackTracker:
    def test_fallback_tracker_lifecycle(self):
        config = TrackerConfig(track_thresh=0.5, track_buffer=5, match_thresh=0.5)
        tracker = LightweightFallbackTracker(config)
        assert tracker.initialize() is True

        # Frame 1: Single detection
        dets1 = np.array([[100.0, 100.0, 150.0, 200.0]], dtype=np.float32)
        scores1 = np.array([0.85], dtype=np.float32)
        active1 = tracker.update(dets1, scores1)
        assert len(active1) == 1
        t_id = active1[0].id
        assert active1[0].hits == 1

        # Frame 2: Slight movement
        dets2 = np.array([[105.0, 105.0, 155.0, 205.0]], dtype=np.float32)
        scores2 = np.array([0.88], dtype=np.float32)
        active2 = tracker.update(dets2, scores2)
        assert len(active2) == 1
        assert active2[0].id == t_id
        assert active2[0].hits == 2

    def test_fallback_tracker_buffer_and_removal(self):
        config = TrackerConfig(track_thresh=0.5, track_buffer=2, match_thresh=0.5)
        tracker = LightweightFallbackTracker(config)

        dets = np.array([[100.0, 100.0, 150.0, 200.0]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        tracker.update(dets, scores)

        # 3 empty frames
        for _ in range(3):
            tracker.update(np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32))

        assert len(tracker.tracked_tracks) == 0
        assert len(tracker.removed_tracks) >= 1


class TestTrackerWithValidation:
    def test_tracker_end_to_end_validation(self):
        cfg = TrackerConfig(
            min_hits=3,
            min_box_area=15.0,
            max_aspect_ratio=3.5,
            track_buffer=10,
        )
        tracker = Tracker(cfg)
        tracker.initialize()

        # Frame 1: Detection starts candidate track
        det_result1 = DetectionResult(
            detections=[
                Detection(bbox=[100.0, 100.0, 140.0, 180.0], confidence=0.85, class_id=0, class_name="person"),
            ],
            frame_number=1,
            timestamp=0.033,
        )
        res1: TrackResult = tracker.update(det_result1)
        assert len(res1.tracks) == 1
        assert res1.tracks[0].hits == 1
        # Hits (1) < min_hits (3) -> Not yet validated
        assert res1.tracks[0].is_validated is False
        assert len(res1.validated_tracks) == 0
        assert res1.rejected_count == 1

        # Frame 2: Hit 2
        det_result2 = DetectionResult(
            detections=[
                Detection(bbox=[102.0, 102.0, 142.0, 182.0], confidence=0.87, class_id=0, class_name="person"),
            ],
            frame_number=2,
            timestamp=0.066,
        )
        res2: TrackResult = tracker.update(det_result2)
        assert len(res2.tracks) == 1
        assert res2.tracks[0].hits == 2
        assert res2.tracks[0].is_validated is False

        # Frame 3: Hit 3 -> Meets min_hits!
        det_result3 = DetectionResult(
            detections=[
                Detection(bbox=[104.0, 104.0, 144.0, 184.0], confidence=0.89, class_id=0, class_name="person"),
            ],
            frame_number=3,
            timestamp=0.099,
        )
        res3: TrackResult = tracker.update(det_result3)
        assert len(res3.tracks) == 1
        assert res3.tracks[0].hits == 3
        # Validated!
        assert res3.tracks[0].is_validated is True
        assert len(res3.validated_tracks) == 1
        assert res3.num_validated_tracks == 1
        assert res3.rejected_count == 0

    def test_get_validated_tracks_convenience(self):
        cfg = TrackerConfig(min_hits=1)  # instant validation
        tracker = Tracker(cfg)
        tracker.initialize()

        det_result = DetectionResult(
            detections=[
                Detection(bbox=[100.0, 100.0, 140.0, 180.0], confidence=0.85, class_id=0, class_name="person"),
            ],
            frame_number=1,
        )
        validated = tracker.get_validated_tracks(det_result)
        assert len(validated) == 1
        assert validated[0].is_validated is True

