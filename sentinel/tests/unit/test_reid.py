"""
Unit tests for SENTINEL Appearance Re-Identification (Re-ID) Module
Verifies:
  1. Fast CPU appearance descriptor extraction (multi-stripe HSV+Lab+gradient histograms).
  2. Bounding box cropping, clamping, and invalid input safety.
  3. L2 normalization and cosine similarity matching.
  4. Gallery TTL expiration and capacity pruning.
  5. End-to-end identity stitching across temporary occlusions in Tracker.
"""

import pytest
import numpy as np
from pathlib import Path
import cv2
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.tracking.reid import (
    AppearanceReID,
    ReIDConfig,
    ReIDMatch,
    GalleryEntry,
)
from app.tracking.models import (
    Track,
    TrackedPerson,
    TrackResult,
    TrackerConfig,
    TrackingContext,
)
from app.tracking.tracker import Tracker
from app.detection.models import Detection, DetectionResult


def create_synthetic_person_crop(top_color_bgr, bottom_color_bgr, width=40, height=90):
    """Generates synthetic person crop with distinct top and bottom color blocks."""
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    half = height // 2
    crop[:half, :] = top_color_bgr
    crop[half:, :] = bottom_color_bgr
    return crop


class TestAppearanceReIDExtraction:
    @pytest.fixture
    def reid_engine(self):
        return AppearanceReID(ReIDConfig(enabled=True))

    def test_extract_signature_shape_and_l2_norm(self, reid_engine):
        crop = create_synthetic_person_crop([0, 0, 255], [255, 0, 0])  # Red top, Blue bottom
        sig = reid_engine.extract_signature(crop)
        assert sig is not None
        assert isinstance(sig, np.ndarray)
        assert sig.dtype == np.float32
        # 3 stripes * (16 + 8 + 8 + 8 + 8 + 2) = 3 * 50 = 150 dimensions
        assert sig.ndim == 1
        assert len(sig) > 60
        # L2 norm must be approximately 1.0
        norm = np.linalg.norm(sig)
        assert norm == pytest.approx(1.0, abs=1e-4)

    def test_extract_from_frame_bounding_box(self, reid_engine):
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # Put synthetic person in middle
        frame[100:200, 150:200] = create_synthetic_person_crop([0, 200, 0], [20, 20, 20], width=50, height=100)
        bbox = [150.0, 100.0, 200.0, 200.0]

        sig = reid_engine.extract_from_frame(frame, bbox)
        assert sig is not None
        assert np.linalg.norm(sig) == pytest.approx(1.0, abs=1e-4)

    def test_invalid_crops_and_out_of_bounds_handling(self, reid_engine):
        # Empty/None
        assert reid_engine.extract_signature(None) is None
        assert reid_engine.extract_signature(np.empty((0, 0, 3), dtype=np.uint8)) is None

        # Too small (< 10x10)
        tiny = np.zeros((5, 5, 3), dtype=np.uint8)
        assert reid_engine.extract_signature(tiny) is None

        # Clamping out-of-bounds bounding box
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox_out = [-50.0, -50.0, 200.0, 200.0]
        # Should clamp safely to (0, 0, 100, 100) and return valid signature
        sig = reid_engine.extract_from_frame(frame, bbox_out)
        assert sig is not None


class TestAppearanceReIDMatching:
    @pytest.fixture
    def reid_engine(self):
        return AppearanceReID(
            ReIDConfig(
                enabled=True,
                similarity_threshold=0.65,
                ttl_frames=60,
                max_spatial_distance=300.0,
            )
        )

    def test_cosine_similarity_discriminates_identities(self, reid_engine):
        # Person A: Red jacket, Dark pants
        crop_a1 = create_synthetic_person_crop([0, 0, 220], [30, 30, 30])
        crop_a2 = create_synthetic_person_crop([0, 10, 210], [35, 35, 35])  # slight variation
        # Person B: Yellow shirt, White shorts
        crop_b = create_synthetic_person_crop([0, 240, 240], [240, 240, 240])

        sig_a1 = reid_engine.extract_signature(crop_a1)
        sig_a2 = reid_engine.extract_signature(crop_a2)
        sig_b = reid_engine.extract_signature(crop_b)

        sim_same = float(np.dot(sig_a1, sig_a2))
        sim_diff = float(np.dot(sig_a1, sig_b))

        # Same person must have high similarity (> 0.85)
        assert sim_same > 0.85
        # Different person must have much lower similarity (< 0.60)
        assert sim_diff < 0.60
        assert sim_same > sim_diff + 0.25

    def test_reid_match_candidate_restores_identity(self, reid_engine):
        crop = create_synthetic_person_crop([0, 150, 0], [10, 10, 100])
        sig = reid_engine.extract_signature(crop)
        bbox_lost = [100.0, 100.0, 140.0, 190.0]

        # 1. Register active track
        reid_engine.register_active_track(track_id=15, signature=sig, bbox=bbox_lost, frame_number=10, timestamp=0.33)
        assert 15 in reid_engine.active_gallery

        # 2. Track gets lost
        reid_engine.register_lost_track(track_id=15, bbox=bbox_lost, frame_number=12, timestamp=0.40)
        assert 15 not in reid_engine.active_gallery
        assert 15 in reid_engine.lost_gallery
        assert reid_engine.gallery_size == 1

        # 3. New candidate appears near last position in frame 18 (within TTL)
        candidate_bbox = [115.0, 110.0, 155.0, 200.0]  # distance ~ 18 px
        match: Optional[ReIDMatch] = reid_engine.match_candidate(
            candidate_sig=sig,
            candidate_bbox=candidate_bbox,
            current_frame=18,
            current_timestamp=0.60,
        )

        assert match is not None
        assert match.old_track_id == 15
        assert match.similarity >= 0.85
        assert reid_engine.total_matches == 1
        # Track 15 is restored back into active gallery
        assert 15 in reid_engine.active_gallery
        assert 15 not in reid_engine.lost_gallery

    def test_spatial_distance_gating(self, reid_engine):
        crop = create_synthetic_person_crop([100, 100, 100], [50, 50, 50])
        sig = reid_engine.extract_signature(crop)
        bbox_lost = [50.0, 50.0, 90.0, 140.0]

        reid_engine.register_active_track(1, sig, bbox_lost, 5, 0.16)
        reid_engine.register_lost_track(1, bbox_lost, 6, 0.20)

        # Candidate with identical signature appears 600 px away (> max_spatial_distance 300)
        far_bbox = [700.0, 700.0, 740.0, 790.0]
        match = reid_engine.match_candidate(sig, far_bbox, current_frame=10)
        assert match is None  # Gated by spatial radius


class TestGalleryTTLAndPruning:
    def test_prune_expired_entries(self):
        engine = AppearanceReID(ReIDConfig(enabled=True, ttl_frames=30))
        sig = np.ones(50, dtype=np.float32) / np.sqrt(50)

        engine.register_active_track(101, sig, [0, 0, 10, 10], frame_number=10, timestamp=0.33)
        engine.register_lost_track(101, [0, 0, 10, 10], frame_number=10, timestamp=0.33)

        assert engine.gallery_size == 1

        # Frame 35: delta = 25 <= 30 -> Still retained
        engine.prune_gallery(current_frame=35)
        assert engine.gallery_size == 1

        # Frame 45: delta = 35 > 30 -> Expired and pruned
        engine.prune_gallery(current_frame=45)
        assert engine.gallery_size == 0

    def test_max_gallery_size_enforcement(self):
        engine = AppearanceReID(ReIDConfig(enabled=True, max_gallery_size=3))
        sig = np.ones(50, dtype=np.float32) / np.sqrt(50)

        for i in range(5):
            engine.register_active_track(i, sig, [0, 0, 10, 10], frame_number=i, timestamp=0.1 * i)
            engine.register_lost_track(i, [0, 0, 10, 10], frame_number=i, timestamp=0.1 * i)

        # 5 registered, but max_gallery_size is 3
        engine.prune_gallery(current_frame=10)
        assert engine.gallery_size <= 3


class TestTrackerReIDIntegration:
    def test_tracker_end_to_end_reid_stitching(self):
        cfg = TrackerConfig(
            reid_enabled=True,
            reid_threshold=0.65,
            track_buffer=5,
            min_hits=1,  # immediate validation for test
        )
        tracker = Tracker(cfg)
        tracker.initialize()

        # Create synthetic frame with 1 person
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        person_crop = create_synthetic_person_crop([0, 0, 250], [200, 0, 0], width=30, height=70)
        frame[50:120, 50:80] = person_crop

        bbox1 = [50.0, 50.0, 80.0, 120.0]
        det1 = DetectionResult(
            detections=[Detection(bbox=bbox1, confidence=0.90, class_id=0, class_name="person")],
            frame_number=1,
            timestamp=0.033,
        )

        res1: TrackResult = tracker.update(det1, frame=frame)
        assert len(res1.tracks) == 1
        initial_id = res1.tracks[0].track_id
        assert res1.tracks[0].embedding is not None
        assert len(res1.tracks[0].embedding) > 0

        # Simulate track moving and updating gallery
        frame2 = frame.copy()
        frame2[52:122, 52:82] = person_crop
        bbox2 = [52.0, 52.0, 82.0, 122.0]
        det2 = DetectionResult(
            detections=[Detection(bbox=bbox2, confidence=0.90, class_id=0, class_name="person")],
            frame_number=2,
            timestamp=0.066,
        )
        res2 = tracker.update(det2, frame=frame2)
        assert res2.tracks[0].track_id == initial_id

        # Disappear for 2 frames (temporary occlusion)
        empty_det = DetectionResult(detections=[], frame_number=3, timestamp=0.099)
        tracker.update(empty_det)
        empty_det2 = DetectionResult(detections=[], frame_number=4, timestamp=0.132)
        tracker.update(empty_det2)

        # Verify track is transferred to lost gallery in Re-ID engine
        assert tracker.reid.gallery_size >= 1
        assert initial_id in tracker.reid.lost_gallery

        # Person re-emerges at frame 5 after occlusion with displacement (IoU = 0.0)
        frame5 = np.zeros((400, 400, 3), dtype=np.uint8)
        frame5[130:200, 130:160] = person_crop
        bbox5 = [130.0, 130.0, 160.0, 200.0]
        det5 = DetectionResult(
            detections=[Detection(bbox=bbox5, confidence=0.92, class_id=0, class_name="person")],
            frame_number=5,
            timestamp=0.165,
        )
        res5: TrackResult = tracker.update(det5, frame=frame5)

        assert len(res5.tracks) == 1
        # Re-ID matched and stitched back to initial_id!
        assert res5.tracks[0].track_id == initial_id
        assert res5.reid_matches >= 1
