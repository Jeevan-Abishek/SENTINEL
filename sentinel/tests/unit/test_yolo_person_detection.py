"""
Unit tests for SENTINEL YOLO Person Detection Module.

Validates:
1. Typed Detection models (Detection, DetectionResult, FrameDetections) with center_xy, foot_xy, area, bbox.
2. YOLODetector & YOLOPersonDetector using Ultralytics YOLOv8n weights on CPU.
3. Person-only filtering (classes=[0]) and confidence threshold compliance.
4. Clean bounding box coordinate remapping back to original unscaled/letterboxed frame dimensions.
5. Graceful handling of missing weights, empty frames, and corrupt inputs.
6. High-contrast bounding box and centroid/foot point visualization without heavy CPU overhead.
"""

import sys
import time
import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock

# Add project root to sys.path
_sentinel_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_sentinel_root))

from app.detection.models import (
    Detection,
    DetectionResult,
    FrameDetections,
    DetectorConfig,
    ModelInfo,
)
from app.detection.detector import (
    DetectionContext,
    ModelLoadError,
    InferenceError,
)
from app.detection.yolo_detector import (
    YOLODetector,
    YOLOPersonDetector,
    create_detector,
)
from app.detection.visualization import (
    draw_detection,
    draw_detections,
    draw_detection_result,
    draw_count_overlay,
    draw_fps_overlay,
    ColorPalette,
    PALETTE,
)
from app.config import settings


class TestDetectionModels:
    """Tests for typed detection models."""

    def test_detection_properties(self):
        det = Detection(
            bbox=[100.0, 200.0, 300.0, 500.0],
            confidence=0.88,
            class_id=0,
            class_name="person",
            track_id=7,
        )
        assert det.x1 == 100.0
        assert det.y1 == 200.0
        assert det.x2 == 300.0
        assert det.y2 == 500.0
        assert det.width == 200.0
        assert det.height == 300.0
        assert det.area == 60000.0
        assert det.center_x == 200.0
        assert det.center_y == 350.0
        assert det.center_xy == (200.0, 350.0)
        assert det.foot_xy == (200.0, 500.0)
        assert det.to_xywh() == [100.0, 200.0, 200.0, 300.0]
        assert det.to_xyxy() == [100.0, 200.0, 300.0, 500.0]

    def test_frame_detections_alias_and_filters(self):
        d1 = Detection(bbox=[10, 20, 30, 40], confidence=0.9, class_id=0)
        d2 = Detection(bbox=[50, 60, 70, 80], confidence=0.4, class_id=0)
        d3 = Detection(bbox=[90, 100, 110, 120], confidence=0.75, class_id=1, class_name="bicycle")

        res = FrameDetections(
            detections=[d1, d2, d3],
            frame_number=10,
            timestamp=0.33,
            orig_width=1920,
            orig_height=1080,
            scale_x=0.333,
            scale_y=0.333,
        )
        assert res.count == 3
        assert res.person_count == 2
        assert res.orig_width == 1920

        # Filter by confidence
        high_conf = res.filter_by_confidence(0.7)
        assert high_conf.count == 2
        assert high_conf.detections[0].confidence >= 0.7

        # Filter by class
        persons_only = res.filter_by_class(0)
        assert persons_only.count == 2


class TestYOLOPersonDetectorInference:
    """Tests for YOLOPersonDetector using real YOLOv8 weights on CPU."""

    @pytest.fixture
    def detector(self):
        cfg = DetectorConfig(
            model_path=str(settings.resolved_yolo_model_path),
            confidence_threshold=0.35,
            iou_threshold=0.45,
            device="cpu",
            classes=[0],
        )
        det = YOLOPersonDetector(cfg)
        loaded = det.load_model()
        assert loaded is True, "Failed to load YOLO model weights"
        return det

    def test_yolo_person_detector_alias(self):
        assert YOLOPersonDetector is YOLODetector

    def test_model_loading_and_metadata(self, detector):
        assert detector.is_loaded is True
        info = detector.model_info
        assert info is not None
        assert info.device == "cpu"
        assert info.name.startswith("yolo")

    def test_missing_model_file_fails_gracefully(self):
        cfg = DetectorConfig(model_path="models/non_existent_weights_abc123.pt")
        det = YOLODetector(cfg)
        assert det.load_model() is False
        assert det.is_loaded is False

    def test_detect_empty_synthetic_frame(self, detector):
        frame = np.full((360, 640, 3), 128, dtype=np.uint8)
        ctx = DetectionContext(frame_number=1, timestamp=0.033, original_frame_shape=(360, 640))
        result = detector.detect(frame, ctx)

        assert isinstance(result, DetectionResult)
        assert result.count == 0
        assert result.image_width == 640
        assert result.image_height == 360
        assert result.processing_time_ms >= 0

    def test_invalid_input_frame_raises_error(self, detector):
        with pytest.raises(InferenceError):
            detector.detect(None)
        with pytest.raises(InferenceError):
            detector.detect(np.zeros((100, 100), dtype=np.uint8))  # 2D instead of 3D

    def test_real_video_frame_person_detection(self, detector):
        video_path = str(settings.sample_videos_dir / "demo_drone_crowd.mp4")
        if not Path(video_path).exists():
            pytest.skip("Demo drone video not found.")

        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened()
        ret, frame = cap.read()
        cap.release()
        assert ret and frame is not None

        # Resize to 640px width for fast CPU inference
        h, w = frame.shape[:2]
        scale = 640.0 / float(w)
        frame_resized = cv2.resize(frame, (640, int(h * scale)))

        ctx = DetectionContext(
            frame_number=1,
            timestamp=0.0,
            original_frame_shape=(h, w),
            scale_x=scale,
            scale_y=scale,
        )

        result = detector.detect(frame_resized, ctx)
        assert isinstance(result, DetectionResult)
        assert result.count > 0, "Expected person detections in demo drone footage"
        assert result.person_count == result.count

        # Verify all detections are person (class_id=0) with valid bounding boxes
        for det in result.detections:
            assert det.class_id == 0
            assert det.class_name == "person"
            assert det.confidence >= 0.35
            # Bounding box should be in original frame coordinates (w, h)
            assert 0 <= det.x1 < det.x2 <= w + 1
            assert 0 <= det.y1 < det.y2 <= h + 1


class TestBoundingBoxRemapping:
    """Tests for automatic bounding box remapping from resized/letterboxed frames back to original dimensions."""

    def test_box_remapping_from_half_resolution(self):
        cfg = DetectorConfig()
        det = YOLODetector(cfg)
        det._is_loaded = True
        det._class_names = {0: "person"}

        # Mock YOLO result with box at [50, 100, 150, 200] on a 640x360 frame
        mock_boxes = Mock()
        mock_boxes.__len__ = Mock(return_value=1)
        mock_boxes.xyxy = Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([[50.0, 100.0, 150.0, 200.0]])))))
        mock_boxes.conf = Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([0.85])))))
        mock_boxes.cls = Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([0])))))

        mock_result = Mock()
        mock_result.boxes = mock_boxes

        # Context specifies that original frame was 1920x1080 (downscaled by scale=0.5, pad=0)
        ctx = DetectionContext(
            original_frame_shape=(1080, 1920),
            scale_x=0.5,
            scale_y=0.5,
            pad_left=0,
            pad_top=0,
        )

        detections = det._parse_results(mock_result, (360, 640), ctx)
        assert len(detections) == 1
        d = detections[0]
        # In original 1920x1080: coordinates should be 2x [100, 200, 300, 400]
        assert abs(d.x1 - 100.0) < 1e-3
        assert abs(d.y1 - 200.0) < 1e-3
        assert abs(d.x2 - 300.0) < 1e-3
        assert abs(d.y2 - 400.0) < 1e-3


class TestVisualizationOverlays:
    """Tests for high-contrast bounding box and centroid/foot point visualization."""

    def test_draw_detection_with_center_and_foot(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = Detection(bbox=[100, 100, 200, 200], confidence=0.92, class_id=0, class_name="person")

        # Warm up OpenCV text/drawing engine
        _ = draw_detection(frame.copy(), det)

        t0 = time.perf_counter()
        annotated = draw_detection(
            frame,
            det,
            color=(46, 125, 50),
            show_center=True,
            show_foot=True,
            thickness=2,
        )
        dt_us = (time.perf_counter() - t0) * 1_000_000.0

        assert annotated is not None
        assert dt_us < 10000.0  # Ultra-fast CPU drawing (< 10ms)
        # Verify pixels were written to the frame
        assert np.any(annotated > 0)

    def test_draw_detection_result_multiple(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = [
            Detection(bbox=[10, 10, 50, 50], confidence=0.8),
            Detection(bbox=[60, 60, 100, 100], confidence=0.85),
            Detection(bbox=[110, 110, 150, 150], confidence=0.9),
        ]
        res = DetectionResult(detections=dets)
        annotated = draw_detection_result(frame, res, show_foot=True)
        assert annotated.shape == frame.shape

    def test_overlays(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with_count = draw_count_overlay(frame, 42)
        assert with_count is not None
        with_fps = draw_fps_overlay(with_count, 29.97)
        assert with_fps is not None
