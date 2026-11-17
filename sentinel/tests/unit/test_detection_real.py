"""
Unit tests for YOLO Person Detection with real weights (Phase 2)
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import time

from app.detection.yolo_detector import YOLODetector
from app.detection.models import DetectorConfig, DetectionResult
from app.detection.detector import DetectionContext, ModelLoadError


class TestYOLODetectionRealWeights:
    """Tests verifying real YOLO model weights, device selection, and detection inference."""

    @pytest.fixture
    def detector(self):
        config = DetectorConfig(
            model_path="data/models/yolov8n.pt",
            confidence_threshold=0.3,
            iou_threshold=0.45,
            device="cpu",
            classes=[0],  # Person class only
        )
        det = YOLODetector(config)
        det.load_model()
        return det

    def test_model_loading_and_metadata(self, detector):
        assert detector.is_loaded is True
        info = detector.model_info
        assert info is not None
        assert info.name == "yolov8n"
        assert info.device == "cpu"
        assert 0 in info.class_names
        assert info.class_names[0] == "person"

    def test_missing_model_handling(self):
        bad_config = DetectorConfig(model_path="models/non_existent_weights_xyz.pt")
        bad_detector = YOLODetector(bad_config)
        # Should return False gracefully and not crash
        assert bad_detector.load_model() is False
        assert bad_detector.is_loaded is False

    def test_detection_on_empty_frame(self, detector):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        ctx = DetectionContext(frame_number=1, timestamp=0.0, original_frame_shape=(480, 640))
        result = detector.detect(frame, ctx)

        assert isinstance(result, DetectionResult)
        assert result.frame_number == 1
        assert result.image_width == 640
        assert result.image_height == 480
        assert result.processing_time_ms >= 0
        assert len(result.detections) == 0

    def test_person_class_filtering_and_confidence(self, detector):
        # Create an image with drawing that simulates a person
        frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        # Draw a synthetic humanoid figure
        cv2.circle(frame, (320, 180), 30, (30, 30, 30), -1)  # head
        cv2.rectangle(frame, (280, 210), (360, 340), (40, 40, 180), -1)  # body
        cv2.rectangle(frame, (290, 340), (320, 440), (20, 20, 20), -1)  # left leg
        cv2.rectangle(frame, (330, 340), (360, 440), (20, 20, 20), -1)  # right leg

        ctx = DetectionContext(frame_number=2, timestamp=0.066, original_frame_shape=(480, 640))
        result = detector.detect(frame, ctx)

        # For any detected objects, they must strictly be the person class (0)
        for det in result.detections:
            assert det.class_id == 0
            assert det.class_name == "person"
            assert det.confidence >= 0.3

    def test_detection_performance_measurement(self, detector):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        ctx = DetectionContext(frame_number=3, timestamp=0.1, original_frame_shape=(480, 640))
        t0 = time.perf_counter()
        result = detector.detect(frame, ctx)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert result.processing_time_ms > 0
        assert result.processing_time_ms <= elapsed_ms + 10  # within margin

    def test_detect_without_context_argument(self, detector):
        """Verify detector.detect(frame) works smoothly without context positional argument."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = detector.detect(frame)
        assert isinstance(result, DetectionResult)
        assert result.frame_number == 0
        assert result.image_width == 640
        assert result.image_height == 480
        assert result.processing_time_ms >= 0


