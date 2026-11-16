"""
Unit tests for SENTINEL Detection Module
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.detection.models import Detection, DetectionResult, DetectorConfig, ModelInfo
from app.detection.detector import BaseDetector, DetectionContext, DetectionError, ModelLoadError, InferenceError
from app.detection.visualization import (
    draw_detection, draw_detections, draw_count_overlay, 
    draw_fps_overlay, draw_processing_info, create_detection_summary_image,
    ColorPalette, PALETTE
)


class TestDetectionModel:
    def test_detection_creation(self):
        det = Detection(bbox=[100, 100, 200, 200], confidence=0.9, class_id=0, class_name="person")
        assert det.x1 == 100
        assert det.y1 == 100
        assert det.x2 == 200
        assert det.y2 == 200
        assert det.width == 100
        assert det.height == 100
        assert det.center_x == 150
        assert det.center_y == 150
        assert det.area == 10000

    def test_detection_to_xywh(self):
        det = Detection(bbox=[100, 100, 200, 200], confidence=0.9)
        xywh = det.to_xywh()
        assert xywh == [100, 100, 100, 100]

    def test_detection_to_xyxy(self):
        det = Detection(bbox=[100, 100, 200, 200], confidence=0.9)
        xyxy = det.to_xyxy()
        assert xyxy == [100, 100, 200, 200]


class TestDetectionResult:
    def test_detection_result_creation(self):
        detections = [
            Detection(bbox=[100, 100, 200, 200], confidence=0.9),
            Detection(bbox=[300, 300, 400, 400], confidence=0.8),
        ]
        result = DetectionResult(detections=detections, frame_number=1, timestamp=0.033)
        assert result.count == 2
        assert result.person_count == 2

    def test_detection_result_filter_by_confidence(self):
        detections = [
            Detection(bbox=[100, 100, 200, 200], confidence=0.9),
            Detection(bbox=[300, 300, 400, 400], confidence=0.3),
            Detection(bbox=[500, 500, 600, 600], confidence=0.7),
        ]
        result = DetectionResult(detections=detections, frame_number=1)
        filtered = result.filter_by_confidence(0.5)
        assert filtered.count == 2
        assert all(d.confidence >= 0.5 for d in filtered.detections)

    def test_detection_result_filter_by_class(self):
        detections = [
            Detection(bbox=[100, 100, 200, 200], confidence=0.9, class_id=0),
            Detection(bbox=[300, 300, 400, 400], confidence=0.8, class_id=1),
            Detection(bbox=[500, 500, 600, 600], confidence=0.7, class_id=0),
        ]
        result = DetectionResult(detections=detections, frame_number=1)
        filtered = result.filter_by_class(0)
        assert filtered.count == 2
        assert all(d.class_id == 0 for d in filtered.detections)


class TestDetectorConfig:
    def test_default_config(self):
        config = DetectorConfig()
        assert config.model_path == "models/yolov8n.pt"
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.device == "cpu"
        assert config.person_class_id == 0

    def test_custom_config(self):
        config = DetectorConfig(
            model_path="custom/yolo.pt",
            confidence_threshold=0.7,
            device="cuda",
        )
        assert config.model_path == "custom/yolo.pt"
        assert config.confidence_threshold == 0.7
        assert config.device == "cuda"


class TestModelInfo:
    def test_model_info_creation(self):
        info = ModelInfo(
            name="yolov8n",
            path="/models/yolov8n.pt",
            input_size=(640, 640),
            num_classes=80,
            class_names={0: "person"},
            device="cpu",
        )
        assert info.name == "yolov8n"
        assert info.num_classes == 80


class TestDetectionContext:
    def test_context_creation(self):
        ctx = DetectionContext(frame_number=100, timestamp=3.33, original_frame_shape=(720, 1280))
        assert ctx.frame_number == 100
        assert ctx.timestamp == 3.33
        assert ctx.original_frame_shape == (720, 1280)


class TestDetectionErrors:
    def test_detection_error_inheritance(self):
        assert issubclass(ModelLoadError, DetectionError)
        assert issubclass(InferenceError, DetectionError)

    def test_error_messages(self):
        with pytest.raises(ModelLoadError, match="Model loading failed"):
            raise ModelLoadError("Model loading failed")


class MockDetector(BaseDetector):
    def load_model(self) -> bool:
        return True

    def detect(self, frame: "np.ndarray", context: DetectionContext) -> DetectionResult:
        return DetectionResult()

    def unload_model(self) -> None:
        pass


class TestBaseDetector:
    def test_base_detector_initialization(self):
        config = DetectorConfig()
        detector = MockDetector(config)
        assert detector.config == config
        assert detector.is_loaded is False
        assert detector.model_info is None

    def test_validate_input_valid_frame(self):
        config = DetectorConfig()
        detector = MockDetector(config)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        assert detector.validate_input(frame) is True

    def test_validate_input_none(self):
        config = DetectorConfig()
        detector = MockDetector(config)
        assert detector.validate_input(None) is False

    def test_validate_input_wrong_dimensions(self):
        config = DetectorConfig()
        detector = MockDetector(config)
        frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        assert detector.validate_input(frame) is False

    def test_validate_input_wrong_channels(self):
        config = DetectorConfig()
        detector = MockDetector(config)
        frame = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        assert detector.validate_input(frame) is False


class TestColorPalette:
    def test_get_color_consistency(self):
        color1 = PALETTE.get_color(1)
        color2 = PALETTE.get_color(1)
        assert color1 == color2

    def test_get_color_different_ids(self):
        color1 = PALETTE.get_color(1)
        color2 = PALETTE.get_color(2)
        assert color1 != color2

    def test_get_class_color(self):
        color = PALETTE.get_class_color(0)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)


class TestVisualization:
    def test_draw_detection(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)  # Gray frame
        det = Detection(bbox=[100, 100, 200, 200], confidence=0.9)
        annotated = draw_detection(frame.copy(), det)
        assert annotated.shape == frame.shape
        assert not np.array_equal(annotated, frame)

    def test_draw_detections_multiple(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        detections = [
            Detection(bbox=[100, 100, 200, 200], confidence=0.9),
            Detection(bbox=[300, 300, 400, 400], confidence=0.8),
        ]
        annotated = draw_detections(frame.copy(), detections)
        assert annotated.shape == frame.shape

    def test_draw_count_overlay(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        annotated = draw_count_overlay(frame.copy(), 42)
        assert annotated.shape == frame.shape
        assert not np.array_equal(annotated, frame)

    def test_draw_fps_overlay(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        annotated = draw_fps_overlay(frame.copy(), 29.5)
        assert annotated.shape == frame.shape

    def test_draw_processing_info(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = DetectionResult(
            detections=[Detection(bbox=[100, 100, 200, 200], confidence=0.9)],
            frame_number=1,
            timestamp=0.033,
            processing_time_ms=45.2,
            model_name="yolov8n",
        )
        annotated = draw_processing_info(frame.copy(), result)
        assert annotated.shape == frame.shape

    def test_create_detection_summary_image(self):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = DetectionResult(
            detections=[Detection(bbox=[100, 100, 200, 200], confidence=0.9)],
            frame_number=1,
            timestamp=0.033,
            processing_time_ms=45.2,
            model_name="yolov8n",
        )
        annotated = create_detection_summary_image(frame.copy(), result, show_fps=29.5)
        assert annotated.shape == frame.shape


class TestYOLODetectorMock:
    @patch("ultralytics.YOLO")
    def test_yolo_detector_load_model_success(self, mock_yolo_class):
        from app.detection.yolo_detector import YOLODetector

        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "bicycle"}
        mock_yolo_class.return_value = mock_model

        config = DetectorConfig(model_path="models/yolov8n.pt")
        detector = YOLODetector(config)

        with patch.object(Path, "exists", return_value=True):
            result = detector.load_model()

        assert result is True
        assert detector.is_loaded is True
        assert detector.model_info is not None
        assert detector.model_info.name == "yolov8n"

    @patch("ultralytics.YOLO")
    def test_yolo_detector_load_model_missing_file(self, mock_yolo_class):
        from app.detection.yolo_detector import YOLODetector

        config = DetectorConfig(model_path="models/missing.pt")
        detector = YOLODetector(config)

        with patch.object(Path, "exists", return_value=False):
            result = detector.load_model()

        assert result is False
        assert detector.is_loaded is False

    @patch("ultralytics.YOLO")
    def test_yolo_detector_detect(self, mock_yolo_class):
        from app.detection.yolo_detector import YOLODetector

        mock_model = MagicMock()
        mock_model.names = {0: "person"}

        mock_result = MagicMock()
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 2
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        mock_boxes.conf.cpu().numpy.return_value = np.array([0.9, 0.8])
        mock_boxes.cls.cpu().numpy.return_value = np.array([0, 0])
        mock_result.boxes = mock_boxes

        mock_model.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        config = DetectorConfig()
        detector = YOLODetector(config)
        detector._model = mock_model
        detector._is_loaded = True
        detector._class_names = {0: "person"}

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        context = DetectionContext(frame_number=1, timestamp=0.033, original_frame_shape=(480, 640))

        result = detector.detect(frame, context)

        assert isinstance(result, DetectionResult)
        assert result.count == 2
        assert result.frame_number == 1
        assert result.timestamp == 0.033
        assert result.processing_time_ms > 0

    @patch("ultralytics.YOLO")
    def test_yolo_detector_detect_empty(self, mock_yolo_class):
        from app.detection.yolo_detector import YOLODetector

        mock_model = MagicMock()
        mock_model.names = {0: "person"}

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_yolo_class.return_value = mock_model

        config = DetectorConfig()
        detector = YOLODetector(config)
        detector._model = mock_model
        detector._is_loaded = True
        detector._class_names = {0: "person"}

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        context = DetectionContext(frame_number=1, timestamp=0.033, original_frame_shape=(480, 640))

        result = detector.detect(frame, context)

        assert result.count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])