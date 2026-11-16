import time
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2
from loguru import logger

from app.detection.detector import BaseDetector, DetectionError, ModelLoadError, InferenceError, DetectionContext
from app.detection.models import Detection, DetectionResult, DetectorConfig, ModelInfo


class YOLODetector(BaseDetector):
    def __init__(self, config: DetectorConfig):
        super().__init__(config)
        self._model = None
        self._class_names = {}

    def load_model(self) -> bool:
        try:
            from ultralytics import YOLO

            model_path = Path(self.config.model_path)
            if not model_path.is_absolute():
                from app.config import settings
                model_path = settings.models_dir / model_path.name

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                logger.info("Download model weights from: https://github.com/ultralytics/assets/releases")
                return False

            logger.info(f"Loading YOLO model from: {model_path}")
            self._model = YOLO(str(model_path))

            self._model.to(self.config.device)
            if self.config.half_precision and self.config.device != "cpu":
                self._model.model.half()

            self._class_names = self._model.names
            logger.info(f"Model classes: {self._class_names}")

            self._model_info = ModelInfo(
                name=model_path.stem,
                path=str(model_path),
                input_size=(self.config.input_width, self.config.input_height),
                num_classes=len(self._class_names),
                class_names=self._class_names,
                device=self.config.device,
                half_precision=self.config.half_precision,
            )

            self._is_loaded = True
            logger.info(f"YOLO model loaded successfully on {self.config.device}")
            return True

        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            raise ModelLoadError("Ultralytics package not found")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")

    def detect(self, frame: np.ndarray, context: DetectionContext) -> DetectionResult:
        if not self._is_loaded or self._model is None:
            raise InferenceError("Model not loaded. Call load_model() first.")

        if not self.validate_input(frame):
            raise InferenceError("Invalid input frame")

        start_time = time.perf_counter()

        try:
            results = self._model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                verbose=False,
                device=self.config.device,
                half=self.config.half_precision and self.config.device != "cpu",
            )

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            detections = self._parse_results(results[0], frame.shape)

            return DetectionResult(
                detections=detections,
                frame_number=context.frame_number,
                timestamp=context.timestamp,
                processing_time_ms=processing_time_ms,
                model_name=self._model_info.name if self._model_info else "unknown",
                image_width=frame.shape[1],
                image_height=frame.shape[0],
            )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise InferenceError(f"Detection failed: {e}")

    def _parse_results(self, result, frame_shape: tuple) -> List[Detection]:
        detections = []

        if result.boxes is None:
            return detections

        # Check if boxes has any detections - handle both real Ultralytics objects and mocks
        try:
            num_boxes = len(result.boxes)
        except (TypeError, AttributeError):
            num_boxes = 0
        
        if num_boxes == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            if cls_id != self.config.person_class_id:
                continue

            if conf < self.config.confidence_threshold:
                continue

            class_name = self._class_names.get(cls_id, f"class_{cls_id}")

            detection = Detection(
                bbox=box.tolist(),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=class_name,
                track_id=None,
            )
            detections.append(detection)

        return detections

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            self._model_info = None
            logger.info("YOLO model unloaded")

    def get_model_info(self) -> Optional[ModelInfo]:
        return self._model_info


def create_detector(config: Optional[DetectorConfig] = None) -> YOLODetector:
    if config is None:
        config = DetectorConfig()
    return YOLODetector(config)