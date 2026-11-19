"""
SENTINEL — Fast, Clean, and Lightweight YOLO Person Detection Module.

Strictly optimized for low-RAM Windows CPU environments (~145 MB footprint target).
Provides:
1. High-accuracy person detection using Ultralytics YOLOv8n CPU weights.
2. Person-only filtering (classes=[0]) to speed up CPU Non-Maximum Suppression (NMS).
3. Clean bounding box coordinate mapping back to original camera dimensions when
   preprocessing resized or letterboxed the frame.
4. Robust error handling for missing weights, empty frames, and inference timeouts.
"""

import time
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2
from loguru import logger

from app.detection.detector import (
    BaseDetector,
    DetectionError,
    ModelLoadError,
    InferenceError,
    DetectionContext,
)
from app.detection.models import (
    Detection,
    DetectionResult,
    FrameDetections,
    DetectorConfig,
    ModelInfo,
)
from app.config import settings


class YOLODetector(BaseDetector):
    """Ultralytics YOLO person detector optimized for real-time drone surveillance on CPU."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        if config is None:
            config = DetectorConfig(
                model_path=str(settings.resolved_yolo_model_path),
                confidence_threshold=settings.default_confidence_threshold,
                iou_threshold=settings.default_iou_threshold,
                device="cpu",
                half_precision=False,
                classes=[0],
            )
        super().__init__(config)
        self._model = None
        self._class_names = {}

    def load_model(self) -> bool:
        """Loads Ultralytics YOLO model with robust weight discovery and CPU optimization."""
        try:
            from ultralytics import YOLO

            model_path = Path(self.config.model_path)
            candidate_paths = [
                model_path,
                settings.models_dir / model_path.name,
                Path("data/models") / model_path.name,
                Path("sentinel/data/models") / model_path.name,
                settings.base_dir / "data" / "models" / model_path.name,
            ]
            resolved_path = None
            for cp in candidate_paths:
                if cp.exists() and cp.is_file():
                    resolved_path = cp
                    break

            if resolved_path is None:
                logger.error(f"Model file not found: {model_path}")
                logger.info("Download model weights from: https://github.com/ultralytics/assets/releases")
                return False

            logger.info(f"Loading YOLO model from: {resolved_path}")
            self._model = YOLO(str(resolved_path))

            # Strictly enforce CPU device to avoid Windows CUDA driver crashes
            target_device = self.config.device if self.config.device in ("cpu", "cuda", "mps") else "cpu"
            self._model.to(target_device)

            if self.config.half_precision and target_device != "cpu":
                self._model.model.half()

            self._class_names = self._model.names or {0: "person"}
            logger.info(f"YOLO model ready on {target_device} (classes: {len(self._class_names)})")

            self._model_info = ModelInfo(
                name=resolved_path.stem,
                path=str(resolved_path),
                input_size=(self.config.input_width, self.config.input_height),
                num_classes=len(self._class_names),
                class_names=self._class_names,
                device=target_device,
                half_precision=self.config.half_precision,
            )

            self._is_loaded = True
            return True

        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            raise ModelLoadError("Ultralytics package not found")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadError(f"Model loading failed: {e}")

    def detect(self, frame: np.ndarray, context: Optional[DetectionContext] = None) -> DetectionResult:
        """Executes YOLO inference with person-only filtering and box coordinate mapping."""
        if not self._is_loaded or self._model is None:
            raise InferenceError("Model not loaded. Call load_model() first.")

        if not self.validate_input(frame):
            raise InferenceError("Invalid input frame")

        if context is None:
            frame_shape = frame.shape[:2] if hasattr(frame, "shape") else (0, 0)
            context = DetectionContext(
                frame_number=0,
                timestamp=time.time(),
                original_frame_shape=frame_shape,
            )

        start_time = time.perf_counter()

        try:
            # Person-only class filter (classes=[0]) dramatically accelerates CPU NMS
            kwargs = {
                "conf": self.config.confidence_threshold,
                "iou": self.config.iou_threshold,
                "max_det": self.config.max_detections,
                "verbose": False,
                "device": self.config.device,
                "classes": [self.config.person_class_id],
            }
            if self.config.half_precision and self.config.device != "cpu":
                kwargs["half"] = True

            results = self._model(frame, **kwargs)
            processing_time_ms = (time.perf_counter() - start_time) * 1000.0

            detections = self._parse_results(results[0], frame.shape, context)

            orig_h, orig_w = context.original_frame_shape if (context.original_frame_shape and context.original_frame_shape != (0, 0)) else frame.shape[:2]

            return DetectionResult(
                detections=detections,
                frame_number=context.frame_number,
                timestamp=context.timestamp,
                processing_time_ms=processing_time_ms,
                model_name=self._model_info.name if self._model_info else "yolov8n",
                image_width=orig_w,
                image_height=orig_h,
                orig_width=orig_w,
                orig_height=orig_h,
                scale_x=context.scale_x,
                scale_y=context.scale_y,
                pad_left=context.pad_left,
                pad_top=context.pad_top,
            )

        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise InferenceError(f"Detection failed: {e}")

    def _parse_results(self, result, frame_shape: tuple, context: Optional[DetectionContext] = None) -> List[Detection]:
        """Parses raw YOLO detection boxes and maps coordinates back to original frame space."""
        detections = []

        if result.boxes is None:
            return detections

        try:
            num_boxes = len(result.boxes)
        except (TypeError, AttributeError):
            num_boxes = 0

        if num_boxes == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Determine whether coordinate remapping back to unpadded / unscaled original frame is needed
        need_remapping = False
        scale_x = 1.0
        scale_y = 1.0
        pad_left = 0
        pad_top = 0
        orig_w = frame_shape[1]
        orig_h = frame_shape[0]

        if context is not None:
            if context.scale_x != 1.0 or context.scale_y != 1.0 or context.pad_left != 0 or context.pad_top != 0:
                need_remapping = True
                scale_x = context.scale_x
                scale_y = context.scale_y
                pad_left = context.pad_left
                pad_top = context.pad_top
                if context.original_frame_shape and context.original_frame_shape != (0, 0):
                    orig_h, orig_w = context.original_frame_shape
            elif context.original_frame_shape and context.original_frame_shape != (0, 0) and context.original_frame_shape != frame_shape[:2]:
                orig_h, orig_w = context.original_frame_shape
                cur_h, cur_w = frame_shape[:2]
                if cur_w > 0 and cur_h > 0:
                    need_remapping = True
                    scale_x = cur_w / float(orig_w)
                    scale_y = cur_h / float(orig_h)

        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            if cls_id != self.config.person_class_id:
                continue

            if conf < self.config.confidence_threshold:
                continue

            b = box.tolist()
            if need_remapping:
                x1 = (b[0] - pad_left) / max(scale_x, 1e-6)
                y1 = (b[1] - pad_top) / max(scale_y, 1e-6)
                x2 = (b[2] - pad_left) / max(scale_x, 1e-6)
                y2 = (b[3] - pad_top) / max(scale_y, 1e-6)

                x1 = float(np.clip(x1, 0, orig_w))
                y1 = float(np.clip(y1, 0, orig_h))
                x2 = float(np.clip(x2, 0, orig_w))
                y2 = float(np.clip(y2, 0, orig_h))
                mapped_box = [x1, y1, x2, y2]
            else:
                mapped_box = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]

            class_name = self._class_names.get(cls_id, self.config.person_class_name)

            detection = Detection(
                bbox=mapped_box,
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=class_name,
                track_id=None,
            )
            detections.append(detection)

        return detections

    def unload_model(self) -> None:
        """Releases YOLO weights and frees CPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            self._model_info = None
            logger.info("YOLO model unloaded")

    def get_model_info(self) -> Optional[ModelInfo]:
        return self._model_info


# Class alias for SENTINEL pipeline naming clarity
YOLOPersonDetector = YOLODetector


def create_detector(config: Optional[DetectorConfig] = None) -> YOLODetector:
    """Factory creating a configured YOLODetector instance."""
    if config is None:
        config = DetectorConfig()
    return YOLODetector(config)
