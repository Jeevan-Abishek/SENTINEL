from typing import Optional, Tuple, List
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass
import numpy as np


class Detection(BaseModel):
    bbox: list[float] = Field(..., description="Bounding box [x1, y1, x2, y2] in original frame coordinates")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    class_id: int = Field(default=0, description="Class ID (0 for person)")
    class_name: str = Field(default="person", description="Class name")
    track_id: Optional[int] = Field(default=None, description="Associated track ID if tracking is enabled")

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def y1(self) -> float:
        return self.bbox[1]

    @property
    def x2(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def center_xy(self) -> Tuple[float, float]:
        """(center_x, center_y) centroid coordinates."""
        return (self.center_x, self.center_y)

    @property
    def foot_xy(self) -> Tuple[float, float]:
        """(center_x, y2) bottom-center ground contact coordinates."""
        return (self.center_x, self.y2)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]

    def to_xyxy(self) -> list[float]:
        return self.bbox


class DetectionResult(BaseModel):
    detections: list[Detection] = Field(default_factory=list)
    frame_number: int = 0
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    model_name: str = ""
    image_width: int = 0
    image_height: int = 0
    orig_width: Optional[int] = None
    orig_height: Optional[int] = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_left: int = 0
    pad_top: int = 0

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def person_count(self) -> int:
        return len([d for d in self.detections if d.class_id == 0])

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(
            detections=filtered,
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            processing_time_ms=self.processing_time_ms,
            model_name=self.model_name,
            image_width=self.image_width,
            image_height=self.image_height,
            orig_width=self.orig_width,
            orig_height=self.orig_height,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            pad_left=self.pad_left,
            pad_top=self.pad_top,
        )

    def filter_by_class(self, class_id: int) -> "DetectionResult":
        filtered = [d for d in self.detections if d.class_id == class_id]
        return DetectionResult(
            detections=filtered,
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            processing_time_ms=self.processing_time_ms,
            model_name=self.model_name,
            image_width=self.image_width,
            image_height=self.image_height,
            orig_width=self.orig_width,
            orig_height=self.orig_height,
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            pad_left=self.pad_left,
            pad_top=self.pad_top,
        )


# Clean typed alias for pipeline clarity
FrameDetections = DetectionResult


@dataclass
class ModelInfo:
    name: str
    path: str
    input_size: tuple[int, int]
    num_classes: int
    class_names: dict[int, str]
    device: str
    half_precision: bool = False


class DetectorConfig(BaseModel):
    model_path: str = "models/yolov8n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cpu"
    half_precision: bool = False
    max_detections: int = 1000
    person_class_id: int = 0
    person_class_name: str = "person"
    classes: list[int] = Field(default_factory=lambda: [0])
    input_width: int = 640
    input_height: int = 640

    model_config = ConfigDict(arbitrary_types_allowed=True)