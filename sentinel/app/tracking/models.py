from typing import Optional, List
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import numpy as np


class Track(BaseModel):
    track_id: int = Field(..., description="Unique track identifier")
    bbox: list[float] = Field(..., description="Current bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(default=0, description="Class ID")
    class_name: str = Field(default="person", description="Class name")
    age: int = Field(default=0, description="Number of frames since track creation")
    hits: int = Field(default=1, description="Number of detection matches")
    time_since_update: int = Field(default=0, description="Frames since last update")
    state: str = Field(default="active", description="Track state: active, lost, removed")
    
    # History
    bbox_history: list[list[float]] = Field(default_factory=list, description="Historical bounding boxes")
    centroid_history: list[list[float]] = Field(default_factory=list, description="Historical centroids")
    
    # Optional Re-ID features
    embedding: Optional[list[float]] = Field(default=None, description="Appearance embedding for Re-ID")

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
    def centroid(self) -> list[float]:
        return [self.center_x, self.center_y]

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xywh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]

    def get_velocity(self, frame_rate: float = 30.0) -> list[float]:
        if len(self.centroid_history) < 2:
            return [0.0, 0.0]
        prev = np.array(self.centroid_history[-2])
        curr = np.array(self.centroid_history[-1])
        velocity = (curr - prev) * frame_rate
        return velocity.tolist()

    def get_speed(self, frame_rate: float = 30.0) -> float:
        vel = self.get_velocity(frame_rate)
        return float(np.linalg.norm(vel))

    def get_direction(self, frame_rate: float = 30.0) -> float:
        vel = self.get_velocity(frame_rate)
        if vel[0] == 0 and vel[1] == 0:
            return 0.0
        return float(np.arctan2(vel[1], vel[0]))


class TrackResult(BaseModel):
    tracks: list[Track] = Field(default_factory=list)
    frame_number: int = 0
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    num_detections: int = 0

    @property
    def num_active_tracks(self) -> int:
        return len([t for t in self.tracks if t.state == "active"])

    @property
    def num_lost_tracks(self) -> int:
        return len([t for t in self.tracks if t.state == "lost"])

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state == "active"]

    @property
    def lost_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state == "lost"]


@dataclass
class TrackerConfig:
    tracker_type: str = "bytetrack"  # "bytetrack" or "botsort"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_box_area: float = 10.0
    mot20: bool = False
    frame_rate: int = 30
    
    # Re-ID (for BoT-SORT)
    reid_enabled: bool = False
    reid_model_path: str = ""
    reid_threshold: float = 0.6

    # ByteTrack specific
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    new_track_thresh: float = 0.7


@dataclass
class TrackingContext:
    frame_number: int = 0
    timestamp: float = 0.0
    original_frame_shape: tuple[int, int] = (1080, 1920)


class TrackingError(Exception):
    pass


class TrackerInitError(TrackingError):
    pass