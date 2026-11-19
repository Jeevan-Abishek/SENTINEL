from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
import numpy as np


class TrackState(str, Enum):
    """Lifecycle states of a tracked person."""
    TENTATIVE = "tentative"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


class Track(BaseModel):
    """
    Core tracked person entity representing a unique identity across video frames.
    Also aliased as TrackedPerson for second-stage tracking and validation.
    """
    track_id: int = Field(..., description="Unique track identifier")
    bbox: list[float] = Field(..., description="Current bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(default=0, description="Class ID")
    class_name: str = Field(default="person", description="Class name")
    age: int = Field(default=0, description="Number of frames since track creation")
    hits: int = Field(default=1, description="Number of detection matches")
    time_since_update: int = Field(default=0, description="Frames since last update")
    state: str = Field(default="active", description="Track state: tentative, active, lost, removed")
    is_validated: bool = Field(default=False, description="Whether track passed second-stage validation")
    validation_reason: Optional[str] = Field(default=None, description="Reason or status of validation")
    
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
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def centroid(self) -> list[float]:
        return [self.center_x, self.center_y]

    @property
    def center_xy(self) -> tuple[float, float]:
        """Centroid coordinates (center_x, center_y)."""
        return (float(self.center_x), float(self.center_y))

    @property
    def foot_xy(self) -> tuple[float, float]:
        """Ground contact footprint point (center_x, y2) for homography mapping."""
        return (float(self.center_x), float(self.y2))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width / Height ratio (crucial for drone top-down filtering)."""
        h = self.height
        return (self.width / h) if h > 0 else 0.0

    def to_xywh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]

    def get_velocity(self, frame_rate: float = 30.0) -> list[float]:
        if len(self.centroid_history) < 2:
            return [0.0, 0.0]
        prev = np.array(self.centroid_history[-2])
        curr = np.array(self.centroid_history[-1])
        velocity = (curr - prev) * frame_rate
        return velocity.tolist()

    @property
    def velocity_xy(self) -> tuple[float, float]:
        """Current velocity vector (vx, vy) in px/sec."""
        vel = self.get_velocity()
        return (float(vel[0]), float(vel[1]))

    def get_speed(self, frame_rate: float = 30.0) -> float:
        vel = self.get_velocity(frame_rate)
        return float(np.linalg.norm(vel))

    @property
    def speed(self) -> float:
        """Current speed magnitude in px/sec."""
        return self.get_speed()

    def get_direction(self, frame_rate: float = 30.0) -> float:
        vel = self.get_velocity(frame_rate)
        if vel[0] == 0 and vel[1] == 0:
            return 0.0
        return float(np.arctan2(vel[1], vel[0]))

    @property
    def heading_deg(self) -> float:
        """Current movement heading angle in degrees [0.0, 360.0)."""
        dir_rad = self.get_direction()
        deg = float(np.degrees(dir_rad))
        return float((deg + 360.0) % 360.0)


# Backward-compatible and typed alias for Second-Stage Tracking
TrackedPerson = Track


class TrackResult(BaseModel):
    model_config = {"extra": "ignore"}

    tracks: list[Track] = Field(default_factory=list)
    frame_number: int = 0
    timestamp: float = 0.0
    processing_time_ms: float = 0.0
    num_detections: int = 0
    rejected_count: int = 0
    validation_reasons: dict[int, str] = Field(default_factory=dict)
    reid_matches: int = 0
    gallery_size: int = 0
    reid_events: list[dict] = Field(default_factory=list)

    @property
    def num_active_tracks(self) -> int:
        return len([t for t in self.tracks if t.state == "active"])

    @property
    def num_lost_tracks(self) -> int:
        return len([t for t in self.tracks if t.state == "lost"])

    @property
    def num_validated_tracks(self) -> int:
        return len(self.validated_tracks)

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state == "active"]

    @property
    def lost_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.state == "lost"]

    @property
    def validated_tracks(self) -> list[Track]:
        """Active tracks that have successfully passed second-stage validation."""
        return [t for t in self.tracks if getattr(t, "is_validated", False) and t.state == "active"]


@dataclass
class TrackerConfig:
    tracker_type: str = "bytetrack"  # "bytetrack", "lightweight", or "botsort"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    min_box_area: float = 10.0
    max_box_area: float = 500000.0
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 3.5
    min_hits: int = 3
    max_velocity_jump: float = 150.0
    validation_enabled: bool = True
    mot20: bool = False
    frame_rate: int = 30
    
    # Re-ID (for BoT-SORT / Appearance Re-ID)
    reid_enabled: bool = False
    reid_model_path: str = ""
    reid_threshold: float = 0.65
    reid_ttl_frames: int = 90
    reid_max_gallery_size: int = 200

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
