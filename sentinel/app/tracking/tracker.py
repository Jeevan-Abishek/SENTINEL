import time
import numpy as np
from typing import List, Optional, Tuple, Any

from app.tracking.models import (
    Track,
    TrackedPerson,
    TrackResult,
    TrackerConfig,
    TrackingContext,
)
from app.tracking.bytetrack import create_tracker
from app.tracking.validator import SecondStageValidator, ValidationConfig, ValidationReport
from app.tracking.reid import AppearanceReID, ReIDConfig, ReIDMatch
from app.detection.models import Detection, DetectionResult


class Tracker:
    """
    Second-Stage Tracking, Validation & Appearance Re-ID Engine for SENTINEL.
    Combines high-efficiency multi-object tracking (ByteTrack / pure NumPy fallback)
    with aerial drone geometric sanity checks, temporal validation, and Appearance Re-ID.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self.bytetracker = create_tracker(self.config)
        self.validator = SecondStageValidator(
            ValidationConfig(
                min_hits=getattr(self.config, "min_hits", 3),
                min_box_area=getattr(self.config, "min_box_area", 10.0),
                max_box_area=getattr(self.config, "max_box_area", 500000.0),
                min_aspect_ratio=getattr(self.config, "min_aspect_ratio", 0.15),
                max_aspect_ratio=getattr(self.config, "max_aspect_ratio", 3.5),
                max_velocity_jump=getattr(self.config, "max_velocity_jump", 150.0),
                enabled=getattr(self.config, "validation_enabled", True),
            )
        )
        self.reid = AppearanceReID(
            ReIDConfig(
                enabled=getattr(self.config, "reid_enabled", False),
                similarity_threshold=getattr(self.config, "reid_threshold", 0.65),
                ttl_frames=getattr(self.config, "reid_ttl_frames", 90),
                max_gallery_size=getattr(self.config, "reid_max_gallery_size", 200),
                model_path=getattr(self.config, "reid_model_path", None),
            )
        )
        self._is_initialized = False

    def initialize(self) -> bool:
        result = self.bytetracker.initialize()
        self.reid.initialize()
        self._is_initialized = True
        return result

    def update(
        self,
        detection_result: DetectionResult,
        context: Optional[TrackingContext] = None,
        frame: Optional[np.ndarray] = None,
    ) -> TrackResult:
        """
        Processes frame detections through multi-object tracking, second-stage validation,
        and optional Appearance Re-ID identity matching across temporary occlusions.
        Returns TrackResult with active tracks, verified validated_tracks, and Re-ID metrics.
        """
        if not self._is_initialized:
            self.initialize()

        if context is None:
            context = TrackingContext(
                frame_number=getattr(detection_result, "frame_number", 0),
                timestamp=getattr(detection_result, "timestamp", time.time()),
                original_frame_shape=(
                    getattr(detection_result, "image_height", 1080),
                    getattr(detection_result, "image_width", 1920),
                ),
            )

        start_time = time.perf_counter()

        # Convert detections to tracker bounding box and score arrays
        detections, scores = self._detections_to_bytetrack(detection_result.detections)

        # Update tracker state
        active_tracks = self.bytetracker.update(detections, scores)

        # Convert raw tracker output to typed TrackedPerson instances
        tracks = self._bytetrack_to_tracks(active_tracks, detection_result)

        # Apply Second-Stage Scanning & Validation
        val_report: ValidationReport = self.validator.validate_tracks(tracks)

        # Appearance Re-ID Identity Matching (if enabled)
        if self.reid.enabled:
            # 1. First transfer newly lost tracks to lost gallery for matching
            lost_tracker_tracks = getattr(self.bytetracker, "lost_tracks", [])
            for lt in lost_tracker_tracks:
                self.reid.register_lost_track(
                    track_id=lt.id,
                    bbox=lt.get_state().tolist(),
                    frame_number=context.frame_number,
                    timestamp=context.timestamp,
                )

            # 2. Extract appearance signatures & match candidates against lost gallery (if frame provided)
            if frame is not None:
                for track in tracks:
                    if track.state == "active":
                        sig = self.reid.extract_from_frame(frame, track.bbox)
                        if sig is not None:
                            track.embedding = sig.tolist()
                            # If track is newly initialized (hits <= 1), check for re-entry match in lost gallery
                            if track.hits <= 1:
                                match = self.reid.match_candidate(
                                    candidate_sig=sig,
                                    candidate_bbox=track.bbox,
                                    current_frame=context.frame_number,
                                    current_timestamp=context.timestamp,
                                )
                                if match is not None:
                                    track.track_id = match.old_track_id

                            # Update active track signature gallery
                            self.reid.register_active_track(
                                track_id=track.track_id,
                                signature=sig,
                                bbox=track.bbox,
                                frame_number=context.frame_number,
                                timestamp=context.timestamp,
                            )

            # 3. Prune expired entries beyond TTL
            self.reid.prune_gallery(context.frame_number)

        processing_time_ms = (time.perf_counter() - start_time) * 1000.0

        return TrackResult(
            tracks=tracks,
            frame_number=context.frame_number,
            timestamp=context.timestamp,
            processing_time_ms=processing_time_ms,
            num_detections=len(detection_result.detections),
            rejected_count=val_report.rejected_count,
            validation_reasons=val_report.validation_reasons,
            reid_matches=self.reid.total_matches,
            gallery_size=self.reid.gallery_size,
            reid_events=[m.model_dump() for m in self.reid.match_history[-10:]],
        )

    def validate(self, tracks: List[TrackedPerson]) -> ValidationReport:
        """Directly run second-stage validation on a list of tracks."""
        return self.validator.validate_tracks(tracks)

    def get_validated_tracks(
        self,
        detection_result: DetectionResult,
        context: Optional[TrackingContext] = None,
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackedPerson]:
        """Convenience method returning only tracks that passed second-stage validation."""
        result = self.update(detection_result, context, frame=frame)
        return result.validated_tracks

    def _detections_to_bytetrack(self, detections: List[Detection]) -> Tuple[np.ndarray, np.ndarray]:
        if not detections:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)
        return boxes, scores

    def _bytetrack_to_tracks(self, bytetracks: List, detection_result: DetectionResult) -> List[TrackedPerson]:
        tracks = []
        for bt in bytetracks:
            bbox = bt.get_state()
            history = bt.history[-30:] if hasattr(bt, "history") and bt.history else []
            bbox_history = [h.flatten().tolist() for h in history]
            centroid_history = [
                [(h.flatten()[0] + h.flatten()[2]) / 2.0, (h.flatten()[1] + h.flatten()[3]) / 2.0]
                for h in history
            ] if history else []

            track = TrackedPerson(
                track_id=bt.id,
                bbox=bbox.tolist(),
                confidence=1.0,
                class_id=0,
                class_name="person",
                age=bt.age,
                hits=bt.hits,
                time_since_update=bt.time_since_update,
                state=bt.state,
                bbox_history=bbox_history,
                centroid_history=centroid_history,
            )
            tracks.append(track)
        return tracks

    def reset(self):
        self.bytetracker.reset()
        self.reid.reset()


def create_tracker_wrapper(config: Optional[TrackerConfig] = None) -> Tracker:
    if config is None:
        config = TrackerConfig()
    return Tracker(config)
