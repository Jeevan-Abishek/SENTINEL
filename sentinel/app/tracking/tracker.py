import time
import numpy as np
from typing import List, Optional, Tuple

from app.tracking.models import Track, TrackResult, TrackerConfig, TrackingContext
from app.tracking.bytetrack import BYTETracker, create_tracker
from app.detection.models import Detection, DetectionResult


class Tracker:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.bytetracker = create_tracker(config)
        self._is_initialized = False

    def initialize(self) -> bool:
        result = self.bytetracker.initialize()
        self._is_initialized = True
        return result

    def update(self, detection_result: DetectionResult, context: TrackingContext) -> TrackResult:
        if not self._is_initialized:
            self.initialize()

        start_time = time.perf_counter()

        # Convert detections to ByteTrack format
        detections, scores = self._detections_to_bytetrack(detection_result.detections)

        # Update tracker
        active_tracks = self.bytetracker.update(detections, scores)

        # Convert back to Track objects
        tracks = self._bytetrack_to_tracks(active_tracks, detection_result)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return TrackResult(
            tracks=tracks,
            frame_number=context.frame_number,
            timestamp=context.timestamp,
            processing_time_ms=processing_time_ms,
            num_detections=len(detection_result.detections),
            num_active_tracks=len([t for t in tracks if t.state == "active"]),
            num_lost_tracks=len([t for t in tracks if t.state == "lost"]),
        )

    def _detections_to_bytetrack(self, detections: List[Detection]) -> Tuple[np.ndarray, np.ndarray]:
        if not detections:
            return np.empty((0, 4)), np.empty((0,))

        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)
        return boxes, scores

    def _bytetrack_to_tracks(self, bytetracks: List, detection_result: DetectionResult) -> List[Track]:
        tracks = []
        for bt in bytetracks:
            bbox = bt.get_state()
            track = Track(
                track_id=bt.id,
                bbox=bbox.tolist(),
                confidence=1.0,  # Track confidence (could be average of matched detections)
                class_id=0,
                class_name="person",
                age=bt.age,
                hits=bt.hits,
                time_since_update=bt.time_since_update,
                state=bt.state,
                bbox_history=[h.flatten().tolist() for h in bt.history[-30:]],  # Keep last 30 frames
                centroid_history=[[(h.flatten()[0] + h.flatten()[2])/2, (h.flatten()[1] + h.flatten()[3])/2] for h in bt.history[-30:]] if bt.history else [],
            )
            tracks.append(track)
        return tracks

    def reset(self):
        self.bytetracker.reset()


def create_tracker_wrapper(config: Optional[TrackerConfig] = None) -> Tracker:
    if config is None:
        config = TrackerConfig()
    return Tracker(config)