import numpy as np
from typing import List, Optional, Tuple
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from loguru import logger

from app.tracking.models import Track, TrackResult, TrackerConfig, TrackingContext, TrackerInitError


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: np.ndarray, track_id: Optional[int] = None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        self.kf.x[4:] = 0.0

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count if track_id is None else track_id
        if track_id is None:
            KalmanBoxTracker.count += 1
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.history = []
        self.state = "active"

    def _convert_bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        return np.array([x, y, w, h]).reshape((4, 1))

    def _convert_x_to_bbox(self, x: np.ndarray) -> np.ndarray:
        w = x[2]
        h = x[3]
        x1 = x[0] - w / 2.0
        y1 = x[1] - h / 2.0
        x2 = x[0] + w / 2.0
        y2 = x[1] + h / 2.0
        return np.array([x1, y1, x2, y2]).reshape((1, 4))

    def update(self, bbox: np.ndarray):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._convert_bbox_to_z(bbox))
        self.state = "active"

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        return self._convert_x_to_bbox(self.kf.x).flatten()


class ByteTrack:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracked_tracks: List[KalmanBoxTracker] = []
        self.lost_tracks: List[KalmanBoxTracker] = []
        self.removed_tracks: List[KalmanBoxTracker] = []
        self.frame_id = 0
        self._next_id = 1

    def update(self, detections: np.ndarray, scores: np.ndarray) -> List[KalmanBoxTracker]:
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        if len(detections) > 0:
            det_scores = scores
            det_bboxes = detections

            # Split detections by confidence
            high_conf_mask = det_scores >= self.config.track_thresh
            low_conf_mask = det_scores < self.config.track_thresh

            high_conf_dets = det_bboxes[high_conf_mask]
            high_conf_scores = det_scores[high_conf_mask]
            low_conf_dets = det_bboxes[low_conf_mask]
            low_conf_scores = det_scores[low_conf_mask]

            # Predict existing tracked tracks
            for track in self.tracked_tracks:
                track.predict()

            # First association: high confidence detections with tracked tracks
            unmatched_dets = list(range(len(high_conf_dets)))
            unmatched_tracks = list(range(len(self.tracked_tracks)))
            if len(self.tracked_tracks) > 0 and len(high_conf_dets) > 0:
                iou_matrix = self._iou_matrix(high_conf_dets, self.tracked_tracks)
                matches, unmatched_dets, unmatched_tracks = self._hungarian_match(
                    iou_matrix, self.config.match_thresh
                )

                matched_track_indices = set()
                for det_idx, track_idx in matches:
                    track = self.tracked_tracks[track_idx]
                    track.update(high_conf_dets[det_idx])
                    activated_tracks.append(track)
                    matched_track_indices.add(track_idx)

                # Handle unmatched tracked tracks
                for track_idx in unmatched_tracks:
                    track = self.tracked_tracks[track_idx]
                    if track.time_since_update > self.config.track_buffer:
                        track.state = "removed"
                        removed_tracks.append(track)
                    else:
                        track.state = "lost"
                        lost_tracks.append(track)

            else:
                # No tracked tracks to match, all tracked tracks become lost
                for track in self.tracked_tracks:
                    track.state = "lost"
                    lost_tracks.append(track)

            # Second association: unmatched high confidence detections + low confidence detections with lost tracks
            # Track which high_conf_det indices are used for lost track recovery
            matched_high_conf_indices = set()
            
            all_lost_candidates = []
            high_conf_start_idx = 0
            if len(unmatched_dets) > 0 and len(high_conf_dets) > 0:
                unmatched_high_conf = high_conf_dets[unmatched_dets]
                if len(unmatched_high_conf) > 0:
                    all_lost_candidates.append(unmatched_high_conf)
                    high_conf_start_idx = 0  # high_conf detections are at the beginning
            low_conf_start_idx = len(all_lost_candidates[0]) if all_lost_candidates and len(unmatched_dets) > 0 and len(high_conf_dets) > 0 else 0
            if len(low_conf_dets) > 0:
                all_lost_candidates.append(low_conf_dets)

            if all_lost_candidates and len(self.lost_tracks) > 0:
                lost_candidate_dets = np.vstack(all_lost_candidates) if len(all_lost_candidates) > 1 else all_lost_candidates[0]
                iou_matrix = self._iou_matrix(lost_candidate_dets, self.lost_tracks)
                matches, unmatched_dets_lost, unmatched_tracks_lost = self._hungarian_match(
                    iou_matrix, 0.5
                )

                for det_idx, track_idx in matches:
                    track = self.lost_tracks[track_idx]
                    track.update(lost_candidate_dets[det_idx])
                    track.state = "active"
                    refind_tracks.append(track)
                    # Track which high_conf_dets were matched
                    if det_idx < len(unmatched_dets):
                        matched_high_conf_indices.add(unmatched_dets[det_idx])

                # Track which lost tracks were recovered
                recovered_lost_indices = {track_idx for _, track_idx in matches}

            else:
                recovered_lost_indices = set()

            # Create new tracks from remaining unmatched high confidence detections
            # (exclude those matched with lost tracks in second association)
            for det_idx in unmatched_dets:
                if det_idx not in matched_high_conf_indices:
                    new_track = KalmanBoxTracker(high_conf_dets[det_idx], self._next_id)
                    self._next_id += 1
                    activated_tracks.append(new_track)

            # Update lost tracks that were not recovered
            for i, track in enumerate(self.lost_tracks):
                if i not in recovered_lost_indices:
                    if track.time_since_update > self.config.track_buffer:
                        track.state = "removed"
                        removed_tracks.append(track)
                    else:
                        track.predict()

        else:
            # No detections, predict all tracks
            for track in self.tracked_tracks:
                track.predict()
                if track.time_since_update > self.config.track_buffer:
                    track.state = "removed"
                    removed_tracks.append(track)
                else:
                    track.state = "lost"
                    lost_tracks.append(track)

            for track in self.lost_tracks:
                track.predict()
                if track.time_since_update > self.config.track_buffer:
                    track.state = "removed"
                    removed_tracks.append(track)
                else:
                    track.state = "lost"
                    lost_tracks.append(track)

        # Update track lists
        self.tracked_tracks = [t for t in activated_tracks + refind_tracks if t.state == "active"]
        self.lost_tracks = [t for t in lost_tracks if t.state == "lost"]
        self.removed_tracks.extend(removed_tracks)

        # Return all active tracks
        return self.tracked_tracks

    def _iou_matrix(self, detections: np.ndarray, tracks: List[KalmanBoxTracker]) -> np.ndarray:
        if len(detections) == 0 or len(tracks) == 0:
            return np.zeros((len(detections), len(tracks)))

        track_boxes = np.array([t.get_state() for t in tracks])
        return self._batch_iou(detections, track_boxes)

    def _batch_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        # boxes: [x1, y1, x2, y2]
        x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

        xi1 = np.maximum(x11[:, None], x21[None, :])
        yi1 = np.maximum(y11[:, None], y21[None, :])
        xi2 = np.minimum(x12[:, None], x22[None, :])
        yi2 = np.minimum(y12[:, None], y22[None, :])

        inter_area = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        union_area = area1[:, None] + area2[None, :] - inter_area

        iou = inter_area / (union_area + 1e-6)
        return iou

    def _hungarian_match(self, iou_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if iou_matrix.size == 0:
            return [], list(range(iou_matrix.shape[0])), list(range(iou_matrix.shape[1]))

        # Convert to cost matrix (1 - IoU)
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_dets = []
        unmatched_tracks = []

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thresh:
                matches.append((r, c))
            else:
                unmatched_dets.append(r)
                unmatched_tracks.append(c)

        for r in range(iou_matrix.shape[0]):
            if r not in [m[0] for m in matches]:
                unmatched_dets.append(r)

        for c in range(iou_matrix.shape[1]):
            if c not in [m[1] for m in matches]:
                unmatched_tracks.append(c)

        return matches, unmatched_dets, unmatched_tracks

    def get_tracks(self) -> List[KalmanBoxTracker]:
        return self.tracked_tracks + self.lost_tracks

    def reset(self):
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        KalmanBoxTracker.count = 0
        self._next_id = 1


class BYTETracker:
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracker = ByteTrack(config)
        self._is_initialized = False

    def initialize(self) -> bool:
        try:
            if not self._is_initialized:
                self.tracker.reset()
                self._is_initialized = True
                logger.info("ByteTrack initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ByteTrack: {e}")
            raise TrackerInitError(f"ByteTrack initialization failed: {e}")

    def update(self, detections: np.ndarray, scores: np.ndarray) -> List[KalmanBoxTracker]:
        if not self._is_initialized:
            self.initialize()
        return self.tracker.update(detections, scores)

    def get_tracks(self) -> List[KalmanBoxTracker]:
        return self.tracker.get_tracks()

    def reset(self):
        self.tracker.reset()


def create_tracker(config: Optional[TrackerConfig] = None) -> BYTETracker:
    if config is None:
        config = TrackerConfig()
    return BYTETracker(config)