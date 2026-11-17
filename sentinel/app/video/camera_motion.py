"""Camera Motion Compensation (CMC) for drone surveillance footage.

When a drone hovers or moves, camera ego-motion (pan, tilt, drift, vibration)
causes stationary ground objects and people to appear in motion across frames.
This module uses sparse optical flow (Lucas-Kanade) and robust affine RANSAC
estimation to determine inter-frame camera movement, enabling accurate ground-relative
crowd movement analysis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
from loguru import logger


@dataclass
class CameraMotionResult:
    """Represents estimated inter-frame camera motion."""
    dx: float = 0.0
    dy: float = 0.0
    rotation_deg: float = 0.0
    scale: float = 1.0
    transform_matrix: np.ndarray = None  # 2x3 affine matrix
    inliers_count: int = 0
    total_features: int = 0
    confidence: float = 1.0
    is_valid: bool = True

    def __post_init__(self):
        if self.transform_matrix is None:
            self.transform_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ], dtype=np.float32)


class CameraMotionCompensator:
    """Estimates and compensates for drone camera movement between consecutive frames."""

    def __init__(
        self,
        max_features: int = 300,
        quality_level: float = 0.01,
        min_distance: float = 15.0,
        ransac_threshold: float = 3.0,
    ):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.ransac_threshold = ransac_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal history."""
        self._prev_gray = None
        self._prev_pts = None

    def estimate_motion(
        self,
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> CameraMotionResult:
        """Estimate camera motion between the previous frame and curr_frame.

        Args:
            curr_frame: Current BGR or grayscale video frame.
            mask: Optional binary mask (0 = ignore, e.g. bounding boxes of moving people,
                  255 = feature candidate region).

        Returns:
            CameraMotionResult containing translation, rotation, scale and affine matrix.
        """
        if curr_frame is None or curr_frame.size == 0:
            return CameraMotionResult(is_valid=False, confidence=0.0)

        # Convert to grayscale
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame.copy()

        # Handle first frame
        if self._prev_gray is None:
            self._prev_gray = curr_gray
            self._prev_pts = cv2.goodFeaturesToTrack(
                curr_gray,
                maxCorners=self.max_features,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask,
            )
            return CameraMotionResult(
                dx=0.0, dy=0.0, rotation_deg=0.0, scale=1.0, is_valid=True, confidence=1.0
            )

        # If we have too few features from previous frame, re-detect
        if self._prev_pts is None or len(self._prev_pts) < 8:
            self._prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray,
                maxCorners=self.max_features,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                mask=mask,
            )

        if self._prev_pts is None or len(self._prev_pts) < 4:
            # Cannot estimate reliably
            self._prev_gray = curr_gray
            return CameraMotionResult(is_valid=False, confidence=0.0)

        # Optical flow tracking (Lucas-Kanade)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            curr_gray,
            self._prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        valid_prev = self._prev_pts[status.flatten() == 1]
        valid_curr = curr_pts[status.flatten() == 1]

        if len(valid_prev) < 4:
            self._prev_gray = curr_gray
            self._prev_pts = None
            return CameraMotionResult(is_valid=False, confidence=0.0)

        # Robust Affine Partial (translation + rotation + uniform scale)
        try:
            matrix, inliers = cv2.estimateAffinePartial2D(
                valid_prev,
                valid_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
            )
        except Exception as e:
            logger.debug(f"Motion estimation failed: {e}")
            matrix = None
            inliers = None

        if matrix is None:
            self._prev_gray = curr_gray
            self._prev_pts = None
            return CameraMotionResult(is_valid=False, confidence=0.0)

        inliers_count = int(np.sum(inliers)) if inliers is not None else 0
        total_tracked = len(valid_prev)
        confidence = inliers_count / max(total_tracked, 1)

        # Extract translation, rotation, scale
        dx = float(matrix[0, 2])
        dy = float(matrix[1, 2])
        a = float(matrix[0, 0])
        b = float(matrix[0, 1])
        scale = float(np.sqrt(a * a + b * b))
        rotation_rad = float(np.arctan2(b, a))
        rotation_deg = float(np.degrees(rotation_rad))

        # Update history for next frame
        self._prev_gray = curr_gray
        self._prev_pts = cv2.goodFeaturesToTrack(
            curr_gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
        )

        return CameraMotionResult(
            dx=dx,
            dy=dy,
            rotation_deg=rotation_deg,
            scale=scale,
            transform_matrix=matrix,
            inliers_count=inliers_count,
            total_features=total_tracked,
            confidence=confidence,
            is_valid=inliers_count >= 4,
        )

    def compensate_point(
        self, point: Tuple[float, float], motion: CameraMotionResult
    ) -> Tuple[float, float]:
        """Compensate a point (x, y) in current frame back to the previous frame's coordinates."""
        if not motion.is_valid or motion.transform_matrix is None:
            return point

        # Invert the 2x3 affine matrix
        M = motion.transform_matrix
        A = M[:, :2]
        b = M[:, 2]

        det = np.linalg.det(A)
        if abs(det) < 1e-6:
            return point

        A_inv = np.linalg.inv(A)
        pt = np.array([point[0] - b[0], point[1] - b[1]], dtype=np.float32)
        comp_pt = A_inv @ pt
        return (float(comp_pt[0]), float(comp_pt[1]))

    def compensate_bbox(
        self, bbox: Tuple[float, float, float, float], motion: CameraMotionResult
    ) -> Tuple[float, float, float, float]:
        """Compensate a bounding box (x1, y1, x2, y2) for camera motion."""
        x1, y1, x2, y2 = bbox
        p1 = self.compensate_point((x1, y1), motion)
        p2 = self.compensate_point((x2, y2), motion)
        return (
            min(p1[0], p2[0]),
            min(p1[1], p2[1]),
            max(p1[0], p2[0]),
            max(p1[1], p2[1]),
        )

