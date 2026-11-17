"""Geospatial calibration for drone camera surveillance.

Provides homography-based perspective transformation (mapping image pixels to real-world
meters or GPS coordinates) and altitude/FOV ground sampling distance (GSD) calculations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
import cv2
from loguru import logger


@dataclass
class CalibrationPoint:
    image_point: Tuple[float, float]  # (x, y) in pixels
    world_point: Tuple[float, float]  # (x, y) in meters or (lon, lat)


@dataclass
class CalibrationResult:
    homography_matrix: np.ndarray  # 3x3 matrix
    reprojection_error: float
    is_valid: bool
    world_bounds: Optional[Tuple[float, float, float, float]] = None  # min_x, min_y, max_x, max_y


@dataclass
class CameraFOVParams:
    """Drone camera sensor and optical characteristics."""
    hfov_deg: float = 84.0  # Horizontal FOV (typical DJI wide camera)
    vfov_deg: float = 56.0  # Vertical FOV
    image_width: int = 1920
    image_height: int = 1080


@dataclass
class GroundSamplingDistance:
    """Represents real-world physical area represented by camera pixels."""
    meters_per_pixel_x: float
    meters_per_pixel_y: float
    ground_footprint_width_m: float
    ground_footprint_height_m: float
    ground_area_per_pixel_m2: float
    is_valid: bool = True


class GroundResolutionCalculator:
    """Calculates ground footprint and ground sampling distance (GSD) from drone altitude."""

    @staticmethod
    def calculate_gsd(
        altitude_m: float,
        params: Optional[CameraFOVParams] = None,
        gimbal_pitch_deg: float = -90.0,
    ) -> GroundSamplingDistance:
        """Compute ground resolution at nadir or oblique gimbal pitch.

        Args:
            altitude_m: Drone altitude above ground in meters.
            params: Camera optics parameters.
            gimbal_pitch_deg: Gimbal angle (-90 is straight down nadir).

        Returns:
            GroundSamplingDistance object with meters-per-pixel and footprint metrics.
        """
        if altitude_m <= 0:
            return GroundSamplingDistance(
                meters_per_pixel_x=0.0,
                meters_per_pixel_y=0.0,
                ground_footprint_width_m=0.0,
                ground_footprint_height_m=0.0,
                ground_area_per_pixel_m2=0.0,
                is_valid=False,
            )

        p = params or CameraFOVParams()
        hfov_rad = math.radians(p.hfov_deg)
        vfov_rad = math.radians(p.vfov_deg)

        # Footprint at nadir (straight down)
        footprint_w = 2.0 * altitude_m * math.tan(hfov_rad / 2.0)
        footprint_h = 2.0 * altitude_m * math.tan(vfov_rad / 2.0)

        # Account for oblique pitch if gimbal is tilted from -90
        tilt_from_nadir = abs(gimbal_pitch_deg - (-90.0))
        if tilt_from_nadir > 0.1:
            pitch_rad = math.radians(tilt_from_nadir)
            footprint_h = footprint_h / max(math.cos(pitch_rad), 0.1)

        gsd_x = footprint_w / max(p.image_width, 1)
        gsd_y = footprint_h / max(p.image_height, 1)
        pixel_area_m2 = gsd_x * gsd_y

        return GroundSamplingDistance(
            meters_per_pixel_x=gsd_x,
            meters_per_pixel_y=gsd_y,
            ground_footprint_width_m=footprint_w,
            ground_footprint_height_m=footprint_h,
            ground_area_per_pixel_m2=pixel_area_m2,
            is_valid=True,
        )


class CoordinateCalibrator:
    """Computes homography between camera pixels and ground-plane real world coordinates."""

    def __init__(self, min_points: int = 4):
        self.min_points = min_points
        self.calibration_points: List[CalibrationPoint] = []
        self.homography_matrix: Optional[np.ndarray] = None
        self.world_bounds: Optional[Tuple[float, float, float, float]] = None

    def add_calibration_point(self, image_x: float, image_y: float, world_x: float, world_y: float):
        point = CalibrationPoint(
            image_point=(float(image_x), float(image_y)),
            world_point=(float(world_x), float(world_y)),
        )
        self.calibration_points.append(point)

    def remove_calibration_point(self, index: int):
        if 0 <= index < len(self.calibration_points):
            self.calibration_points.pop(index)

    def clear_calibration_points(self):
        self.calibration_points = []
        self.homography_matrix = None
        self.world_bounds = None

    def compute_homography(self) -> CalibrationResult:
        """Compute 3x3 homography matrix using RANSAC from reference points."""
        if len(self.calibration_points) < self.min_points:
            return CalibrationResult(
                homography_matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False,
            )

        image_points = np.array([p.image_point for p in self.calibration_points], dtype=np.float32)
        world_points = np.array([p.world_point for p in self.calibration_points], dtype=np.float32)

        try:
            H, mask = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)

            if H is None or np.isnan(H).any() or abs(np.linalg.det(H)) < 1e-12:
                return CalibrationResult(
                    homography_matrix=np.eye(3),
                    reprojection_error=float('inf'),
                    is_valid=False,
                )

            # Compute reprojection error
            projected = cv2.perspectiveTransform(image_points.reshape(-1, 1, 2), H).reshape(-1, 2)
            errors = np.linalg.norm(projected - world_points, axis=1)
            mean_error = float(np.mean(errors))

            # Compute world bounds
            min_x = float(np.min(world_points[:, 0]))
            max_x = float(np.max(world_points[:, 0]))
            min_y = float(np.min(world_points[:, 1]))
            max_y = float(np.max(world_points[:, 1]))

            self.homography_matrix = H
            self.world_bounds = (min_x, min_y, max_x, max_y)

            return CalibrationResult(
                homography_matrix=H,
                reprojection_error=mean_error,
                is_valid=mean_error < 15.0,
                world_bounds=(min_x, min_y, max_x, max_y),
            )
        except Exception as e:
            logger.warning(f"Homography computation failed: {e}")
            return CalibrationResult(
                homography_matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False,
            )

    def image_to_world(self, image_x: float, image_y: float) -> Optional[Tuple[float, float]]:
        """Transform image pixel coordinates (x, y) to world plane coordinates."""
        if self.homography_matrix is None:
            return None

        point = np.array([[[image_x, image_y]]], dtype=np.float32)
        try:
            world_point = cv2.perspectiveTransform(point, self.homography_matrix)
            return (float(world_point[0, 0, 0]), float(world_point[0, 0, 1]))
        except Exception:
            return None

    def world_to_image(self, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        """Transform world plane coordinates to image pixel coordinates."""
        if self.homography_matrix is None:
            return None

        try:
            H_inv = np.linalg.inv(self.homography_matrix)
            point = np.array([[[world_x, world_y]]], dtype=np.float32)
            image_point = cv2.perspectiveTransform(point, H_inv)
            return (float(image_point[0, 0, 0]), float(image_point[0, 0, 1]))
        except Exception:
            return None

    def calculate_polygon_ground_area(
        self,
        polygon_pixels: List[Tuple[float, float]],
        altitude_m: Optional[float] = None,
    ) -> Optional[float]:
        """Calculate real-world area in square meters (m²) of a pixel polygon.

        Uses homography if valid; falls back to altitude-based GSD if altitude is given.
        """
        if len(polygon_pixels) < 3:
            return None

        # Method 1: Homography-based world polygon area
        if self.homography_matrix is not None:
            world_pts = [self.image_to_world(px, py) for px, py in polygon_pixels]
            if all(pt is not None for pt in world_pts):
                # Green's theorem / shoelace formula for polygon area
                x = np.array([p[0] for p in world_pts])
                y = np.array([p[1] for p in world_pts])
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                return float(area)

        # Method 2: Altitude GSD fallback
        if altitude_m is not None and altitude_m > 0:
            gsd = GroundResolutionCalculator.calculate_gsd(altitude_m)
            if gsd.is_valid:
                # Pixel area via shoelace
                x = np.array([p[0] for p in polygon_pixels])
                y = np.array([p[1] for p in polygon_pixels])
                pixel_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                return float(pixel_area * gsd.ground_area_per_pixel_m2)

        return None

    def get_calibration_data(self) -> dict:
        return {
            "points": [
                {
                    "image_x": p.image_point[0],
                    "image_y": p.image_point[1],
                    "world_x": p.world_point[0],
                    "world_y": p.world_point[1],
                }
                for p in self.calibration_points
            ],
            "homography": self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            "world_bounds": self.world_bounds,
        }

    def load_calibration_data(self, data: dict):
        self.calibration_points = [
            CalibrationPoint(
                image_point=(p["image_x"], p["image_y"]),
                world_point=(p["world_x"], p["world_y"]),
            )
            for p in data.get("points", [])
        ]
        if data.get("homography"):
            self.homography_matrix = np.array(data["homography"], dtype=np.float32)
        self.world_bounds = data.get("world_bounds")

