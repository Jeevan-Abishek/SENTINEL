"""
Validators Utility Module.
Validates input video files, GPS coordinates, and polygonal zones.
"""

from typing import Tuple, Optional, List
from pathlib import Path
from shapely.geometry import Polygon

from app.video.validator import (
    validate_video_file,
    is_supported_video_format,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from app.mapping.zones import PolygonZoneValidator


def validate_gps_coordinates(lat: float, lon: float, alt: Optional[float] = None) -> bool:
    """Validate latitude, longitude and altitude boundaries."""
    if not (-90.0 <= lat <= 90.0):
        return False
    if not (-180.0 <= lon <= 180.0):
        return False
    if alt is not None and alt < -500.0:
        return False
    return True


def validate_polygon_points(points: List[List[float]]) -> Tuple[bool, str]:
    """Validate that points form a valid non-self-intersecting polygon with >= 3 vertices."""
    return PolygonZoneValidator.validate_polygon(points)


__all__ = [
    "validate_video_file",
    "is_supported_video_format",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "validate_gps_coordinates",
    "validate_polygon_points",
]

