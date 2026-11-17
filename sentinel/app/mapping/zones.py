"""Zone management and polygon geometry validation for drone surveillance areas."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid
from loguru import logger

from app.mapping.calibration import CoordinateCalibrator


@dataclass
class ZoneDefinition:
    zone_id: str
    name: str
    zone_type: str  # e.g., 'crowd', 'bottleneck', 'ingress', 'egress', 'restricted'
    image_polygon: List[Tuple[float, float]]
    world_polygon: Optional[List[Tuple[float, float]]] = None
    capacity: int = 100
    area_m2: Optional[float] = None
    is_calibrated: bool = False
    entry_line: Optional[List[Tuple[float, float]]] = None
    exit_line: Optional[List[Tuple[float, float]]] = None
    current_count: int = 0
    current_density_people_per_m2: Optional[float] = None
    risk_level: str = "safe"


class PolygonZoneValidator:
    """Validates and sanitizes 2D polygon boundaries using Shapely."""

    @staticmethod
    def validate_polygon(points: List[Tuple[float, float]]) -> Tuple[bool, str, Optional[Polygon]]:
        """Validate a list of (x, y) points as a proper non-degenerate polygon.

        Returns:
            (is_valid, message, shapely_polygon)
        """
        if not points or len(points) < 3:
            return False, "A polygon must contain at least 3 points", None

        # Check for finite values
        for p in points:
            if not np.isfinite(p[0]) or not np.isfinite(p[1]):
                return False, "Polygon coordinates contain non-finite values", None

        try:
            poly = Polygon(points)
            if not poly.is_valid:
                # Attempt to repair self-intersecting or bowtie polygon
                fixed = make_valid(poly)
                if isinstance(fixed, Polygon) and fixed.is_valid and not fixed.is_empty:
                    return True, "Polygon repaired with make_valid", fixed
                return False, f"Invalid self-intersecting polygon: {poly.is_valid}", None

            if poly.area < 1e-5:
                return False, "Degenerate polygon with near-zero area", None

            return True, "Valid polygon", poly
        except Exception as e:
            return False, f"Polygon validation error: {e}", None


class ZoneManager:
    """Manages monitoring zones, point-in-polygon spatial queries, and density calculations."""

    def __init__(self, calibrator: Optional[CoordinateCalibrator] = None):
        self.calibrator = calibrator
        self.zones: Dict[str, ZoneDefinition] = {}

    def set_calibrator(self, calibrator: CoordinateCalibrator) -> None:
        self.calibrator = calibrator

    def add_zone(
        self,
        zone_id: str,
        name: str,
        zone_type: str,
        image_polygon: List[Tuple[float, float]],
        capacity: int = 100,
        entry_line: Optional[List[Tuple[float, float]]] = None,
        exit_line: Optional[List[Tuple[float, float]]] = None,
        altitude_m: Optional[float] = None,
    ) -> bool:
        """Validate and add a zone definition."""
        is_valid, msg, poly = PolygonZoneValidator.validate_polygon(image_polygon)
        if not is_valid:
            logger.warning(f"Failed to add zone '{zone_id}': {msg}")
            return False

        world_polygon = None
        area_m2 = None
        is_calibrated = False

        if self.calibrator:
            # Try ground area calculation
            area_m2 = self.calibrator.calculate_polygon_ground_area(image_polygon, altitude_m=altitude_m)
            if area_m2 is not None and area_m2 > 0:
                is_calibrated = True

            # Transform polygon to world coords if homography is available
            if self.calibrator.homography_matrix is not None:
                transformed = [self.calibrator.image_to_world(x, y) for x, y in image_polygon]
                if all(p is not None for p in transformed):
                    world_polygon = transformed

        self.zones[zone_id] = ZoneDefinition(
            zone_id=zone_id,
            name=name,
            zone_type=zone_type,
            image_polygon=image_polygon,
            world_polygon=world_polygon,
            capacity=capacity,
            area_m2=area_m2,
            is_calibrated=is_calibrated,
            entry_line=entry_line,
            exit_line=exit_line,
        )
        return True

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self.zones:
            del self.zones[zone_id]
            return True
        return False

    def get_zone(self, zone_id: str) -> Optional[ZoneDefinition]:
        return self.zones.get(zone_id)

    def get_all_zones(self) -> List[ZoneDefinition]:
        return list(self.zones.values())

    def is_point_in_zone(self, zone_id: str, x: float, y: float, use_world: bool = False) -> bool:
        """Check if a point (x, y) is inside the specified zone."""
        zone = self.zones.get(zone_id)
        if not zone:
            return False

        pts = zone.world_polygon if use_world and zone.world_polygon else zone.image_polygon
        if not pts or len(pts) < 3:
            return False

        poly = Polygon(pts)
        return poly.contains(Point(x, y))

    def update_zone_metrics(self, zone_id: str, count: int) -> Dict[str, Any]:
        """Update count and compute density (calibrated people/m² or estimated)."""
        zone = self.zones.get(zone_id)
        if not zone:
            return {}

        zone.current_count = count

        if zone.is_calibrated and zone.area_m2 and zone.area_m2 > 0:
            zone.current_density_people_per_m2 = round(count / zone.area_m2, 2)
            density_mode = "calibrated_physical_density"
        else:
            zone.current_density_people_per_m2 = None
            density_mode = "uncalibrated_occupancy_estimate"

        # Occupancy ratio
        occ_ratio = count / max(zone.capacity, 1)
        if occ_ratio > 0.85:
            zone.risk_level = "critical"
        elif occ_ratio > 0.65:
            zone.risk_level = "warning"
        else:
            zone.risk_level = "safe"

        return {
            "zone_id": zone_id,
            "count": count,
            "capacity": zone.capacity,
            "occupancy_ratio": round(occ_ratio, 2),
            "density_people_per_m2": zone.current_density_people_per_m2,
            "density_mode": density_mode,
            "is_calibrated": zone.is_calibrated,
            "area_m2": round(zone.area_m2, 2) if zone.area_m2 else None,
            "risk_level": zone.risk_level,
        }

    def to_geojson(self) -> dict:
        features = []
        for zone in self.zones.values():
            coords = zone.world_polygon if zone.world_polygon else zone.image_polygon
            if coords:
                closed_coords = coords + [coords[0]]
                features.append({
                    "type": "Feature",
                    "properties": {
                        "zone_id": zone.zone_id,
                        "name": zone.name,
                        "zone_type": zone.zone_type,
                        "capacity": zone.capacity,
                        "current_count": zone.current_count,
                        "area_m2": zone.area_m2,
                        "is_calibrated": zone.is_calibrated,
                        "risk_level": zone.risk_level,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [closed_coords],
                    },
                })

        return {"type": "FeatureCollection", "features": features}

