"""Zone management, polygon geometry validation, and area calibration for drone surveillance areas."""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any, Union
import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.validation import make_valid
from loguru import logger

from app.mapping.calibration import CoordinateCalibrator, GPSCoordinateConverter, ScaleCalibrator


@dataclass
class ZoneDefinition:
    zone_id: str
    name: str
    zone_type: str  # 'ingress', 'egress', 'main_assembly', 'choke_point', 'monitoring', 'emergency_exit', 'restricted'
    image_polygon: List[Tuple[float, float]]
    world_polygon: Optional[List[Tuple[float, float]]] = None
    gps_polygon: Optional[List[Tuple[float, float]]] = None  # [(lat, lon), ...]
    capacity: int = 100
    area_m2: Optional[float] = None
    is_calibrated: bool = False
    entry_line: Optional[List[Tuple[float, float]]] = None
    exit_line: Optional[List[Tuple[float, float]]] = None
    current_count: int = 0
    current_density_people_per_m2: Optional[float] = None
    flow_vector: Optional[Tuple[float, float]] = None  # (heading_deg, speed_m_s)
    risk_level: str = "safe"  # "safe", "warning", "critical"

    # Calibration & Threshold Parameters
    warning_density_p_sqm: float = 1.5
    critical_density_p_sqm: float = 3.0
    warning_occupancy_ratio: float = 0.65
    critical_occupancy_ratio: float = 0.85
    risk_weight: float = 1.0
    scale_factor: Optional[float] = None  # meters per pixel
    calibration_mode: str = "direct"  # 'direct', 'two_point_scale', 'altitude_gsd', 'homography'
    notes: str = ""

    @property
    def area_sqm(self) -> float:
        """Alias for real-world area in square meters."""
        return float(self.area_m2) if self.area_m2 is not None else 0.0

    @area_sqm.setter
    def area_sqm(self, value: float) -> None:
        self.area_m2 = float(value)

    @property
    def max_capacity(self) -> int:
        """Alias for maximum safe occupancy capacity."""
        return self.capacity

    @max_capacity.setter
    def max_capacity(self, value: int) -> None:
        self.capacity = int(value)

    @property
    def density_p_sqm(self) -> Optional[float]:
        """Alias for current crowd density in persons/m²."""
        return self.current_density_people_per_m2

    @density_p_sqm.setter
    def density_p_sqm(self, value: Optional[float]) -> None:
        self.current_density_people_per_m2 = float(value) if value is not None else None

    @property
    def risk_status(self) -> str:
        """Normalized uppercase risk status (SAFE / WARNING / CRITICAL)."""
        return self.risk_level.upper()

    @risk_status.setter
    def risk_status(self, value: str) -> None:
        self.risk_level = str(value).lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert zone definition to a clean serializable dictionary."""
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "zone_type": self.zone_type,
            "image_polygon": [list(p) for p in self.image_polygon],
            "world_polygon": [list(p) for p in self.world_polygon] if self.world_polygon else None,
            "gps_polygon": [list(p) for p in self.gps_polygon] if self.gps_polygon else None,
            "capacity": self.capacity,
            "max_capacity": self.max_capacity,
            "area_m2": self.area_m2,
            "area_sqm": self.area_sqm,
            "is_calibrated": self.is_calibrated,
            "current_count": self.current_count,
            "density_p_sqm": self.current_density_people_per_m2,
            "risk_status": self.risk_status,
            "risk_level": self.risk_level,
            "flow_vector": list(self.flow_vector) if self.flow_vector else None,
            "warning_density_p_sqm": self.warning_density_p_sqm,
            "critical_density_p_sqm": self.critical_density_p_sqm,
            "warning_occupancy_ratio": self.warning_occupancy_ratio,
            "critical_occupancy_ratio": self.critical_occupancy_ratio,
            "risk_weight": self.risk_weight,
            "scale_factor": self.scale_factor,
            "calibration_mode": self.calibration_mode,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZoneDefinition":
        """Instantiate a ZoneDefinition from a dictionary representation."""
        img_poly = [tuple(p) for p in data.get("image_polygon", [])]
        world_poly = [tuple(p) for p in data.get("world_polygon", [])] if data.get("world_polygon") else None
        gps_poly = [tuple(p) for p in data.get("gps_polygon", [])] if data.get("gps_polygon") else None
        flow = tuple(data["flow_vector"]) if data.get("flow_vector") else None

        return cls(
            zone_id=str(data["zone_id"]),
            name=str(data.get("name", data["zone_id"])),
            zone_type=str(data.get("zone_type", "monitoring")),
            image_polygon=img_poly,
            world_polygon=world_poly,
            gps_polygon=gps_poly,
            capacity=int(data.get("capacity") or data.get("max_capacity", 100)),
            area_m2=float(data.get("area_m2") or data.get("area_sqm", 0.0)) if (data.get("area_m2") or data.get("area_sqm")) else None,
            is_calibrated=bool(data.get("is_calibrated", True if data.get("area_m2") or data.get("area_sqm") else False)),
            current_count=int(data.get("current_count", 0)),
            current_density_people_per_m2=float(data["density_p_sqm"]) if data.get("density_p_sqm") is not None else None,
            flow_vector=flow,
            risk_level=str(data.get("risk_level") or data.get("risk_status", "safe")).lower(),
            warning_density_p_sqm=float(data.get("warning_density_p_sqm", 1.5)),
            critical_density_p_sqm=float(data.get("critical_density_p_sqm", 3.0)),
            warning_occupancy_ratio=float(data.get("warning_occupancy_ratio", 0.65)),
            critical_occupancy_ratio=float(data.get("critical_occupancy_ratio", 0.85)),
            risk_weight=float(data.get("risk_weight", 1.0)),
            scale_factor=float(data["scale_factor"]) if data.get("scale_factor") is not None else None,
            calibration_mode=str(data.get("calibration_mode", "direct")),
            notes=str(data.get("notes", "")),
        )


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
    """Manages monitoring zones, point-in-polygon spatial queries, area calibrations, and persistence."""

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
        scale_m_per_px: Optional[float] = None,
        gps_polygon: Optional[List[Tuple[float, float]]] = None,
        drone_lat: Optional[float] = None,
        drone_lon: Optional[float] = None,
        warning_density_p_sqm: float = 1.5,
        critical_density_p_sqm: float = 3.0,
        warning_occupancy_ratio: float = 0.65,
        critical_occupancy_ratio: float = 0.85,
        risk_weight: float = 1.0,
        calibration_mode: str = "direct",
        notes: str = "",
    ) -> bool:
        """Validate and add a zone definition with world and GPS projection."""
        is_valid, msg, poly = PolygonZoneValidator.validate_polygon(image_polygon)
        if not is_valid:
            logger.warning(f"Failed to add zone '{zone_id}': {msg}")
            return False

        world_polygon = None
        area_m2 = None
        is_calibrated = False

        if self.calibrator:
            # Try ground area calculation
            area_m2 = self.calibrator.calculate_polygon_ground_area(
                image_polygon,
                altitude_m=altitude_m,
                scale_m_per_px=scale_m_per_px,
            )
            if area_m2 is not None and area_m2 > 0:
                is_calibrated = True

            # Transform polygon to world coords if homography is available
            if self.calibrator.homography_matrix is not None:
                transformed = [self.calibrator.image_to_world(x, y) for x, y in image_polygon]
                if all(p is not None for p in transformed):
                    world_polygon = transformed

            # Compute GPS polygon if not explicitly provided
            if gps_polygon is None:
                computed_gps = self.calibrator.polygon_pixels_to_gps(
                    image_polygon,
                    drone_alt_m=altitude_m,
                    drone_lat=drone_lat,
                    drone_lon=drone_lon,
                )
                if computed_gps and len(computed_gps) >= 3:
                    gps_polygon = computed_gps

        self.zones[zone_id] = ZoneDefinition(
            zone_id=zone_id,
            name=name,
            zone_type=zone_type,
            image_polygon=image_polygon,
            world_polygon=world_polygon,
            gps_polygon=gps_polygon,
            capacity=capacity,
            area_m2=area_m2,
            is_calibrated=is_calibrated,
            entry_line=entry_line,
            exit_line=exit_line,
            warning_density_p_sqm=warning_density_p_sqm,
            critical_density_p_sqm=critical_density_p_sqm,
            warning_occupancy_ratio=warning_occupancy_ratio,
            critical_occupancy_ratio=critical_occupancy_ratio,
            risk_weight=risk_weight,
            scale_factor=scale_m_per_px,
            calibration_mode=calibration_mode,
            notes=notes,
        )
        return True

    def add_rectangular_zone(
        self,
        zone_id: str,
        name: str,
        zone_type: str,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        capacity: int = 100,
        area_m2: Optional[float] = None,
        altitude_m: Optional[float] = None,
        scale_m_per_px: Optional[float] = None,
        gps_polygon: Optional[List[Tuple[float, float]]] = None,
        warning_density_p_sqm: float = 1.5,
        critical_density_p_sqm: float = 3.0,
        warning_occupancy_ratio: float = 0.65,
        critical_occupancy_ratio: float = 0.85,
        risk_weight: float = 1.0,
        notes: str = "",
    ) -> bool:
        """Convenience method to define rectangular event zones (Ingress, Egress, Assembly, Choke)."""
        half_w = width / 2.0
        half_h = height / 2.0
        rect_pts = [
            (center_x - half_w, center_y - half_h),
            (center_x + half_w, center_y - half_h),
            (center_x + half_w, center_y + half_h),
            (center_x - half_w, center_y + half_h),
        ]
        success = self.add_zone(
            zone_id=zone_id,
            name=name,
            zone_type=zone_type,
            image_polygon=rect_pts,
            capacity=capacity,
            altitude_m=altitude_m,
            scale_m_per_px=scale_m_per_px,
            gps_polygon=gps_polygon,
            warning_density_p_sqm=warning_density_p_sqm,
            critical_density_p_sqm=critical_density_p_sqm,
            warning_occupancy_ratio=warning_occupancy_ratio,
            critical_occupancy_ratio=critical_occupancy_ratio,
            risk_weight=risk_weight,
            notes=notes,
        )
        if success and area_m2 is not None and area_m2 > 0:
            self.zones[zone_id].area_m2 = area_m2
            self.zones[zone_id].is_calibrated = True
        return success

    def update_zone_calibration(
        self,
        zone_id: str,
        name: Optional[str] = None,
        zone_type: Optional[str] = None,
        area_sqm: Optional[float] = None,
        max_capacity: Optional[int] = None,
        warning_density_p_sqm: Optional[float] = None,
        critical_density_p_sqm: Optional[float] = None,
        warning_occupancy_ratio: Optional[float] = None,
        critical_occupancy_ratio: Optional[float] = None,
        risk_weight: Optional[float] = None,
        scale_factor: Optional[float] = None,
        calibration_mode: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update calibration thresholds and physical dimensions of an existing zone."""
        zone = self.zones.get(zone_id)
        if not zone:
            return False

        if name is not None:
            zone.name = str(name)
        if zone_type is not None:
            zone.zone_type = str(zone_type)
        if area_sqm is not None and area_sqm > 0:
            zone.area_sqm = float(area_sqm)
            zone.is_calibrated = True
        if max_capacity is not None and max_capacity > 0:
            zone.max_capacity = int(max_capacity)
        if warning_density_p_sqm is not None and warning_density_p_sqm > 0:
            zone.warning_density_p_sqm = float(warning_density_p_sqm)
        if critical_density_p_sqm is not None and critical_density_p_sqm > 0:
            zone.critical_density_p_sqm = float(critical_density_p_sqm)
        if warning_occupancy_ratio is not None and 0.0 < warning_occupancy_ratio <= 1.0:
            zone.warning_occupancy_ratio = float(warning_occupancy_ratio)
        if critical_occupancy_ratio is not None and 0.0 < critical_occupancy_ratio <= 1.0:
            zone.critical_occupancy_ratio = float(critical_occupancy_ratio)
        if risk_weight is not None and risk_weight > 0:
            zone.risk_weight = float(risk_weight)
        if scale_factor is not None and scale_factor > 0:
            zone.scale_factor = float(scale_factor)
        if calibration_mode is not None:
            zone.calibration_mode = str(calibration_mode)
        if notes is not None:
            zone.notes = str(notes)

        return True

    def set_gps_polygon(self, zone_id: str, gps_points: List[Tuple[float, float]]) -> bool:
        """Assign or update real-world GPS coordinates for a zone."""
        if zone_id in self.zones:
            self.zones[zone_id].gps_polygon = gps_points
            return True
        return False

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self.zones:
            del self.zones[zone_id]
            return True
        return False

    def clear_zones(self) -> None:
        """Remove all configured zones."""
        self.zones.clear()

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

    def update_zone_metrics(
        self,
        zone_id: str,
        count: int,
        flow_vector: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Update count, flow vector, and compute density & risk status against calibrated thresholds."""
        zone = self.zones.get(zone_id)
        if not zone:
            return {}

        zone.current_count = int(count)
        if flow_vector is not None:
            zone.flow_vector = flow_vector

        if zone.is_calibrated and zone.area_m2 and zone.area_m2 > 0:
            zone.current_density_people_per_m2 = round(count / zone.area_m2, 2)
            density_mode = "calibrated_physical_density"
        else:
            zone.current_density_people_per_m2 = None
            density_mode = "uncalibrated_occupancy_estimate"

        # Occupancy ratio & dynamic calibrated threshold evaluation
        occ_ratio = count / max(zone.capacity, 1)
        density = zone.current_density_people_per_m2 or 0.0

        crit_dens = zone.critical_density_p_sqm
        warn_dens = zone.warning_density_p_sqm
        crit_occ = zone.critical_occupancy_ratio
        warn_occ = zone.warning_occupancy_ratio

        if density >= crit_dens or occ_ratio >= crit_occ:
            zone.risk_level = "critical"
        elif density >= warn_dens or occ_ratio >= warn_occ:
            zone.risk_level = "warning"
        else:
            zone.risk_level = "safe"

        return {
            "zone_id": zone_id,
            "name": zone.name,
            "zone_type": zone.zone_type,
            "count": zone.current_count,
            "current_count": zone.current_count,
            "capacity": zone.capacity,
            "max_capacity": zone.capacity,
            "occupancy_ratio": round(occ_ratio, 2),
            "density_people_per_m2": zone.current_density_people_per_m2,
            "density_p_sqm": zone.current_density_people_per_m2,
            "density_mode": density_mode,
            "is_calibrated": zone.is_calibrated,
            "area_m2": round(zone.area_m2, 2) if zone.area_m2 else None,
            "area_sqm": round(zone.area_sqm, 2),
            "risk_level": zone.risk_level,
            "risk_status": zone.risk_status,
            "flow_vector": zone.flow_vector,
            "gps_polygon": zone.gps_polygon,
            "warning_density_p_sqm": zone.warning_density_p_sqm,
            "critical_density_p_sqm": zone.critical_density_p_sqm,
            "risk_weight": zone.risk_weight,
        }

    def get_calibrated_zones_summary(self) -> List[Dict[str, Any]]:
        """Return a structured summary list of all active zones for dashboard/APIs."""
        summary = []
        for z in self.zones.values():
            occ_ratio = z.current_count / max(z.capacity, 1)
            summary.append({
                "zone_id": z.zone_id,
                "name": z.name,
                "zone_type": z.zone_type,
                "current_count": z.current_count,
                "max_capacity": z.capacity,
                "capacity": z.capacity,
                "occupancy_ratio": round(occ_ratio, 2),
                "density_p_sqm": z.current_density_people_per_m2,
                "area_sqm": z.area_sqm,
                "risk_status": z.risk_status,
                "risk_level": z.risk_level,
                "flow_vector": z.flow_vector,
                "is_calibrated": z.is_calibrated,
                "has_gps": bool(z.gps_polygon and len(z.gps_polygon) >= 3),
                "warning_density_p_sqm": z.warning_density_p_sqm,
                "critical_density_p_sqm": z.critical_density_p_sqm,
                "risk_weight": z.risk_weight,
            })
        return summary

    def export_calibrations_json(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Export all active zone calibrations to a serialized JSON string or file."""
        data = {
            "version": "1.0",
            "zones": [z.to_dict() for z in self.zones.values()]
        }
        json_str = json.dumps(data, indent=2)

        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")

        return json_str

    def import_calibrations_json(self, data: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Import zone calibrations from JSON string, dictionary, or list."""
        try:
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data

            zones_list = parsed.get("zones", parsed) if isinstance(parsed, dict) else parsed
            if not isinstance(zones_list, list):
                return False

            self.clear_zones()
            for z_data in zones_list:
                zone = ZoneDefinition.from_dict(z_data)
                self.zones[zone.zone_id] = zone
            return True
        except Exception as e:
            logger.warning(f"Failed to import zone calibrations: {e}")
            return False

    def save_to_file(self, filepath: Union[str, Path] = "data/calibrations/zones_calibration.json") -> bool:
        """Persist current zone calibrations to disk for session continuity."""
        try:
            self.export_calibrations_json(filepath=filepath)
            return True
        except Exception as e:
            logger.warning(f"Failed to save zone calibration file: {e}")
            return False

    def load_from_file(self, filepath: Union[str, Path] = "data/calibrations/zones_calibration.json") -> bool:
        """Load zone calibrations from persistent disk storage."""
        p = Path(filepath)
        if not p.exists():
            return False
        try:
            content = p.read_text(encoding="utf-8")
            return self.import_calibrations_json(content)
        except Exception as e:
            logger.warning(f"Failed to load zone calibration file: {e}")
            return False

    def to_geojson(self) -> dict:
        features = []
        for zone in self.zones.values():
            coords = zone.gps_polygon if zone.gps_polygon else (zone.world_polygon if zone.world_polygon else zone.image_polygon)
            if coords:
                if zone.gps_polygon:
                    geo_pts = [[p[1], p[0]] for p in coords]
                else:
                    geo_pts = [[p[0], p[1]] for p in coords]
                closed_coords = geo_pts + [geo_pts[0]]
                features.append({
                    "type": "Feature",
                    "properties": {
                        "zone_id": zone.zone_id,
                        "name": zone.name,
                        "zone_type": zone.zone_type,
                        "capacity": zone.capacity,
                        "max_capacity": zone.capacity,
                        "current_count": zone.current_count,
                        "area_m2": zone.area_m2,
                        "area_sqm": zone.area_sqm,
                        "density_p_sqm": zone.current_density_people_per_m2,
                        "is_calibrated": zone.is_calibrated,
                        "risk_level": zone.risk_level,
                        "risk_status": zone.risk_status,
                        "flow_vector": zone.flow_vector,
                        "warning_density_p_sqm": zone.warning_density_p_sqm,
                        "critical_density_p_sqm": zone.critical_density_p_sqm,
                        "risk_weight": zone.risk_weight,
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [closed_coords],
                    },
                })

        return {"type": "FeatureCollection", "features": features}
