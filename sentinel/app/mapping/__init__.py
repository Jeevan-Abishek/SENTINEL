import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import cv2
import json


@dataclass
class CalibrationPoint:
    image_point: Tuple[float, float]  # (x, y) in pixels
    world_point: Tuple[float, float]  # (lat, lon) or (x, y) in meters


@dataclass
class CalibrationResult:
    homography_matrix: np.ndarray  # 3x3 matrix
    reprojection_error: float
    is_valid: bool
    world_bounds: Optional[Tuple[float, float, float, float]] = None  # min_x, min_y, max_x, max_y


class CoordinateCalibrator:
    def __init__(self, min_points: int = 4):
        self.min_points = min_points
        self.calibration_points: List[CalibrationPoint] = []
        self.homography_matrix: Optional[np.ndarray] = None
        self.world_bounds: Optional[Tuple[float, float, float, float]] = None

    def add_calibration_point(self, image_x: float, image_y: float, world_x: float, world_y: float):
        point = CalibrationPoint(
            image_point=(image_x, image_y),
            world_point=(world_x, world_y),
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
            
            if H is None:
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
                is_valid=mean_error < 10.0,  # Threshold in meters/pixels
                world_bounds=(min_x, min_y, max_x, max_y),
            )
        except Exception as e:
            return CalibrationResult(
                homography_matrix=np.eye(3),
                reprojection_error=float('inf'),
                is_valid=False,
            )

    def image_to_world(self, image_x: float, image_y: float) -> Optional[Tuple[float, float]]:
        if self.homography_matrix is None:
            return None
        
        point = np.array([[[image_x, image_y]]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(point, self.homography_matrix)
        return (float(world_point[0, 0, 0]), float(world_point[0, 0, 1]))

    def world_to_image(self, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        if self.homography_matrix is None:
            return None
        
        H_inv = np.linalg.inv(self.homography_matrix)
        point = np.array([[[world_x, world_y]]], dtype=np.float32)
        image_point = cv2.perspectiveTransform(point, H_inv)
        return (float(image_point[0, 0, 0]), float(image_point[0, 0, 1]))

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


class ZoneManager:
    def __init__(self):
        self.zones: Dict[str, dict] = {}
        self.calibrator: Optional[CoordinateCalibrator] = None

    def set_calibrator(self, calibrator: CoordinateCalibrator):
        self.calibrator = calibrator

    def add_zone(
        self,
        zone_id: str,
        name: str,
        zone_type: str,
        image_polygon: List[Tuple[float, float]],
        capacity: int = 0,
        entry_line: Optional[List[Tuple[float, float]]] = None,
        exit_line: Optional[List[Tuple[float, float]]] = None,
    ):
        world_polygon = None
        world_entry_line = None
        world_exit_line = None
        
        if self.calibrator and self.calibrator.homography_matrix is not None:
            world_polygon = [
                self.calibrator.image_to_world(x, y) for x, y in image_polygon
            ]
            world_polygon = [p for p in world_polygon if p is not None]
            
            if entry_line:
                world_entry_line = [
                    self.calibrator.image_to_world(x, y) for x, y in entry_line
                ]
                world_entry_line = [p for p in world_entry_line if p is not None]
            
            if exit_line:
                world_exit_line = [
                    self.calibrator.image_to_world(x, y) for x, y in exit_line
                ]
                world_exit_line = [p for p in world_exit_line if p is not None]
        
        self.zones[zone_id] = {
            "zone_id": zone_id,
            "name": name,
            "zone_type": zone_type,
            "image_polygon": image_polygon,
            "world_polygon": world_polygon,
            "capacity": capacity,
            "entry_line": entry_line,
            "world_entry_line": world_entry_line,
            "exit_line": exit_line,
            "world_exit_line": world_exit_line,
            "current_count": 0,
            "risk_level": "safe",
        }

    def remove_zone(self, zone_id: str):
        if zone_id in self.zones:
            del self.zones[zone_id]

    def get_zone(self, zone_id: str) -> Optional[dict]:
        return self.zones.get(zone_id)

    def get_all_zones(self) -> List[dict]:
        return list(self.zones.values())

    def update_zone_count(self, zone_id: str, count: int):
        if zone_id in self.zones:
            self.zones[zone_id]["current_count"] = count

    def update_zone_risk(self, zone_id: str, risk_level: str):
        if zone_id in self.zones:
            self.zones[zone_id]["risk_level"] = risk_level

    def is_point_in_zone(self, zone_id: str, world_x: float, world_y: float) -> bool:
        zone = self.zones.get(zone_id)
        if not zone or not zone.get("world_polygon"):
            return False
        
        from matplotlib.path import Path
        polygon = zone["world_polygon"]
        path = Path(polygon)
        return path.contains_point((world_x, world_y))

    def get_zone_at_world_point(self, world_x: float, world_y: float) -> Optional[str]:
        for zone_id, zone in self.zones.items():
            if self.is_point_in_zone(zone_id, world_x, world_y):
                return zone_id
        return None

    def to_geojson(self) -> dict:
        features = []
        for zone in self.zones.values():
            if zone.get("world_polygon"):
                feature = {
                    "type": "Feature",
                    "properties": {
                        "zone_id": zone["zone_id"],
                        "name": zone["name"],
                        "zone_type": zone["zone_type"],
                        "capacity": zone["capacity"],
                        "current_count": zone["current_count"],
                        "risk_level": zone["risk_level"],
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [zone["world_polygon"] + [zone["world_polygon"][0]]],
                    },
                }
                features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
        }


class MapRenderer:
    def __init__(self, center_lat: float = 0.0, center_lon: float = 0.0, zoom: int = 18):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.tile_layer = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

    def create_base_map(self) -> str:
        """Returns HTML for a Folium map."""
        import folium
        
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom,
            tiles=None,
        )
        
        folium.TileLayer(
            tiles=self.tile_layer,
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            name="OpenStreetMap",
        ).add_to(m)
        
        return m._repr_html_()

    def add_zones_to_map(self, map_obj, zones: List[dict], calibrator: Optional[CoordinateCalibrator] = None):
        import folium
        
        color_map = {
            "safe": "green",
            "warning": "orange",
            "critical": "red",
        }
        
        for zone in zones:
            if zone.get("world_polygon"):
                color = color_map.get(zone.get("risk_level", "safe"), "blue")
                
                folium.Polygon(
                    locations=[(p[1], p[0]) for p in zone["world_polygon"]],
                    color=color,
                    weight=2,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.3,
                    popup=f"{zone['name']} ({zone['current_count']}/{zone['capacity']})",
                    tooltip=f"{zone['name']} - {zone['risk_level']}",
                ).add_to(map_obj)
        
        return map_obj

    def add_drone_to_map(self, map_obj, drone_lat: float, drone_lon: float, heading: float = 0):
        import folium
        
        folium.Marker(
            location=[drone_lat, drone_lon],
            icon=folium.Icon(icon="plane", prefix="fa", color="blue"),
            popup="Drone",
            tooltip="Drone Position",
        ).add_to(map_obj)
        
        return map_obj

    def add_alerts_to_map(self, map_obj, alerts: List[dict], zones: List[dict]):
        import folium
        
        icon_map = {
            "warning": "exclamation-triangle",
            "critical": "exclamation-circle",
        }
        color_map = {
            "warning": "orange",
            "critical": "red",
        }
        
        for alert in alerts:
            zone = next((z for z in zones if z["zone_id"] == alert.get("zone_id")), None)
            if zone and zone.get("world_polygon"):
                # Use centroid of zone for alert marker
                centroid = np.mean(zone["world_polygon"], axis=0)
                level = alert.get("risk_level", "warning")
                
                folium.Marker(
                    location=[centroid[1], centroid[0]],
                    icon=folium.Icon(
                        icon=icon_map.get(level, "info-sign"),
                        prefix="fa",
                        color=color_map.get(level, "blue"),
                    ),
                    popup=f"{alert.get('message', 'Alert')}",
                    tooltip=f"{level.upper()}: {alert.get('zone_id')}",
                ).add_to(map_obj)
        
        return map_obj

    def add_heatmap_to_map(self, map_obj, grid_data: np.ndarray, bounds: Tuple[float, float, float, float]):
        import folium
        from folium.plugins import HeatMap
        
        min_x, min_y, max_x, max_y = bounds
        h, w = grid_data.shape
        
        heat_data = []
        for i in range(h):
            for j in range(w):
                if grid_data[i, j] > 0:
                    # Convert grid indices to world coordinates
                    world_x = min_x + (j + 0.5) * (max_x - min_x) / w
                    world_y = min_y + (i + 0.5) * (max_y - min_y) / h
                    heat_data.append([world_y, world_x, float(grid_data[i, j])])
        
        if heat_data:
            HeatMap(heat_data, radius=15, blur=10).add_to(map_obj)
        
        return map_obj