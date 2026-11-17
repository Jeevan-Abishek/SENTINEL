"""Map rendering and visualization using Folium."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import folium
from folium.plugins import HeatMap


class MapRenderer:
    def __init__(self, center_lat: float = 0.0, center_lon: float = 0.0, zoom: int = 18):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.tile_layer = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"

    def create_base_map(self) -> str:
        """Returns HTML for a Folium map."""
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

    def add_zones_to_map(self, map_obj, zones: List[Any], color_map: Optional[Dict[str, str]] = None):
        colors = color_map or {
            "safe": "green",
            "warning": "orange",
            "critical": "red",
        }
        for zone in zones:
            # Support both dict and ZoneDefinition object
            world_poly = getattr(zone, "world_polygon", None) or (zone.get("world_polygon") if isinstance(zone, dict) else None)
            risk = getattr(zone, "risk_level", "safe") if hasattr(zone, "risk_level") else (zone.get("risk_level", "safe") if isinstance(zone, dict) else "safe")
            name = getattr(zone, "name", "Zone") if hasattr(zone, "name") else (zone.get("name", "Zone") if isinstance(zone, dict) else "Zone")
            count = getattr(zone, "current_count", 0) if hasattr(zone, "current_count") else (zone.get("current_count", 0) if isinstance(zone, dict) else 0)
            capacity = getattr(zone, "capacity", 100) if hasattr(zone, "capacity") else (zone.get("capacity", 100) if isinstance(zone, dict) else 100)

            if world_poly:
                c = colors.get(risk, "blue")
                folium.Polygon(
                    locations=[(p[1], p[0]) for p in world_poly],
                    color=c,
                    weight=2,
                    fill=True,
                    fill_color=c,
                    fill_opacity=0.3,
                    popup=f"{name} ({count}/{capacity})",
                    tooltip=f"{name} - {risk}",
                ).add_to(map_obj)
        return map_obj

    def add_drone_to_map(self, map_obj, drone_lat: float, drone_lon: float, heading: float = 0):
        folium.Marker(
            location=[drone_lat, drone_lon],
            icon=folium.Icon(icon="plane", prefix="fa", color="blue"),
            popup="Drone",
            tooltip="Drone Position",
        ).add_to(map_obj)
        return map_obj
