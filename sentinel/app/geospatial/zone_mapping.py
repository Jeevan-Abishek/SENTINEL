"""
Zone Mapping Module.
Polygonal zone definitions, Shapely geometry validation/repair, and Folium geospatial map rendering.
"""

from app.mapping.zones import ZoneManager, PolygonZoneValidator, ZoneDefinition
from app.mapping.renderer import MapRenderer

__all__ = ["ZoneManager", "PolygonZoneValidator", "ZoneDefinition", "MapRenderer"]

