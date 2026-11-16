from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


router = APIRouter(prefix="/mapping", tags=["Mapping"])


class ZonePolygonRequest(BaseModel):
    zone_id: str
    name: str
    zone_type: str = Field(..., description="monitoring, entry, exit, emergency_exit, restricted")
    image_polygon: List[List[float]] = Field(..., description="List of [x, y] points in image coordinates")
    capacity: int = Field(default=0, description="Maximum capacity for this zone")
    entry_line: Optional[List[List[float]]] = Field(None, description="Entry line [x1,y1, x2,y2] in image coordinates")
    exit_line: Optional[List[List[float]]] = Field(None, description="Exit line [x1,y1, x2,y2] in image coordinates")


class ZonePolygonResponse(BaseModel):
    zone_id: str
    name: str
    zone_type: str
    image_polygon: List[List[float]]
    world_polygon: Optional[List[List[float]]] = None
    capacity: int
    entry_line: Optional[List[List[float]]] = None
    world_entry_line: Optional[List[List[float]]] = None
    exit_line: Optional[List[List[float]]] = None
    world_exit_line: Optional[List[List[float]]] = None
    current_count: int = 0
    risk_level: str = "safe"


class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[Dict[str, Any]]


class MapConfigRequest(BaseModel):
    center_lat: float = Field(default=0.0)
    center_lon: float = Field(default=0.0)
    zoom: int = Field(default=18, ge=1, le=22)
    tile_layer: str = Field(default="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png")


class DronePositionRequest(BaseModel):
    drone_id: str
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    heading: float = Field(default=0.0)
    timestamp: float


class AlertMarkerRequest(BaseModel):
    alert_id: str
    zone_id: str
    risk_level: str
    message: str
    timestamp: float


# In-memory storage for zones (replace with database in production)
_zones: Dict[str, Dict[str, Any]] = {}
_map_configs: Dict[str, MapConfigRequest] = {}
_drone_positions: Dict[str, DronePositionRequest] = {}
_alert_markers: Dict[str, AlertMarkerRequest] = {}


@router.post("/{event_id}/zones", response_model=ZonePolygonResponse, status_code=status.HTTP_201_CREATED)
async def create_zone(event_id: str, zone: ZonePolygonRequest):
    # Transform image polygon to world coordinates if calibration exists
    from app.api.routes.calibration import _calibrations
    import numpy as np
    import cv2
    
    world_polygon = None
    world_entry_line = None
    world_exit_line = None
    
    if event_id in _calibrations and _calibrations[event_id].get("homography"):
        calib = _calibrations[event_id]
        H = np.array(calib["homography"], dtype=np.float32)
        
        # Transform polygon
        image_poly = np.array(zone.image_polygon, dtype=np.float32).reshape(-1, 1, 2)
        world_poly = cv2.perspectiveTransform(image_poly, H).reshape(-1, 2)
        world_polygon = world_poly.tolist()
        
        # Transform entry line
        if zone.entry_line:
            entry = np.array(zone.entry_line, dtype=np.float32).reshape(-1, 1, 2)
            world_entry = cv2.perspectiveTransform(entry, H).reshape(-1, 2)
            world_entry_line = world_entry.tolist()
        
        # Transform exit line
        if zone.exit_line:
            exit_line = np.array(zone.exit_line, dtype=np.float32).reshape(-1, 1, 2)
            world_exit = cv2.perspectiveTransform(exit_line, H).reshape(-1, 2)
            world_exit_line = world_exit.tolist()
    
    zone_data = zone.model_dump()
    zone_data.update({
        "world_polygon": world_polygon,
        "world_entry_line": world_entry_line,
        "world_exit_line": world_exit_line,
        "current_count": 0,
        "risk_level": "safe",
    })
    
    key = f"{event_id}:{zone.zone_id}"
    _zones[key] = zone_data
    
    return ZonePolygonResponse(**zone_data)


@router.get("/{event_id}/zones", response_model=List[ZonePolygonResponse])
async def get_zones(event_id: str):
    result = []
    for key, zone in _zones.items():
        if key.startswith(f"{event_id}:"):
            result.append(ZonePolygonResponse(**zone))
    return result


@router.get("/{event_id}/zones/{zone_id}", response_model=ZonePolygonResponse)
async def get_zone(event_id: str, zone_id: str):
    key = f"{event_id}:{zone_id}"
    if key not in _zones:
        raise HTTPException(status_code=404, detail="Zone not found")
    return ZonePolygonResponse(**_zones[key])


@router.put("/{event_id}/zones/{zone_id}", response_model=ZonePolygonResponse)
async def update_zone(event_id: str, zone_id: str, zone: ZonePolygonRequest):
    key = f"{event_id}:{zone_id}"
    if key not in _zones:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    # Transform to world coordinates if calibration exists
    from app.api.routes.calibration import _calibrations
    import numpy as np
    import cv2
    
    world_polygon = None
    world_entry_line = None
    world_exit_line = None
    
    if event_id in _calibrations and _calibrations[event_id].get("homography"):
        calib = _calibrations[event_id]
        H = np.array(calib["homography"], dtype=np.float32)
        
        image_poly = np.array(zone.image_polygon, dtype=np.float32).reshape(-1, 1, 2)
        world_poly = cv2.perspectiveTransform(image_poly, H).reshape(-1, 2)
        world_polygon = world_poly.tolist()
        
        if zone.entry_line:
            entry = np.array(zone.entry_line, dtype=np.float32).reshape(-1, 1, 2)
            world_entry = cv2.perspectiveTransform(entry, H).reshape(-1, 2)
            world_entry_line = world_entry.tolist()
        
        if zone.exit_line:
            exit_line = np.array(zone.exit_line, dtype=np.float32).reshape(-1, 1, 2)
            world_exit = cv2.perspectiveTransform(exit_line, H).reshape(-1, 2)
            world_exit_line = world_exit.tolist()
    
    existing = _zones[key]
    zone_data = zone.model_dump()
    zone_data.update({
        "world_polygon": world_polygon,
        "world_entry_line": world_entry_line,
        "world_exit_line": world_exit_line,
        "current_count": existing.get("current_count", 0),
        "risk_level": existing.get("risk_level", "safe"),
    })
    
    _zones[key] = zone_data
    return ZonePolygonResponse(**zone_data)


@router.delete("/{event_id}/zones/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_zone(event_id: str, zone_id: str):
    key = f"{event_id}:{zone_id}"
    if key not in _zones:
        raise HTTPException(status_code=404, detail="Zone not found")
    del _zones[key]


@router.get("/{event_id}/geojson", response_model=GeoJSONResponse)
async def get_geojson(event_id: str):
    features = []
    for key, zone in _zones.items():
        if key.startswith(f"{event_id}:") and zone.get("world_polygon"):
            polygon = zone["world_polygon"]
            # Close the polygon
            coords = polygon + [polygon[0]]
            
            feature = {
                "type": "Feature",
                "properties": {
                    "zone_id": zone["zone_id"],
                    "name": zone["name"],
                    "zone_type": zone["zone_type"],
                    "capacity": zone["capacity"],
                    "current_count": zone.get("current_count", 0),
                    "risk_level": zone.get("risk_level", "safe"),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            }
            features.append(feature)
    
    return GeoJSONResponse(features=features)


@router.post("/{event_id}/config", response_model=MapConfigRequest)
async def set_map_config(event_id: str, config: MapConfigRequest):
    _map_configs[event_id] = config
    return config


@router.get("/{event_id}/config", response_model=MapConfigRequest)
async def get_map_config(event_id: str):
    if event_id not in _map_configs:
        # Return default config
        return MapConfigRequest()
    return _map_configs[event_id]


@router.post("/{event_id}/drone_position", response_model=DronePositionRequest)
async def set_drone_position(event_id: str, position: DronePositionRequest):
    key = f"{event_id}:{position.drone_id}"
    _drone_positions[key] = position
    return position


@router.get("/{event_id}/drone_position/{drone_id}", response_model=Optional[DronePositionRequest])
async def get_drone_position(event_id: str, drone_id: str):
    key = f"{event_id}:{drone_id}"
    return _drone_positions.get(key)


@router.post("/{event_id}/alert_markers", response_model=AlertMarkerRequest)
async def add_alert_marker(event_id: str, marker: AlertMarkerRequest):
    key = f"{event_id}:{marker.alert_id}"
    _alert_markers[key] = marker
    return marker


@router.get("/{event_id}/alert_markers", response_model=List[AlertMarkerRequest])
async def get_alert_markers(event_id: str):
    result = []
    for key, marker in _alert_markers.items():
        if key.startswith(f"{event_id}:"):
            result.append(marker)
    return result


@router.delete("/{event_id}/alert_markers/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_alert_marker(event_id: str, alert_id: str):
    key = f"{event_id}:{alert_id}"
    if key in _alert_markers:
        del _alert_markers[key]


@router.post("/{event_id}/point_in_zone", response_model=Optional[str])
async def get_zone_at_point(
    event_id: str,
    world_x: float = Query(..., description="World X coordinate (meters or lat)"),
    world_y: float = Query(..., description="World Y coordinate (meters or lon)"),
):
    from matplotlib.path import Path
    
    for key, zone in _zones.items():
        if key.startswith(f"{event_id}:") and zone.get("world_polygon"):
            polygon = zone["world_polygon"]
            path = Path(polygon)
            if path.contains_point((world_x, world_y)):
                return zone["zone_id"]
    return None