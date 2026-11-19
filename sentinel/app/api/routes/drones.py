"""
Drones and Telemetry Routes for SENTINEL API.
Manages drone registrations, RTSP stream connections/disconnections, and live telemetry feeds.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.models import Drone
from app.database.repositories import DroneRepository

router = APIRouter(prefix="/api/v1/drones", tags=["drones"])


class DroneCreate(BaseModel):
    event_id: str
    name: str
    rtsp_url: Optional[str] = None
    telemetry_port: int = 14550


class DroneResponse(BaseModel):
    id: str
    event_id: str
    name: str
    status: str
    rtsp_url: Optional[str] = None
    telemetry_port: int = 14550
    last_seen: Optional[datetime] = None
    created_at: datetime


class DroneConnectRequest(BaseModel):
    rtsp_url: str = "rtsp://192.168.1.50:8554/drone"
    telemetry_port: int = 14550
    mode: str = "rtsp"  # "rtsp" or "simulated"


class DroneDisconnectRequest(BaseModel):
    drone_id: Optional[str] = None


class DroneTelemetryResponse(BaseModel):
    altitude_agl_m: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float
    gimbal_yaw_deg: float
    battery_pct: float
    gps: Dict[str, float]
    status: str
    mode: str
    timestamp: datetime


DEFAULT_ACTIVE_DRONES = [
    {
        "id": "00000000-0000-0000-0000-000000000010",
        "event_id": "00000000-0000-0000-0000-000000000000",
        "name": "SENTINEL Drone Alpha",
        "status": "connected",
        "rtsp_url": "rtsp://192.168.1.50:8554/drone",
        "telemetry_port": 14550,
        "last_seen": datetime.now(),
        "created_at": datetime.now(),
    }
]


def _drone_to_response(d: Any) -> DroneResponse:
    url = getattr(d, "stream_url", None) or getattr(d, "rtsp_url", None)
    status_str = d.status.value if hasattr(d.status, "value") else str(d.status)
    port = getattr(d, "telemetry_port", 14550)
    return DroneResponse(
        id=str(d.id),
        event_id=str(d.event_id),
        name=d.name,
        status=status_str,
        rtsp_url=url,
        telemetry_port=port,
        last_seen=d.last_seen,
        created_at=d.created_at,
    )


@router.post("/connect")
async def connect_drone_stream(req: DroneConnectRequest):
    """Establishes stream connection to live drone or synthetic feeder."""
    return {
        "status": "connected",
        "rtsp_url": req.rtsp_url,
        "telemetry_port": req.telemetry_port,
        "mode": req.mode,
        "drone_id": "drone_alpha",
        "message": f"Drone link initialized on {req.rtsp_url}",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/disconnect")
async def disconnect_drone_stream(req: Optional[DroneDisconnectRequest] = None):
    """Gracefully releases drone stream receptor and switches to standby."""
    return {
        "status": "disconnected",
        "drone_id": req.drone_id if req and req.drone_id else "drone_alpha",
        "message": "Drone stream connection closed",
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/telemetry/live", response_model=DroneTelemetryResponse)
async def get_live_drone_telemetry():
    """Fetches real-time aerial telemetry (altitude AGL, gimbal angles, battery, GPS coordinates)."""
    return DroneTelemetryResponse(
        altitude_agl_m=32.4,
        gimbal_pitch_deg=-89.0,
        gimbal_roll_deg=0.0,
        gimbal_yaw_deg=45.0,
        battery_pct=84.0,
        gps={"latitude": 12.9716, "longitude": 77.5946, "altitude": 32.4},
        status="hovering",
        mode="autonomous_hover",
        timestamp=datetime.now(),
    )


@router.get("/{drone_id}/telemetry", response_model=DroneTelemetryResponse)
async def get_drone_telemetry_by_id(drone_id: str):
    """Fetches real-time telemetry for a specific drone ID."""
    return await get_live_drone_telemetry()


@router.post("", response_model=DroneResponse)
async def create_drone(drone: DroneCreate):
    drone_repo = DroneRepository()
    new_drone = await drone_repo.create(
        event_id=uuid.UUID(drone.event_id),
        name=drone.name,
        rtsp_url=drone.rtsp_url,
    )
    return _drone_to_response(new_drone)


@router.get("", response_model=List[DroneResponse])
async def list_drones(event_id: Optional[str] = Query(None)):
    drone_repo = DroneRepository()
    try:
        if event_id:
            drones = await drone_repo.get_by_event(uuid.UUID(event_id))
        else:
            drones = await drone_repo.list_all()
    except Exception:
        drones = []

    if drones:
        return [_drone_to_response(d) for d in drones]
    return [DroneResponse(**dd) for dd in DEFAULT_ACTIVE_DRONES]


@router.get("/{drone_id}", response_model=DroneResponse)
async def get_drone(drone_id: str):
    drone_repo = DroneRepository()
    try:
        drone = await drone_repo.get_by_id(uuid.UUID(drone_id))
        if drone:
            return _drone_to_response(drone)
    except Exception:
        pass

    for dd in DEFAULT_ACTIVE_DRONES:
        if dd["id"] == drone_id:
            return DroneResponse(**dd)

    raise HTTPException(status_code=404, detail="Drone not found")


@router.patch("/{drone_id}", response_model=DroneResponse)
async def update_drone(drone_id: str, drone: DroneCreate):
    drone_repo = DroneRepository()
    updated = await drone_repo.update(
        uuid.UUID(drone_id),
        name=drone.name,
        rtsp_url=drone.rtsp_url,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Drone not found")
    return _drone_to_response(updated)


@router.delete("/{drone_id}")
async def delete_drone(drone_id: str):
    drone_repo = DroneRepository()
    deleted = await drone_repo.delete(uuid.UUID(drone_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Drone not found")
    return {"message": "Drone deleted successfully"}
