from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

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
    telemetry_port: int
    last_seen: Optional[datetime] = None
    created_at: datetime


@router.post("", response_model=DroneResponse)
async def create_drone(drone: DroneCreate):
    drone_repo = DroneRepository()
    new_drone = await drone_repo.create(
        event_id=uuid.UUID(drone.event_id),
        name=drone.name,
        rtsp_url=drone.rtsp_url,
        telemetry_port=drone.telemetry_port,
    )
    return DroneResponse(
        id=str(new_drone.id),
        event_id=str(new_drone.event_id),
        name=new_drone.name,
        status=new_drone.status,
        rtsp_url=new_drone.rtsp_url,
        telemetry_port=new_drone.telemetry_port,
        last_seen=new_drone.last_seen,
        created_at=new_drone.created_at,
    )


@router.get("", response_model=List[DroneResponse])
async def list_drones(event_id: Optional[str] = Query(None)):
    drone_repo = DroneRepository()
    if event_id:
        drones = await drone_repo.get_by_event(uuid.UUID(event_id))
    else:
        drones = await drone_repo.list_all()
    return [
        DroneResponse(
            id=str(d.id),
            event_id=str(d.event_id),
            name=d.name,
            status=d.status,
            rtsp_url=d.rtsp_url,
            telemetry_port=d.telemetry_port,
            last_seen=d.last_seen,
            created_at=d.created_at,
        )
        for d in drones
    ]


@router.get("/{drone_id}", response_model=DroneResponse)
async def get_drone(drone_id: str):
    drone_repo = DroneRepository()
    drone = await drone_repo.get_by_id(uuid.UUID(drone_id))
    if not drone:
        raise HTTPException(status_code=404, detail="Drone not found")
    return DroneResponse(
        id=str(drone.id),
        event_id=str(drone.event_id),
        name=drone.name,
        status=drone.status,
        rtsp_url=drone.rtsp_url,
        telemetry_port=drone.telemetry_port,
        last_seen=drone.last_seen,
        created_at=drone.created_at,
    )


@router.patch("/{drone_id}", response_model=DroneResponse)
async def update_drone(drone_id: str, drone: DroneCreate):
    drone_repo = DroneRepository()
    updated = await drone_repo.update(
        uuid.UUID(drone_id),
        name=drone.name,
        rtsp_url=drone.rtsp_url,
        telemetry_port=drone.telemetry_port,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Drone not found")
    return DroneResponse(
        id=str(updated.id),
        event_id=str(updated.event_id),
        name=updated.name,
        status=updated.status,
        rtsp_url=updated.rtsp_url,
        telemetry_port=updated.telemetry_port,
        last_seen=updated.last_seen,
        created_at=updated.created_at,
    )


@router.delete("/{drone_id}")
async def delete_drone(drone_id: str):
    drone_repo = DroneRepository()
    deleted = await drone_repo.delete(uuid.UUID(drone_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Drone not found")
    return {"message": "Drone deleted successfully"}