from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import ZoneRepository

router = APIRouter(prefix="/api/v1/zones", tags=["zones"])


class ZoneCreate(BaseModel):
    event_id: str
    name: str
    zone_type: str  # entry, exit, monitoring
    polygon: List[List[float]]
    capacity: int = 0


class ZoneResponse(BaseModel):
    id: str
    event_id: str
    name: str
    zone_type: str
    polygon: List[List[float]]
    capacity: int
    current_count: int = 0
    created_at: datetime


@router.post("", response_model=ZoneResponse)
async def create_zone(zone: ZoneCreate):
    zone_repo = ZoneRepository()
    new_zone = await zone_repo.create(
        event_id=uuid.UUID(zone.event_id),
        name=zone.name,
        zone_type=zone.zone_type,
        polygon=zone.polygon,
        capacity=zone.capacity,
    )
    return ZoneResponse(
        id=str(new_zone.id),
        event_id=str(new_zone.event_id),
        name=new_zone.name,
        zone_type=new_zone.zone_type,
        polygon=new_zone.polygon,
        capacity=new_zone.capacity,
        current_count=0,
        created_at=new_zone.created_at,
    )


@router.get("", response_model=List[ZoneResponse])
async def list_zones(event_id: Optional[str] = Query(None)):
    zone_repo = ZoneRepository()
    if event_id:
        zones = await zone_repo.get_by_event(uuid.UUID(event_id))
    else:
        zones = await zone_repo.list_all()
    return [
        ZoneResponse(
            id=str(z.id),
            event_id=str(z.event_id),
            name=z.name,
            zone_type=z.zone_type,
            polygon=z.polygon,
            capacity=z.capacity,
            current_count=0,  # Would be populated from analytics
            created_at=z.created_at,
        )
        for z in zones
    ]


@router.get("/{zone_id}", response_model=ZoneResponse)
async def get_zone(zone_id: str):
    zone_repo = ZoneRepository()
    zone = await zone_repo.get_by_id(uuid.UUID(zone_id))
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return ZoneResponse(
        id=str(zone.id),
        event_id=str(zone.event_id),
        name=zone.name,
        zone_type=zone.zone_type,
        polygon=zone.polygon,
        capacity=zone.capacity,
        current_count=0,
        created_at=zone.created_at,
    )


@router.patch("/{zone_id}", response_model=ZoneResponse)
async def update_zone(zone_id: str, zone: ZoneCreate):
    zone_repo = ZoneRepository()
    updated = await zone_repo.update(
        uuid.UUID(zone_id),
        name=zone.name,
        zone_type=zone.zone_type,
        polygon=zone.polygon,
        capacity=zone.capacity,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Zone not found")
    return ZoneResponse(
        id=str(updated.id),
        event_id=str(updated.event_id),
        name=updated.name,
        zone_type=updated.zone_type,
        polygon=updated.polygon,
        capacity=updated.capacity,
        current_count=0,
        created_at=updated.created_at,
    )


@router.delete("/{zone_id}")
async def delete_zone(zone_id: str):
    zone_repo = ZoneRepository()
    deleted = await zone_repo.delete(uuid.UUID(zone_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"message": "Zone deleted successfully"}