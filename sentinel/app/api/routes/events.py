from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import EventRepository

router = APIRouter(prefix="/api/v1/events", tags=["events"])


class EventCreate(BaseModel):
    name: str
    location: str
    latitude: float = 0.0
    longitude: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    description: Optional[str] = None


class EventResponse(BaseModel):
    id: str
    name: str
    location: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    description: Optional[str] = None
    created_at: datetime


def _to_event_response(event) -> EventResponse:
    status_str = event.status.value if hasattr(event.status, "value") else str(event.status)
    return EventResponse(
        id=str(event.id),
        name=event.name,
        location=getattr(event, "location_name", getattr(event, "location", "")),
        status=status_str,
        start_time=event.start_time,
        end_time=event.end_time,
        description=event.description,
        created_at=event.created_at,
    )


@router.post("", response_model=EventResponse)
async def create_event(event: EventCreate):
    event_repo = EventRepository()
    new_event = await event_repo.create(
        name=event.name,
        location=event.location,
        latitude=event.latitude,
        longitude=event.longitude,
        start_time=event.start_time,
        end_time=event.end_time,
        description=event.description,
    )
    return _to_event_response(new_event)


@router.get("", response_model=List[EventResponse])
async def list_events(status: Optional[str] = Query(None)):
    event_repo = EventRepository()
    events = await event_repo.list_all(status=status)
    return [_to_event_response(e) for e in events]


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: str):
    event_repo = EventRepository()
    event = await event_repo.get_by_id(uuid.UUID(event_id))
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return _to_event_response(event)


@router.patch("/{event_id}", response_model=EventResponse)
async def update_event(event_id: str, event: EventCreate):
    event_repo = EventRepository()
    updated = await event_repo.update(
        uuid.UUID(event_id),
        name=event.name,
        location=event.location,
        latitude=event.latitude,
        longitude=event.longitude,
        start_time=event.start_time,
        end_time=event.end_time,
        description=event.description,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Event not found")
    return _to_event_response(updated)


@router.delete("/{event_id}")
async def delete_event(event_id: str):
    event_repo = EventRepository()
    deleted = await event_repo.delete(uuid.UUID(event_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted successfully"}
