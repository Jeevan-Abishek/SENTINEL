from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.api.dependencies import get_db
from app.database.repositories import (
    EventRepository, ZoneRepository, DroneRepository,
    DetectionRepository, TrackRepository, AnalyticsRepository,
    RiskRepository, AlertRepository, OperatorRepository
)

router = APIRouter(prefix="/api/v1/events", tags=["events"])


class EventCreate(BaseModel):
    name: str
    location: str
    start_time: datetime
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


@router.post("", response_model=EventResponse)
async def create_event(event: EventCreate):
    event_repo = EventRepository()
    new_event = await event_repo.create(
        name=event.name,
        location=event.location,
        start_time=event.start_time,
        end_time=event.end_time,
        description=event.description,
    )
    return EventResponse(
        id=str(new_event.id),
        name=new_event.name,
        location=new_event.location,
        status=new_event.status,
        start_time=new_event.start_time,
        end_time=new_event.end_time,
        description=new_event.description,
        created_at=new_event.created_at,
    )


@router.get("", response_model=List[EventResponse])
async def list_events(status: Optional[str] = Query(None)):
    event_repo = EventRepository()
    events = await event_repo.list_all(status=status)
    return [
        EventResponse(
            id=str(e.id),
            name=e.name,
            location=e.location,
            status=e.status,
            start_time=e.start_time,
            end_time=e.end_time,
            description=e.description,
            created_at=e.created_at,
        )
        for e in events
    ]


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: str):
    event_repo = EventRepository()
    event = await event_repo.get_by_id(uuid.UUID(event_id))
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return EventResponse(
        id=str(event.id),
        name=event.name,
        location=event.location,
        status=event.status,
        start_time=event.start_time,
        end_time=event.end_time,
        description=event.description,
        created_at=event.created_at,
    )


@router.patch("/{event_id}", response_model=EventResponse)
async def update_event(event_id: str, event: EventCreate):
    event_repo = EventRepository()
    updated = await event_repo.update(
        uuid.UUID(event_id),
        name=event.name,
        location=event.location,
        start_time=event.start_time,
        end_time=event.end_time,
        description=event.description,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Event not found")
    return EventResponse(
        id=str(updated.id),
        name=updated.name,
        location=updated.location,
        status=updated.status,
        start_time=updated.start_time,
        end_time=updated.end_time,
        description=updated.description,
        created_at=updated.created_at,
    )


@router.delete("/{event_id}")
async def delete_event(event_id: str):
    event_repo = EventRepository()
    deleted = await event_repo.delete(uuid.UUID(event_id))
    if not deleted:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted successfully"}