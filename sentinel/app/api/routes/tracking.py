from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import TrackRepository

router = APIRouter(prefix="/api/v1/tracking", tags=["tracking"])


class TrackResponse(BaseModel):
    id: str
    drone_id: str
    track_id: int
    class_id: int
    first_frame: int
    last_frame: int
    first_timestamp: datetime
    last_timestamp: datetime
    bbox_history: List[List[float]]
    centroid_history: List[List[float]]
    zone_transitions: Optional[List[dict]] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


@router.get("", response_model=List[TrackResponse])
async def list_tracks(
    drone_id: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(100, le=1000),
):
    track_repo = TrackRepository()
    if drone_id:
        tracks = await track_repo.get_by_drone(uuid.UUID(drone_id), active_only, limit)
    else:
        tracks = await track_repo.list_recent(active_only, limit)
    return [
        TrackResponse(
            id=str(t.id),
            drone_id=str(t.drone_id),
            track_id=t.track_id,
            class_id=t.class_id,
            first_frame=t.first_frame,
            last_frame=t.last_frame,
            first_timestamp=t.first_timestamp,
            last_timestamp=t.last_timestamp,
            bbox_history=t.bbox_history,
            centroid_history=t.centroid_history,
            zone_transitions=t.zone_transitions,
            is_active=t.is_active,
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in tracks
    ]


@router.get("/{track_id}", response_model=TrackResponse)
async def get_track(track_id: str):
    track_repo = TrackRepository()
    track = await track_repo.get_by_id(uuid.UUID(track_id))
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return TrackResponse(
        id=str(track.id),
        drone_id=str(track.drone_id),
        track_id=track.track_id,
        class_id=track.class_id,
        first_frame=track.first_frame,
        last_frame=track.last_frame,
        first_timestamp=track.first_timestamp,
        last_timestamp=track.last_timestamp,
        bbox_history=track.bbox_history,
        centroid_history=track.centroid_history,
        zone_transitions=track.zone_transitions,
        is_active=track.is_active,
        created_at=track.created_at,
        updated_at=track.updated_at,
    )