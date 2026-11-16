from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import DetectionRepository

router = APIRouter(prefix="/api/v1/detections", tags=["detections"])


class DetectionResponse(BaseModel):
    id: str
    drone_id: str
    frame_number: int
    timestamp: datetime
    num_detections: int
    detections_data: List[dict]
    processing_time_ms: float
    model_version: str
    created_at: datetime


@router.get("", response_model=List[DetectionResponse])
async def list_detections(
    drone_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
):
    det_repo = DetectionRepository()
    if drone_id:
        detections = await det_repo.get_by_drone(
            uuid.UUID(drone_id), start_time, end_time, limit
        )
    else:
        detections = await det_repo.list_recent(limit)
    return [
        DetectionResponse(
            id=str(d.id),
            drone_id=str(d.drone_id),
            frame_number=d.frame_number,
            timestamp=d.timestamp,
            num_detections=d.num_detections,
            detections_data=d.detections_data,
            processing_time_ms=d.processing_time_ms,
            model_version=d.model_version,
            created_at=d.created_at,
        )
        for d in detections
    ]


@router.get("/{detection_id}", response_model=DetectionResponse)
async def get_detection(detection_id: str):
    det_repo = DetectionRepository()
    detection = await det_repo.get_by_id(uuid.UUID(detection_id))
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")
    return DetectionResponse(
        id=str(detection.id),
        drone_id=str(detection.drone_id),
        frame_number=detection.frame_number,
        timestamp=detection.timestamp,
        num_detections=detection.num_detections,
        detections_data=detection.detections_data,
        processing_time_ms=detection.processing_time_ms,
        model_version=detection.model_version,
        created_at=detection.created_at,
    )