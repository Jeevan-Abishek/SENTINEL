from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import AnalyticsRepository

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


class AnalyticsResponse(BaseModel):
    id: str
    zone_id: str
    timestamp: datetime
    current_count: int
    unique_tracks: int
    entries: int
    exits: int
    density_per_sqm: float
    occupancy_ratio: float
    avg_speed: float
    flow_rate: float
    movement_direction: Optional[float] = None
    bottleneck_score: float
    raw_data: Optional[dict] = None
    created_at: datetime


class AnalyticsSummary(BaseModel):
    total_count: int
    active_zones: int
    safe_zones: int
    warning_zones: int
    critical_zones: int
    timestamp: datetime


@router.get("", response_model=List[AnalyticsResponse])
async def list_analytics(
    zone_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
):
    analytics_repo = AnalyticsRepository()
    if zone_id:
        analytics = await analytics_repo.get_by_zone(
            uuid.UUID(zone_id), start_time, end_time, limit
        )
    else:
        analytics = await analytics_repo.list_recent(limit)
    return [
        AnalyticsResponse(
            id=str(a.id),
            zone_id=str(a.zone_id),
            timestamp=a.timestamp,
            current_count=a.current_count,
            unique_tracks=a.unique_tracks,
            entries=a.entries,
            exits=a.exits,
            density_per_sqm=a.density_per_sqm,
            occupancy_ratio=a.occupancy_ratio,
            avg_speed=a.avg_speed,
            flow_rate=a.flow_rate,
            movement_direction=a.movement_direction,
            bottleneck_score=a.bottleneck_score,
            raw_data=a.raw_data,
            created_at=a.created_at,
        )
        for a in analytics
    ]


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary():
    analytics_repo = AnalyticsRepository()
    summary = await analytics_repo.get_summary()
    return AnalyticsSummary(**summary)


@router.get("/heatmap/{zone_id}")
async def get_heatmap(
    zone_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(10, le=100),
):
    analytics_repo = AnalyticsRepository()
    heatmaps = await analytics_repo.get_heatmaps(
        uuid.UUID(zone_id), start_time, end_time, limit
    )
    return [
        {
            "id": str(h.id),
            "zone_id": str(h.zone_id),
            "timestamp": h.timestamp,
            "grid_data": h.grid_data,
            "grid_width": h.grid_width,
            "grid_height": h.grid_height,
            "cell_size_meters": h.cell_size_meters,
            "max_density": h.max_density,
            "created_at": h.created_at,
        }
        for h in heatmaps
    ]