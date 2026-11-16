from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import (
    EventRepository, ZoneRepository, AnalyticsRepository,
    RiskRepository, AlertRepository
)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


class EventReport(BaseModel):
    event_id: str
    event_name: str
    start_time: datetime
    end_time: Optional[datetime]
    total_people_count: int
    peak_count: int
    avg_density: float
    max_risk_level: str
    total_alerts: int
    zones_summary: List[dict]


class ZoneReport(BaseModel):
    zone_id: str
    zone_name: str
    zone_type: str
    capacity: int
    avg_count: float
    max_count: int
    avg_occupancy: float
    avg_density: float
    entries: int
    exits: int
    max_risk_level: str
    total_alerts: int
    bottleneck_incidents: int


@router.get("/event/{event_id}", response_model=EventReport)
async def get_event_report(
    event_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
):
    event_repo = EventRepository()
    event = await event_repo.get_by_id(uuid.UUID(event_id))
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    zone_repo = ZoneRepository()
    zones = await zone_repo.get_by_event(uuid.UUID(event_id))
    
    analytics_repo = AnalyticsRepository()
    risk_repo = RiskRepository()
    alert_repo = AlertRepository()
    
    total_people = 0
    peak_count = 0
    total_density = 0
    density_count = 0
    max_risk = "safe"
    total_alerts = 0
    zones_summary = []
    
    risk_order = {"safe": 0, "warning": 1, "critical": 2}
    
    for zone in zones:
        analytics = await analytics_repo.get_by_zone(
            zone.id, start_time, end_time, limit=1000
        )
        risk_assessments = await risk_repo.get_by_zone(
            zone.id, start_time, end_time, limit=1000
        )
        alerts = await alert_repo.list_filtered(
            zone_id=zone.id, start_time=start_time, end_time=end_time, limit=1000
        )
        
        zone_count = sum(a.current_count for a in analytics)
        zone_peak = max((a.current_count for a in analytics), default=0)
        zone_density = sum(a.density_per_sqm for a in analytics) / len(analytics) if analytics else 0
        zone_risk = max((r.risk_level for r in risk_assessments), default="safe", key=lambda x: risk_order.get(x, 0))
        zone_alerts = len(alerts)
        
        total_people += zone_count
        peak_count = max(peak_count, zone_peak)
        total_density += zone_density
        density_count += 1
        if risk_order.get(zone_risk, 0) > risk_order.get(max_risk, 0):
            max_risk = zone_risk
        total_alerts += zone_alerts
        
        zones_summary.append({
            "zone_id": str(zone.id),
            "zone_name": zone.name,
            "avg_count": zone_count / len(analytics) if analytics else 0,
            "peak_count": zone_peak,
            "avg_density": zone_density,
            "risk_level": zone_risk,
            "alerts_count": zone_alerts,
        })
    
    return EventReport(
        event_id=str(event.id),
        event_name=event.name,
        start_time=event.start_time,
        end_time=event.end_time,
        total_people_count=total_people,
        peak_count=peak_count,
        avg_density=total_density / density_count if density_count > 0 else 0,
        max_risk_level=max_risk,
        total_alerts=total_alerts,
        zones_summary=zones_summary,
    )


@router.get("/zone/{zone_id}", response_model=ZoneReport)
async def get_zone_report(
    zone_id: str,
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
):
    zone_repo = ZoneRepository()
    zone = await zone_repo.get_by_id(uuid.UUID(zone_id))
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    analytics_repo = AnalyticsRepository()
    risk_repo = RiskRepository()
    alert_repo = AlertRepository()
    
    analytics = await analytics_repo.get_by_zone(
        uuid.UUID(zone_id), start_time, end_time, limit=1000
    )
    risk_assessments = await risk_repo.get_by_zone(
        uuid.UUID(zone_id), start_time, end_time, limit=1000
    )
    alerts = await alert_repo.list_filtered(
        zone_id=uuid.UUID(zone_id), start_time=start_time, end_time=end_time, limit=1000
    )
    
    avg_count = sum(a.current_count for a in analytics) / len(analytics) if analytics else 0
    max_count = max((a.current_count for a in analytics), default=0)
    avg_occupancy = sum(a.occupancy_ratio for a in analytics) / len(analytics) if analytics else 0
    avg_density = sum(a.density_per_sqm for a in analytics) / len(analytics) if analytics else 0
    entries = sum(a.entries for a in analytics)
    exits = sum(a.exits for a in analytics)
    max_risk = max((r.risk_level for r in risk_assessments), default="safe")
    total_alerts = len(alerts)
    bottleneck_incidents = sum(1 for r in risk_assessments if r.bottleneck_score > 0.5)
    
    return ZoneReport(
        zone_id=str(zone.id),
        zone_name=zone.name,
        zone_type=zone.zone_type,
        capacity=zone.capacity,
        avg_count=avg_count,
        max_count=max_count,
        avg_occupancy=avg_occupancy,
        avg_density=avg_density,
        entries=entries,
        exits=exits,
        max_risk_level=max_risk,
        total_alerts=total_alerts,
        bottleneck_incidents=bottleneck_incidents,
    )