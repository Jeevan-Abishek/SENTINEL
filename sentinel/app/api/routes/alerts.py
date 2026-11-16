from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import AlertRepository

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


class AlertResponse(BaseModel):
    id: str
    zone_id: str
    risk_level: str
    message: str
    timestamp: datetime
    acknowledged: bool
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: dict
    created_at: datetime


class AlertAcknowledge(BaseModel):
    operator_id: str


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    zone_id: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
):
    alert_repo = AlertRepository()
    alerts = await alert_repo.list_filtered(
        zone_id=uuid.UUID(zone_id) if zone_id else None,
        risk_level=risk_level,
        acknowledged=acknowledged,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
    )
    return [
        AlertResponse(
            id=str(a.id),
            zone_id=str(a.zone_id),
            risk_level=a.risk_level,
            message=a.message,
            timestamp=a.timestamp,
            acknowledged=a.acknowledged,
            acknowledged_by=a.acknowledged_by,
            acknowledged_at=a.acknowledged_at,
            metadata=a.metadata,
            created_at=a.created_at,
        )
        for a in alerts
    ]


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str):
    alert_repo = AlertRepository()
    alert = await alert_repo.get_by_id(uuid.UUID(alert_id))
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AlertResponse(
        id=str(alert.id),
        zone_id=str(alert.zone_id),
        risk_level=alert.risk_level,
        message=alert.message,
        timestamp=alert.timestamp,
        acknowledged=alert.acknowledged,
        acknowledged_by=alert.acknowledged_by,
        acknowledged_at=alert.acknowledged_at,
        metadata=alert.metadata,
        created_at=alert.created_at,
    )


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(alert_id: str, ack: AlertAcknowledge):
    alert_repo = AlertRepository()
    alert = await alert_repo.acknowledge(uuid.UUID(alert_id), ack.operator_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return AlertResponse(
        id=str(alert.id),
        zone_id=str(alert.zone_id),
        risk_level=alert.risk_level,
        message=alert.message,
        timestamp=alert.timestamp,
        acknowledged=alert.acknowledged,
        acknowledged_by=alert.acknowledged_by,
        acknowledged_at=alert.acknowledged_at,
        metadata=alert.metadata,
        created_at=alert.created_at,
    )


@router.get("/active/count")
async def active_alerts_count(zone_id: Optional[str] = Query(None)):
    alert_repo = AlertRepository()
    count = await alert_repo.count_active(
        zone_id=uuid.UUID(zone_id) if zone_id else None
    )
    return {"count": count}