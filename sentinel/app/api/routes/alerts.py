from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.models import AlertStatus, AlertSeverity
from app.database.repositories import AlertRepository

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


class AlertCreate(BaseModel):
    event_id: str
    title: str
    message: str
    reason: str
    risk_level: str = "warning"
    zone_id: Optional[str] = None
    drone_id: Optional[str] = None
    risk_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    id: str
    event_id: str
    zone_id: Optional[str] = None
    risk_level: str
    title: str
    message: str
    reason: str
    risk_score: float
    timestamp: datetime
    acknowledged: bool
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any]
    created_at: datetime


class AlertAcknowledge(BaseModel):
    operator_id: Optional[str] = None


def _to_alert_response(alert) -> AlertResponse:
    risk_str = alert.risk_level.value if hasattr(alert.risk_level, "value") else str(alert.risk_level)
    is_ack = (alert.status == AlertStatus.ACKNOWLEDGED) if hasattr(alert, "status") else False
    return AlertResponse(
        id=str(alert.id),
        event_id=str(alert.event_id),
        zone_id=str(alert.zone_id) if alert.zone_id else None,
        risk_level=risk_str,
        title=alert.title,
        message=alert.message,
        reason=alert.reason,
        risk_score=alert.risk_score,
        timestamp=alert.timestamp,
        acknowledged=is_ack,
        acknowledged_by=str(alert.acknowledged_by) if alert.acknowledged_by else None,
        acknowledged_at=alert.acknowledged_at,
        metadata=getattr(alert, "alert_metadata", getattr(alert, "metadata", {})) or {},
        created_at=alert.created_at,
    )


@router.post("", response_model=AlertResponse)
async def create_alert(payload: AlertCreate):
    repo = AlertRepository()
    alert = await repo.create(
        event_id=uuid.UUID(payload.event_id),
        title=payload.title,
        message=payload.message,
        reason=payload.reason,
        risk_level=payload.risk_level,
        zone_id=uuid.UUID(payload.zone_id) if payload.zone_id else None,
        drone_id=uuid.UUID(payload.drone_id) if payload.drone_id else None,
        risk_score=payload.risk_score,
        alert_metadata=payload.metadata or {},
    )
    return _to_alert_response(alert)


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    event_id: Optional[str] = Query(None),
    zone_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
):
    alert_repo = AlertRepository()
    alerts = await alert_repo.list_filtered(
        event_id=uuid.UUID(event_id) if event_id else None,
        zone_id=uuid.UUID(zone_id) if zone_id else None,
        status=status,
        limit=limit,
    )
    return [_to_alert_response(a) for a in alerts]


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str):
    alert_repo = AlertRepository()
    alert = await alert_repo.get_by_id(uuid.UUID(alert_id))
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return _to_alert_response(alert)


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(alert_id: str, ack: Optional[AlertAcknowledge] = None):
    alert_repo = AlertRepository()
    op_id = None
    if ack and ack.operator_id:
        try:
            op_id = uuid.UUID(ack.operator_id)
        except ValueError:
            op_id = uuid.uuid5(uuid.NAMESPACE_DNS, ack.operator_id)

    alert = await alert_repo.acknowledge(uuid.UUID(alert_id), op_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return _to_alert_response(alert)


@router.get("/active/count")
async def active_alerts_count(zone_id: Optional[str] = Query(None)):
    alert_repo = AlertRepository()
    count = await alert_repo.count_active(
        zone_id=uuid.UUID(zone_id) if zone_id else None
    )
    return {"count": count}
