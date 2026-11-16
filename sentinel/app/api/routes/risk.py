from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import uuid

from app.database.repositories import RiskRepository

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])


class RiskFactorResponse(BaseModel):
    name: str
    value: float
    weight: float
    score: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


class RiskAssessmentResponse(BaseModel):
    id: str
    zone_id: str
    timestamp: datetime
    risk_level: str
    risk_score: float
    factors: List[RiskFactorResponse]
    recommended_actions: List[str]
    created_at: datetime


@router.get("", response_model=List[RiskAssessmentResponse])
async def list_risk_assessments(
    zone_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    risk_level: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
):
    risk_repo = RiskRepository()
    if zone_id:
        assessments = await risk_repo.get_by_zone(
            uuid.UUID(zone_id), start_time, end_time, limit
        )
    else:
        assessments = await risk_repo.list_recent(limit)
    
    if risk_level:
        assessments = [a for a in assessments if a.risk_level == risk_level]
    
    return [
        RiskAssessmentResponse(
            id=str(a.id),
            zone_id=str(a.zone_id),
            timestamp=a.timestamp,
            risk_level=a.risk_level,
            risk_score=a.risk_score,
            factors=[
                RiskFactorResponse(
                    name=f.name,
                    value=f.value,
                    weight=f.weight,
                    score=f.score,
                    threshold_warning=f.threshold_warning,
                    threshold_critical=f.threshold_critical,
                )
                for f in a.factors
            ],
            recommended_actions=a.recommended_actions,
            created_at=a.created_at,
        )
        for a in assessments
    ]


@router.get("/current/{zone_id}", response_model=RiskAssessmentResponse)
async def get_current_risk(zone_id: str):
    risk_repo = RiskRepository()
    assessment = await risk_repo.get_latest_for_zone(uuid.UUID(zone_id))
    if not assessment:
        raise HTTPException(status_code=404, detail="No risk assessment found for zone")
    return RiskAssessmentResponse(
        id=str(assessment.id),
        zone_id=str(assessment.zone_id),
        timestamp=assessment.timestamp,
        risk_level=assessment.risk_level,
        risk_score=assessment.risk_score,
        factors=[
            RiskFactorResponse(
                name=f.name,
                value=f.value,
                weight=f.weight,
                score=f.score,
                threshold_warning=f.threshold_warning,
                threshold_critical=f.threshold_critical,
            )
            for f in assessment.factors
        ],
        recommended_actions=assessment.recommended_actions,
        created_at=assessment.created_at,
    )


@router.get("/zones/summary")
async def risk_zones_summary():
    risk_repo = RiskRepository()
    summary = await risk_repo.get_zones_summary()
    return summary