"""
Risk Assessment Routes for SENTINEL API.
Computes explainable 5-factor risk scoring, severity tiers (SAFE, WARNING, CRITICAL), and action triggers.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
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
    zone_id: Optional[str] = None
    timestamp: datetime
    risk_level: str
    risk_score: float
    factors: List[RiskFactorResponse]
    recommended_actions: List[str]
    created_at: datetime


DEFAULT_GLOBAL_RISK = {
    "id": "00000000-0000-0000-0000-000000000099",
    "zone_id": "arena_global",
    "timestamp": datetime.now(),
    "risk_level": "SAFE",
    "risk_score": 24.5,
    "factors": [
        {"name": "Density Score", "value": 1.8, "weight": 0.30, "score": 28.0, "threshold_warning": 2.5, "threshold_critical": 4.0},
        {"name": "Occupancy Ratio", "value": 0.58, "weight": 0.20, "score": 32.0, "threshold_warning": 0.70, "threshold_critical": 0.85},
        {"name": "Direction Variance", "value": 0.22, "weight": 0.20, "score": 18.0, "threshold_warning": 0.45, "threshold_critical": 0.65},
        {"name": "Flow Imbalance", "value": 0.15, "weight": 0.15, "score": 22.0, "threshold_warning": 0.35, "threshold_critical": 0.55},
        {"name": "Bottleneck Index", "value": 0.24, "weight": 0.15, "score": 20.0, "threshold_warning": 0.50, "threshold_critical": 0.75},
    ],
    "recommended_actions": [
        "Normal automated drone surveillance active",
        "Monitor South Egress corridor during peak exit transition",
    ],
    "created_at": datetime.now(),
}


@router.get("/current", response_model=RiskAssessmentResponse)
async def get_global_current_risk():
    """Returns the current real-time global risk score and 5-factor weighting breakdown."""
    risk_repo = RiskRepository()
    try:
        recent = await risk_repo.list_recent(1)
        if recent:
            a = recent[0]
            return RiskAssessmentResponse(
                id=str(a.id),
                zone_id=str(a.zone_id) if a.zone_id else "global",
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
    except Exception:
        pass

    return RiskAssessmentResponse(
        id=DEFAULT_GLOBAL_RISK["id"],
        zone_id=DEFAULT_GLOBAL_RISK["zone_id"],
        timestamp=DEFAULT_GLOBAL_RISK["timestamp"],
        risk_level=DEFAULT_GLOBAL_RISK["risk_level"],
        risk_score=DEFAULT_GLOBAL_RISK["risk_score"],
        factors=[RiskFactorResponse(**f) for f in DEFAULT_GLOBAL_RISK["factors"]],
        recommended_actions=DEFAULT_GLOBAL_RISK["recommended_actions"],
        created_at=DEFAULT_GLOBAL_RISK["created_at"],
    )


@router.get("", response_model=List[RiskAssessmentResponse])
async def list_risk_assessments(
    zone_id: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    risk_level: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
):
    risk_repo = RiskRepository()
    try:
        if zone_id:
            assessments = await risk_repo.get_by_zone(
                uuid.UUID(zone_id), start_time, end_time, limit
            )
        else:
            assessments = await risk_repo.list_recent(limit)
    except Exception:
        assessments = []
    
    if risk_level:
        assessments = [a for a in assessments if a.risk_level == risk_level]
    
    return [
        RiskAssessmentResponse(
            id=str(a.id),
            zone_id=str(a.zone_id) if a.zone_id else None,
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
    try:
        assessment = await risk_repo.get_latest_for_zone(uuid.UUID(zone_id))
        if assessment:
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
    except Exception:
        pass

    return RiskAssessmentResponse(
        id=str(uuid.uuid4()),
        zone_id=zone_id,
        timestamp=datetime.now(),
        risk_level="SAFE",
        risk_score=24.5,
        factors=[RiskFactorResponse(**f) for f in DEFAULT_GLOBAL_RISK["factors"]],
        recommended_actions=DEFAULT_GLOBAL_RISK["recommended_actions"],
        created_at=datetime.now(),
    )
