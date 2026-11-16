from app.api.routes import (
    events, zones, drones, detections, tracking,
    analytics, risk, alerts, reports, calibration, mapping, pipeline
)
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(events.router)
api_router.include_router(zones.router)
api_router.include_router(drones.router)
api_router.include_router(detections.router)
api_router.include_router(tracking.router)
api_router.include_router(analytics.router)
api_router.include_router(risk.router)
api_router.include_router(alerts.router)
api_router.include_router(reports.router)
api_router.include_router(calibration.router)
api_router.include_router(mapping.router)
api_router.include_router(pipeline.router)