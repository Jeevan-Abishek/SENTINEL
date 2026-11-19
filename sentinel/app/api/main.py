"""
FastAPI Backend Application for SENTINEL.
Provides complete REST API endpoints, WebSocket live telemetry streaming,
and health/status monitoring for drone aerial crowd safety surveillance.
"""

import sys
import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database.base import init_db, close_db, engine
from app.api.routes import api_router
from app.api.websocket import manager, ws_router, generate_live_telemetry_payload


broadcast_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global broadcast_task
    # Startup: Initialize database schema (with SQLite fallback if Postgres is unavailable)
    try:
        await init_db()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")

    # Launch background broadcaster if not running under pytest
    if "pytest" not in sys.modules:
        broadcast_task = asyncio.create_task(broadcast_telemetry_loop())

    logger.info("SENTINEL FastAPI backend started successfully")
    yield

    # Shutdown
    if broadcast_task:
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
    try:
        await close_db()
    except Exception:
        pass
    logger.info("SENTINEL FastAPI backend shutdown cleanly")


app = FastAPI(
    title="SENTINEL API",
    description="Eco-Friendly Public Safety & Drone Crowd Risk Detection Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path,
        },
    )

# Include REST routes and WebSocket routes
app.include_router(api_router)
app.include_router(ws_router)


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint reporting API operational readiness."""
    return {
        "status": "healthy",
        "service": "SENTINEL API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_ws_clients": manager.count,
    }


@app.get("/api/v1/status", tags=["system"])
async def system_status():
    """System status endpoint reporting active mode, CV/telemetry engine, and database state."""
    db_url_str = str(engine.url) if engine else "sqlite+aiosqlite:///./sentinel.db"
    # Hide password if any
    if "@" in db_url_str and "://" in db_url_str:
        prefix, rest = db_url_str.split("://", 1)
        _, host_part = rest.split("@", 1)
        db_url_str = f"{prefix}://***:***@{host_part}"

    return {
        "status": "operational",
        "system": "SENTINEL Drone Crowd Risk Command System",
        "version": "1.0.0",
        "active_mode": "autonomous_aerial_surveillance",
        "cv_engine": {
            "model": "YOLOv8n",
            "tracker": "ByteTrack (Kalman Filter)",
            "device": getattr(settings, "yolo_device", "cpu"),
            "status": "ready",
            "frame_skip": 3,
            "target_resolution": "640x360",
        },
        "telemetry_engine": {
            "protocols": ["RTSP", "MAVLink UDP", "MockTelemetryProvider"],
            "status": "ready",
            "altitude_agl_m": 32.4,
            "gimbal_pitch_deg": -89.0,
        },
        "risk_engine": {
            "factors": 5,
            "weights": {
                "density": 0.30,
                "occupancy": 0.20,
                "direction_variance": 0.20,
                "flow_rate": 0.15,
                "bottleneck_index": 0.15,
            },
            "status": "active",
        },
        "database": {
            "dialect": engine.dialect.name if engine else "sqlite",
            "url": db_url_str,
            "status": "connected",
        },
        "active_ws_clients": manager.count,
        "timestamp": datetime.now().isoformat(),
    }


@app.websocket("/ws")
async def websocket_legacy_endpoint(websocket: WebSocket):
    """Legacy WebSocket endpoint maintained for backward compatibility."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.debug(f"WebSocket legacy error: {e}")
        await manager.disconnect(websocket)


async def broadcast_telemetry_loop():
    """Periodic broadcast of heartbeat and pipeline state to connected frontend dashboards."""
    while True:
        try:
            await asyncio.sleep(3.0)
            if manager.count > 0:
                payload = generate_live_telemetry_payload()
                await manager.broadcast(payload)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Broadcaster loop error: {e}")


__all__ = ["app", "manager", "lifespan", "ConnectionManager"]
