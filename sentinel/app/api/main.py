from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from datetime import datetime
from typing import List

from app.config import settings
from app.database.base import init_db, close_db


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title="SENTINEL API",
    description="Drone-Based Crowd Risk Detection System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SENTINEL API",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/events")
async def list_events():
    return {
        "events": [
            {
                "id": "demo-event-1",
                "name": "Demo Event",
                "location": "San Francisco",
                "status": "active",
                "start_time": "2026-01-15T10:00:00Z",
            }
        ]
    }


@app.get("/api/v1/zones")
async def list_zones():
    return {
        "zones": [
            {"id": "zone-1", "name": "Main Entrance", "type": "entry", "capacity": 500, "current_count": 120},
            {"id": "zone-2", "name": "Central Plaza", "type": "monitoring", "capacity": 1000, "current_count": 450},
            {"id": "zone-3", "name": "Exit Gate A", "type": "exit", "capacity": 300, "current_count": 80},
        ]
    }


@app.get("/api/v1/drones")
async def list_drones():
    return {
        "drones": [
            {"id": "drone-1", "name": "Sentinel-1", "status": "streaming", "event_id": "demo-event-1"},
        ]
    }


@app.get("/api/v1/analytics/summary")
async def analytics_summary():
    return {
        "total_count": 650,
        "active_zones": 3,
        "safe_zones": 1,
        "warning_zones": 1,
        "critical_zones": 1,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/alerts")
async def list_alerts():
    return {
        "alerts": [
            {
                "id": "alert-1",
                "zone_id": "zone-2",
                "risk_level": "warning",
                "message": "High density detected in Central Plaza",
                "timestamp": datetime.utcnow().isoformat(),
            }
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_mock_updates():
    while True:
        await asyncio.sleep(5)
        await manager.broadcast({
            "type": "analytics_update",
            "data": {
                "total_count": 650,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_mock_updates())