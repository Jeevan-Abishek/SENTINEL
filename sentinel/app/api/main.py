from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from datetime import datetime
from typing import List
from loguru import logger

from app.config import settings
from app.database.base import init_db, close_db
from app.api.routes import api_router


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.debug(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.debug(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dc in dead_connections:
            self.disconnect(dc)


manager = ConnectionManager()


broadcast_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global broadcast_task
    # Startup: Initialize database schema
    await init_db()
    if "pytest" not in sys.modules:
        broadcast_task = asyncio.create_task(broadcast_telemetry_loop())
    logger.info("SENTINEL API started successfully")
    yield
    # Shutdown: cancel background broadcaster
    if broadcast_task:
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
    logger.info("SENTINEL API shutdown cleanly")


app = FastAPI(
    title="SENTINEL API",
    description="Drone-Based Crowd Risk Detection & Early Warning System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SENTINEL API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "active_ws_clients": len(manager.active_connections),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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
        manager.disconnect(websocket)
    except Exception as e:
        logger.debug(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)


async def broadcast_telemetry_loop():
    """Periodic broadcast of heartbeat and pipeline state to connected frontend dashboards."""
    while True:
        try:
            await asyncio.sleep(5)
            if manager.active_connections:
                await manager.broadcast({
                    "type": "system_heartbeat",
                    "data": {
                        "status": "operational",
                        "timestamp": datetime.now().isoformat(),
                        "connected_clients": len(manager.active_connections),
                    }
                })
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Broadcaster loop error: {e}")
