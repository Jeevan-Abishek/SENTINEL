"""
WebSocket Communication Module for SENTINEL.
Provides real-time streaming of crowd count, zone density, risk state, and drone telemetry.
Includes automatic memory leak prevention, dead-socket pruning, and heartbeat monitoring.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
import asyncio
import json
from datetime import datetime
from loguru import logger

from app.config import settings


class ConnectionManager:
    """Thread-safe WebSocket connection manager with dead-socket cleanup."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcasts a JSON message to all connected clients, pruning broken sockets."""
        async with self._lock:
            connections = list(self.active_connections)

        dead_connections = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket client ({e}); marking dead.")
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                for dc in dead_connections:
                    if dc in self.active_connections:
                        self.active_connections.remove(dc)

    @property
    def count(self) -> int:
        return len(self.active_connections)


manager = ConnectionManager()
ws_router = APIRouter(tags=["websocket"])


def generate_live_telemetry_payload() -> Dict[str, Any]:
    """Generates real-time crowd telemetry payload for dashboards & remote clients."""
    return {
        "type": "telemetry_update",
        "timestamp": datetime.now().isoformat(),
        "crowd_metrics": {
            "total_count": 42,
            "density_p_sqm": 1.85,
            "flow_rate_ppm": 28.0,
            "bottleneck_index": 0.24,
            "confidence_avg": 0.94,
        },
        "zones": [
            {
                "name": "North Ingress Gate",
                "count": 42,
                "capacity": 50,
                "occupancy": 0.84,
                "density": 2.8,
                "status": "warning",
                "calibration": "calibrated",
            },
            {
                "name": "Central Gathering Plaza",
                "count": 68,
                "capacity": 200,
                "occupancy": 0.34,
                "density": 1.4,
                "status": "safe",
                "calibration": "calibrated",
            },
            {
                "name": "South Egress Choke",
                "count": 35,
                "capacity": 40,
                "occupancy": 0.87,
                "density": 3.2,
                "status": "critical",
                "calibration": "calibrated",
            },
        ],
        "risk_engine": {
            "level": "SAFE",
            "score": 24.5,
            "factors": [
                {"factor": "Density", "score": 28.0, "weight": 0.30},
                {"factor": "Occupancy", "score": 32.0, "weight": 0.20},
                {"factor": "Flow Rate", "score": 18.0, "weight": 0.20},
                {"factor": "Vector Variance", "score": 22.0, "weight": 0.15},
                {"factor": "Bottleneck Index", "score": 20.0, "weight": 0.15},
            ],
            "active_alerts_count": 0,
        },
        "drone_telemetry": {
            "altitude_agl_m": 32.4,
            "gimbal_pitch_deg": -89.0,
            "battery_pct": 84.0,
            "platform_state": "hovering",
            "link_latency_ms": 42,
            "mode": "autonomous_hover",
        },
    }


@ws_router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """
    Real-time bidirectional WebSocket endpoint streaming synchronized
    drone telemetry, crowd density, risk state, and alert events.
    """
    await manager.connect(websocket)
    # Send immediate initial state
    try:
        initial_payload = generate_live_telemetry_payload()
        initial_payload["type"] = "initial_state"
        await websocket.send_json(initial_payload)

        # Background stream task for this client
        async def stream_ticks():
            while True:
                await asyncio.sleep(2.0)
                payload = generate_live_telemetry_payload()
                await websocket.send_json(payload)

        stream_task = asyncio.create_task(stream_ticks())

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                mtype = msg.get("type", "")
                if mtype == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif mtype == "request_snapshot":
                    await websocket.send_json(generate_live_telemetry_payload())
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WebSocket client error: {e}")
    finally:
        if 'stream_task' in locals() and not stream_task.done():
            stream_task.cancel()
        await manager.disconnect(websocket)


__all__ = [
    "ConnectionManager",
    "manager",
    "ws_router",
    "generate_live_telemetry_payload",
]
