"""Unit tests for Phase 8: FastAPI REST endpoints and WebSocket functionality."""

import pytest
from fastapi.testclient import TestClient
import uuid
from datetime import datetime
import asyncio

from app.api.main import app
from app.database.base import init_db


class TestAPIEndpoints:
    def test_health_check(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "SENTINEL API"

    def test_events_crud_endpoints(self):
        client = TestClient(app)
        # Create event
        payload = {
            "name": "Independence Day Parade",
            "location": "National Mall",
            "latitude": 38.8893,
            "longitude": -77.0502,
            "description": "Crowd safety monitoring",
        }
        res = client.post("/api/v1/events", json=payload)
        assert res.status_code == 200
        created = res.json()
        event_id = created["id"]
        assert created["name"] == "Independence Day Parade"

        # List events
        res_list = client.get("/api/v1/events")
        assert res_list.status_code == 200
        assert any(e["id"] == event_id for e in res_list.json())

        # Get single event
        res_get = client.get(f"/api/v1/events/{event_id}")
        assert res_get.status_code == 200
        assert res_get.json()["name"] == "Independence Day Parade"

        # Delete event
        res_del = client.delete(f"/api/v1/events/{event_id}")
        assert res_del.status_code == 200

    def test_alerts_endpoints(self):
        client = TestClient(app)
        # First create an event to attach alert to
        ev_res = client.post("/api/v1/events", json={
            "name": "Stadium Match",
            "location": "Sector 4",
        })
        event_id = ev_res.json()["id"]

        # Create alert
        alert_payload = {
            "event_id": event_id,
            "title": "Severe Bottleneck Gate B",
            "message": "Flow rate dropped to 0.1 persons/sec",
            "reason": "Exceeds critical congestion index",
            "risk_level": "critical",
            "risk_score": 85.0,
            "metadata": {"zone": "gate_b"},
        }
        res = client.post("/api/v1/alerts", json=alert_payload)
        assert res.status_code == 200
        alert_data = res.json()
        alert_id = alert_data["id"]
        assert alert_data["title"] == "Severe Bottleneck Gate B"
        assert alert_data["acknowledged"] is False

        # Active count
        res_count = client.get("/api/v1/alerts/active/count")
        assert res_count.status_code == 200
        assert res_count.json()["count"] >= 1

        # Acknowledge
        res_ack = client.post(
            f"/api/v1/alerts/{alert_id}/acknowledge",
            json={"operator_id": str(uuid.uuid4())},
        )
        assert res_ack.status_code == 200
        assert res_ack.json()["acknowledged"] is True

    def test_video_endpoints(self):
        client = TestClient(app)
        # List samples
        res_samples = client.get("/api/v1/video/samples")
        assert res_samples.status_code == 200
        samples = res_samples.json()
        assert isinstance(samples, list)
        assert any(s["filename"] == "demo_drone_crowd.mp4" for s in samples)

        # Get video info
        res_info = client.get(
            "/api/v1/video/info",
            params={"path": "data/sample_videos/demo_drone_crowd.mp4"},
        )
        assert res_info.status_code == 200
        info = res_info.json()
        assert info["is_valid"] is True
        assert info["width"] == 1280
        assert info["height"] == 720
        assert info["fps"] > 0
        assert info["frame_count"] > 0
