"""Unit tests for Phase 9: Database Repositories and CRUD persistence."""

import pytest
import uuid
from datetime import datetime, timedelta

from app.database.base import init_db
from app.database.models import EventStatus, AlertStatus, AlertSeverity, RiskLevel
from app.database.repositories import (
    EventRepository,
    ZoneRepository,
    DroneRepository,
    DetectionRepository,
    TrackRepository,
    AnalyticsRepository,
    RiskRepository,
    AlertRepository,
)


@pytest.mark.asyncio
class TestDatabaseRepositories:
    @pytest.fixture(autouse=True)
    async def setup_db(self):
        await init_db()

    async def test_event_crud(self):
        repo = EventRepository()
        now = datetime.now()

        event = await repo.create(
            name="City Marathon 2026",
            location="Downtown Square",
            latitude=12.9716,
            longitude=77.5946,
            start_time=now,
            description="Annual marathon crowd monitoring",
            status="active",
        )
        assert event.id is not None
        assert event.name == "City Marathon 2026"
        assert event.status == EventStatus.ACTIVE

        # Fetch
        fetched = await repo.get_by_id(event.id)
        assert fetched is not None
        assert fetched.name == "City Marathon 2026"

        # Update
        updated = await repo.update(event.id, name="City Marathon 2026 - Updated")
        assert updated.name == "City Marathon 2026 - Updated"

        # Delete
        deleted = await repo.delete(event.id)
        assert deleted is True
        assert await repo.get_by_id(event.id) is None

    async def test_zone_and_drone_repositories(self):
        event_repo = EventRepository()
        zone_repo = ZoneRepository()
        drone_repo = DroneRepository()

        event = await event_repo.create(
            name="Festival",
            location="Central Park",
            latitude=13.0,
            longitude=80.0,
            start_time=datetime.now(),
        )

        # Create zone
        zone = await zone_repo.create(
            event_id=event.id,
            name="Main Stage",
            zone_type="monitoring",
            capacity=500,
        )
        assert zone.id is not None
        assert zone.event_id == event.id
        assert zone.capacity == 500

        # Create drone
        drone = await drone_repo.create(
            event_id=event.id,
            name="Sentinel Alpha",
            model="Quadcopter Pro",
            status="online",
        )
        assert drone.id is not None
        assert drone.event_id == event.id
        assert drone.status.value == "online"

    async def test_detection_and_track_repositories(self):
        event_repo = EventRepository()
        drone_repo = DroneRepository()
        det_repo = DetectionRepository()
        track_repo = TrackRepository()

        event = await event_repo.create(
            name="Expo",
            location="Convention Hall",
            latitude=12.0,
            longitude=77.0,
            start_time=datetime.now(),
        )
        drone = await drone_repo.create(
            event_id=event.id,
            name="Sentinel Beta",
        )

        now = datetime.now()
        det = await det_repo.create(
            drone_id=drone.id,
            frame_number=42,
            timestamp=now,
            num_detections=3,
            detections_data=[{"bbox": [10, 10, 50, 90], "confidence": 0.88, "class_id": 0}],
            processing_time_ms=15.4,
            model_version="yolov8n",
        )
        assert det.id is not None
        assert det.num_detections == 3

        # Create track
        track = await track_repo.create(
            drone_id=drone.id,
            track_id=101,
            class_id=0,
            first_frame=1,
            last_frame=42,
            first_timestamp=now - timedelta(seconds=2),
            last_timestamp=now,
            bbox_history=[[10, 10, 50, 90]],
            centroid_history=[[30, 50]],
        )
        assert track.id is not None
        assert track.track_id == 101

    async def test_alert_lifecycle(self):
        event_repo = EventRepository()
        alert_repo = AlertRepository()

        event = await event_repo.create(
            name="Stadium Concert",
            location="Arena",
            latitude=12.5,
            longitude=77.5,
            start_time=datetime.now(),
        )

        # Create Alert
        alert = await alert_repo.create(
            event_id=event.id,
            title="High Density Ingress Gate",
            message="Density reached 3.8 people/m² exceeding 3.5 threshold",
            reason="Rapid crowd accumulation at Turnstile A",
            risk_level="warning",
            risk_score=72.0,
        )
        assert alert.id is not None
        assert alert.status == AlertStatus.ACTIVE
        assert alert.risk_level == AlertSeverity.WARNING

        active_count = await alert_repo.count_active()
        assert active_count >= 1

        # Acknowledge
        operator_uuid = uuid.uuid4()
        ack = await alert_repo.acknowledge(alert.id, operator_id=operator_uuid)
        assert ack is not None
        assert ack.status == AlertStatus.ACKNOWLEDGED
        assert ack.acknowledged_by == operator_uuid
        assert ack.acknowledged_at is not None

