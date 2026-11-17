"""Unit tests for Phase 5: Drone Video, Telemetry, and Camera Motion Compensation."""

import pytest
import numpy as np
import cv2
from datetime import datetime, timedelta

from app.video.telemetry import (
    DroneTelemetry,
    GPSCoordinate,
    GimbalOrientation,
    DroneStatus,
    TelemetryBuffer,
    MockTelemetryProvider,
    TelemetrySynchronizer,
)
from app.video.camera_motion import (
    CameraMotionCompensator,
    CameraMotionResult,
)
from app.video.drone_adapter import (
    DroneStreamAdapter,
    DroneFrameBundle,
)


class TestDroneTelemetry:
    def test_telemetry_creation_and_dict_serialization(self):
        coord = GPSCoordinate(latitude=12.9716, longitude=77.5946, altitude=45.0)
        gimbal = GimbalOrientation(pitch=-45.0, roll=0.0, yaw=180.0)
        now = datetime.now()

        telemetry = DroneTelemetry(
            timestamp=now,
            position=coord,
            gimbal=gimbal,
            battery_level=88.5,
            flight_mode="GUIDED",
            status=DroneStatus.FLYING,
            is_simulated=False,
        )

        data = telemetry.to_dict()
        assert data["position"]["lat"] == 12.9716
        assert data["position"]["lon"] == 77.5946
        assert data["position"]["alt"] == 45.0
        assert data["gimbal"]["pitch"] == -45.0
        assert data["battery_level"] == 88.5
        assert data["is_simulated"] is False

        restored = DroneTelemetry.from_dict(data)
        assert restored.position.latitude == 12.9716
        assert restored.position.longitude == 77.5946
        assert restored.gimbal.pitch == -45.0
        assert restored.is_simulated is False

    def test_mock_telemetry_provider_labels_simulated(self):
        provider = MockTelemetryProvider()
        tel = provider.generate_telemetry()

        assert tel.is_simulated is True
        assert tel.status == DroneStatus.FLYING
        assert 0.0 <= tel.battery_level <= 100.0


class TestTelemetrySynchronizer:
    def test_sync_with_real_telemetry_buffer(self):
        buffer = TelemetryBuffer()
        start = datetime.now()

        # Add 3 telemetry records
        for i in range(3):
            t = DroneTelemetry(
                timestamp=start + timedelta(seconds=i),
                position=GPSCoordinate(latitude=10.0 + i * 0.01, longitude=20.0, altitude=30.0),
                gimbal=GimbalOrientation(-90.0, 0.0, 0.0),
                is_simulated=False,
            )
            buffer.add(t)

        syncer = TelemetrySynchronizer(telemetry_buffer=buffer, max_time_skew_seconds=0.6)
        syncer.start_wall_time = start

        # Sync frame at timestamp 1.05s (closest is at 1.0s, delta 0.05s <= 0.6s)
        synced = syncer.sync_frame(frame_timestamp_sec=1.05)
        assert synced.is_simulated is False
        assert pytest.approx(synced.position.latitude, rel=1e-3) == 10.01

    def test_sync_fallback_when_skew_exceeds_threshold(self):
        buffer = TelemetryBuffer()
        start = datetime.now()
        # Telemetry is at 0s only
        buffer.add(DroneTelemetry(
            timestamp=start,
            position=GPSCoordinate(10.0, 20.0, 30.0),
            gimbal=GimbalOrientation(-90.0, 0.0, 0.0),
            is_simulated=False,
        ))

        syncer = TelemetrySynchronizer(telemetry_buffer=buffer, max_time_skew_seconds=0.2)
        syncer.start_wall_time = start

        # Frame at 5.0 seconds has no close telemetry -> falls back with is_simulated=True
        synced = syncer.sync_frame(frame_timestamp_sec=5.0)
        assert synced.is_simulated is True


class TestCameraMotionCompensation:
    def test_first_frame_returns_identity(self):
        cmc = CameraMotionCompensator()
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw some features
        cv2.circle(frame, (50, 50), 10, (255, 255, 255), -1)
        cv2.circle(frame, (150, 150), 10, (255, 255, 255), -1)

        result = cmc.estimate_motion(frame)
        assert result.is_valid is True
        assert result.dx == 0.0
        assert result.dy == 0.0

    def test_motion_estimation_with_synthetic_translation(self):
        cmc = CameraMotionCompensator(max_features=200, quality_level=0.01)
        
        # Create textured frame with many distinct corners
        frame1 = np.zeros((300, 300, 3), dtype=np.uint8)
        for x in range(30, 270, 30):
            for y in range(30, 270, 30):
                cv2.rectangle(frame1, (x, y), (x + 10, y + 10), (255, 255, 255), -1)

        # Estimate motion for frame 1
        cmc.estimate_motion(frame1)

        # Frame 2 is translated by dx=5, dy=3
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        frame2 = cv2.warpAffine(frame1, M, (300, 300))

        result = cmc.estimate_motion(frame2)
        assert result.is_valid is True
        # Estimated translation should be close to (5, 3)
        assert abs(result.dx - 5.0) < 1.0
        assert abs(result.dy - 3.0) < 1.0

    def test_point_and_bbox_compensation(self):
        cmc = CameraMotionCompensator()
        motion = CameraMotionResult(
            dx=10.0,
            dy=5.0,
            transform_matrix=np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 5.0]], dtype=np.float32),
            is_valid=True,
        )

        # Current point (100, 50) in current frame was at (90, 45) before camera motion
        compensated = cmc.compensate_point((100.0, 50.0), motion)
        assert pytest.approx(compensated[0], abs=0.1) == 90.0
        assert pytest.approx(compensated[1], abs=0.1) == 45.0

        # Test bbox compensation
        bbox = (100.0, 50.0, 120.0, 90.0)
        comp_bbox = cmc.compensate_bbox(bbox, motion)
        assert pytest.approx(comp_bbox[0], abs=0.1) == 90.0
        assert pytest.approx(comp_bbox[1], abs=0.1) == 45.0
        assert pytest.approx(comp_bbox[2], abs=0.1) == 110.0
        assert pytest.approx(comp_bbox[3], abs=0.1) == 85.0


@pytest.mark.asyncio
class TestDroneStreamAdapter:
    async def test_drone_stream_adapter_with_mock_video(self):
        adapter = DroneStreamAdapter(
            video_source="mock",
            frame_skip=1,
            enable_motion_compensation=True,
            mock_telemetry=True,
        )

        connected = await adapter.connect()
        assert connected is True
        assert adapter.is_connected is True

        bundle = await adapter.get_next_frame()
        assert bundle is not None
        assert isinstance(bundle, DroneFrameBundle)
        assert bundle.frame is not None
        assert bundle.telemetry is not None
        assert bundle.is_telemetry_simulated is True
        assert bundle.camera_motion is not None

        status = adapter.get_status()
        assert status["connected"] is True
        assert status["frames_served"] >= 1
        assert status["motion_compensation_enabled"] is True

        await adapter.disconnect()
        assert adapter.is_connected is False

