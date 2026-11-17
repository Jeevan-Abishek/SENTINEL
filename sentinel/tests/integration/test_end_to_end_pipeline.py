"""Integration test: Full end-to-end SENTINEL pipeline on drone video footage."""

import pytest
import asyncio
import uuid
import os

from app.config import settings
from app.pipeline import PipelineConfig, pipeline_manager
from app.video import FileVideoAdapter, MockVideoAdapter
from app.detection.models import DetectorConfig
from app.tracking.tracker import TrackerConfig
from app.analytics.models import AnalyticsConfig
from app.analytics.zone_config import ZoneOccupancyConfig
from app.risk.models import RiskConfig, AlertConfig
from app.mapping import CoordinateCalibrator, ZoneManager


@pytest.mark.asyncio
class TestEndToEndPipeline:
    async def test_full_pipeline_on_sample_drone_video(self):
        sample_path = settings.sample_videos_dir / "demo_drone_crowd.mp4"
        assert sample_path.exists(), f"Sample video missing at {sample_path}"

        video_adapter = FileVideoAdapter(str(sample_path))
        detection_config = DetectorConfig(
            model_path=str(settings.resolved_yolo_model_path),
            confidence_threshold=0.35,
            iou_threshold=0.45,
            device="cpu",
        )
        tracker_config = TrackerConfig(track_buffer=30, match_thresh=0.8)
        analytics_config = AnalyticsConfig(density_grid_size=2.0)

        zone_configs = [
            ZoneOccupancyConfig(
                zone_id="zone_central",
                name="Central Promenade",
                zone_type="monitoring",
                capacity=100,
                warning_threshold=0.7,
                critical_threshold=0.9,
                polygon=[[50, 50], [1200, 50], [1200, 650], [50, 650]],
            ),
        ]

        risk_config = RiskConfig(
            safe_threshold=30.0,
            warning_threshold=60.0,
            critical_threshold=80.0,
        )
        alert_config = AlertConfig(cooldown_seconds=5)

        calibrator = CoordinateCalibrator()
        calibrator.add_calibration_point(50, 50, 0, 0)
        calibrator.add_calibration_point(1200, 50, 50, 0)
        calibrator.add_calibration_point(1200, 650, 50, 30)
        calibrator.add_calibration_point(50, 650, 0, 30)
        calibrator.compute_homography()

        zone_manager = ZoneManager(calibrator=calibrator)
        zone_manager.add_zone(
            zone_id="zone_central",
            name="Central Promenade",
            zone_type="monitoring",
            image_polygon=[[50, 50], [1200, 50], [1200, 650], [50, 650]],
            capacity=100,
        )

        pipeline_config = PipelineConfig(
            event_id="integration_test_event",
            video_adapter=video_adapter,
            detection_config=detection_config,
            tracker_config=tracker_config,
            analytics_config=analytics_config,
            zone_configs=zone_configs,
            risk_config=risk_config,
            alert_config=alert_config,
            calibration=calibrator,
            zone_manager=zone_manager,
            enable_analytics=True,
            enable_risk=True,
            enable_alerts=True,
            target_fps=15,
        )

        pipeline_id = str(uuid.uuid4())
        processor = pipeline_manager.create_pipeline(pipeline_id, pipeline_config)

        await processor.start()

        frames_collected = 0
        total_detections = 0
        total_active_tracks = 0

        try:
            for _ in range(100):
                result = await processor.get_result()
                if result:
                    frames_collected += 1
                    total_detections += len(result.detections.detections)
                    total_active_tracks += len(result.tracks.tracks)

                    assert result.detections is not None
                    assert result.tracks is not None
                    assert result.risk is not None
                    assert 0.0 <= result.risk.risk_score <= 100.0

                    if frames_collected >= 3:
                        break
                await asyncio.sleep(0.05)

            assert frames_collected >= 3
            assert total_detections > 0, "Expected person detections in demo footage"
            assert total_active_tracks > 0, "Expected active tracked persons"

        finally:
            await processor.stop()
            pipeline_manager.remove_pipeline(pipeline_id)

    async def test_empty_feed_pipeline_stability(self):
        video_adapter = MockVideoAdapter(fps=10, width=640, height=480, frame_count=10)
        detection_config = DetectorConfig(
            model_path=str(settings.resolved_yolo_model_path),
            confidence_threshold=0.35,
        )
        tracker_config = TrackerConfig()
        analytics_config = AnalyticsConfig()
        risk_config = RiskConfig()
        alert_config = AlertConfig()

        pipeline_config = PipelineConfig(
            event_id="empty_feed_test",
            video_adapter=video_adapter,
            detection_config=detection_config,
            tracker_config=tracker_config,
            analytics_config=analytics_config,
            zone_configs=[],
            risk_config=risk_config,
            alert_config=alert_config,
            enable_analytics=True,
            enable_risk=True,
            enable_alerts=True,
            target_fps=15,
        )

        pipeline_id = str(uuid.uuid4())
        processor = pipeline_manager.create_pipeline(pipeline_id, pipeline_config)

        await processor.start()
        try:
            for _ in range(50):
                result = await processor.get_result()
                if result:
                    assert result.detections is not None
                    assert result.tracks is not None
                    break
                await asyncio.sleep(0.05)
        finally:
            await processor.stop()
            pipeline_manager.remove_pipeline(pipeline_id)

