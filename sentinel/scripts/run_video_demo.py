#!/usr/bin/env python3
"""
SENTINEL Demo Script - Integrated Pipeline Test

This script demonstrates the complete SENTINEL pipeline:
Video Input (File/Mock) -> Detection -> Tracking -> Analytics -> Risk Engine -> Alerts
"""

import asyncio
import sys
import os
import signal
import time
import uuid
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.pipeline import PipelineConfig, pipeline_manager
from app.video import FileVideoAdapter, MockVideoAdapter
from app.detection.models import DetectorConfig
from app.tracking.bytetrack import TrackerConfig
from app.analytics.models import AnalyticsConfig
from app.analytics.zone_config import ZoneOccupancyConfig
from app.risk.models import RiskConfig, AlertConfig
from app.mapping import CoordinateCalibrator, ZoneManager


async def run_demo(video_path: str = None, max_frames: int = 50):
    print("=" * 60)
    print("SENTINEL Integrated Pipeline Demo")
    print("=" * 60)

    # Resolve video source
    if video_path and os.path.exists(video_path):
        print(f"Using Video Source: {video_path}")
        video_adapter = FileVideoAdapter(video_path)
    else:
        sample = settings.sample_videos_dir / "demo_drone_crowd.mp4"
        if sample.exists():
            print(f"Using Sample Drone Footage: {sample}")
            video_adapter = FileVideoAdapter(str(sample))
        else:
            print("Using Synthetic Mock Video Stream")
            video_adapter = MockVideoAdapter(fps=10, width=1280, height=720)

    # Configure detection using real YOLO weights
    yolo_weights = str(settings.resolved_yolo_model_path)
    print(f"Loading YOLO Model: {yolo_weights}")
    detection_config = DetectorConfig(
        model_path=yolo_weights,
        confidence_threshold=0.35,
        iou_threshold=0.45,
        device="cpu",
        input_width=640,
        input_height=640,
    )

    # Configure tracker
    tracker_config = TrackerConfig(
        track_buffer=30,
        match_thresh=0.8,
        min_box_area=10,
        mot20=False,
    )

    # Configure analytics
    analytics_config = AnalyticsConfig(
        density_grid_size=2.0,
        occupancy_warning_threshold=0.7,
        occupancy_critical_threshold=0.9,
        flow_window_seconds=30,
        bottleneck_persistence_frames=15,
        min_tracks_for_flow=3,
        speed_smoothing_alpha=0.3,
        direction_bins=8,
    )

    # Define zones
    zone_configs = [
        ZoneOccupancyConfig(
            zone_id="zone_1",
            name="Main Entrance",
            zone_type="entry",
            capacity=100,
            warning_threshold=0.7,
            critical_threshold=0.9,
            polygon=[[50, 50], [450, 50], [450, 400], [50, 400]],
        ),
        ZoneOccupancyConfig(
            zone_id="zone_2",
            name="Central Plaza",
            zone_type="monitoring",
            capacity=200,
            warning_threshold=0.7,
            critical_threshold=0.9,
            polygon=[[450, 50], [850, 50], [850, 400], [450, 400]],
        ),
        ZoneOccupancyConfig(
            zone_id="zone_3",
            name="Exit Gate",
            zone_type="exit",
            capacity=50,
            warning_threshold=0.7,
            critical_threshold=0.9,
            polygon=[[850, 50], [1200, 50], [1200, 400], [850, 400]],
        ),
    ]

    # Configure risk
    risk_config = RiskConfig(
        density_weight=0.3,
        occupancy_weight=0.2,
        movement_weight=0.2,
        flow_weight=0.15,
        bottleneck_weight=0.15,
        safe_threshold=30.0,
        warning_threshold=60.0,
        critical_threshold=80.0,
        temporal_smoothing=0.3,
    )

    # Configure alerts
    alert_config = AlertConfig(
        cooldown_seconds=5,
        max_active=50,
        sound_enabled=False,
    )

    # Create calibration
    calibrator = CoordinateCalibrator()
    calibrator.add_calibration_point(50, 50, 0, 0)
    calibrator.add_calibration_point(1200, 50, 50, 0)
    calibrator.add_calibration_point(1200, 400, 50, 20)
    calibrator.add_calibration_point(50, 400, 0, 20)
    calibrator.compute_homography()

    # Create zone manager
    zone_manager = ZoneManager(calibrator=calibrator)
    for zone in zone_configs:
        zone_manager.add_zone(
            zone_id=zone.zone_id,
            name=zone.name,
            zone_type=zone.zone_type,
            image_polygon=zone.polygon or [],
            capacity=zone.capacity,
        )

    # Create pipeline config
    pipeline_config = PipelineConfig(
        event_id="demo_event",
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

    print(f"\nPipeline ID: {pipeline_id}")
    print("Starting pipeline processing...")

    try:
        await processor.start()
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        return

    frame_count = 0
    start_time = time.time()
    last_stats_time = time.time()

    shutdown_event = asyncio.Event()

    # Windows-safe signal handling
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_event.set)
    except (NotImplementedError, AttributeError):
        pass

    try:
        while not shutdown_event.is_set():
            result = await processor.get_result()
            if result:
                frame_count += 1
                curr_t = time.time()
                if curr_t - last_stats_time >= 1.0 or frame_count % 10 == 0:
                    stats = processor.get_stats()
                    elapsed = curr_t - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"Frame {frame_count:03d} | Detections: {len(result.detections.detections):02d} | "
                        f"Tracks: {len(result.tracks.tracks):02d} | "
                        f"FPS: {fps:.1f} | Avg Proc: {stats.get('avg_processing_time', 0.0):.1f}ms"
                    )
                    if result.risk:
                        print(f"  -> Risk Level: {result.risk.risk_level.value.upper()} (Score: {result.risk.risk_score:.1f})")
                    if result.alerts:
                        for al in result.alerts:
                            print(f"  🚨 ALERT: [{al.risk_level.value.upper()}] {al.message}")

                    last_stats_time = curr_t

                if max_frames and frame_count >= max_frames:
                    print(f"\nReached max_frames limit ({max_frames}). Stopping gracefully.")
                    break
            else:
                await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        pass
    finally:
        print("\nStopping pipeline...")
        await processor.stop()
        pipeline_manager.remove_pipeline(pipeline_id)

        elapsed = time.time() - start_time
        print(f"\nDemo Execution Summary:")
        print(f"  Total frames processed : {frame_count}")
        print(f"  Elapsed wall time       : {elapsed:.2f}s")
        if elapsed > 0:
            print(f"  Average Pipeline FPS    : {frame_count / elapsed:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL Pipeline Demo")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--frames", type=int, default=30, help="Max frames to process (default: 30)")
    args = parser.parse_args()

    asyncio.run(run_demo(video_path=args.video, max_frames=args.frames))
