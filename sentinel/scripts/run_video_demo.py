#!/usr/bin/env python3
"""
SENTINEL Demo Script - Integrated Pipeline Test

This script demonstrates the complete SENTINEL pipeline:
Video Input -> Detection -> Tracking -> Analytics -> Risk -> Alerts

Run with: python scripts/run_video_demo.py
"""

import asyncio
import sys
import os
import signal
import time
import uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import PipelineConfig, pipeline_manager
from app.video import MockVideoAdapter
from app.detection.models import DetectorConfig
from app.tracking.bytetrack import TrackerConfig
from app.analytics.models import AnalyticsConfig
from app.analytics.zone_config import ZoneOccupancyConfig
from app.risk.models import RiskConfig, AlertConfig
from app.mapping import CoordinateCalibrator, ZoneManager


async def run_demo():
    print("=" * 60)
    print("SENTINEL Integrated Pipeline Demo")
    print("=" * 60)
    
    # Create a mock video adapter (generates synthetic frames with people)
    video_adapter = MockVideoAdapter(
        fps=10,
        width=1280,
        height=720,
    )
    
    # Configure detection
    detection_config = DetectorConfig(
        model_path="models/yolov8n.pt",
        confidence_threshold=0.5,
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
            polygon=[[100, 100], [500, 100], [500, 400], [100, 400]],
        ),
        ZoneOccupancyConfig(
            zone_id="zone_2",
            name="Central Plaza",
            zone_type="monitoring",
            capacity=200,
            warning_threshold=0.7,
            critical_threshold=0.9,
            polygon=[[500, 100], [900, 100], [900, 400], [500, 400]],
        ),
        ZoneOccupancyConfig(
            zone_id="zone_3",
            name="Exit Gate",
            zone_type="exit",
            capacity=50,
            warning_threshold=0.7,
            critical_threshold=0.9,
            polygon=[[900, 100], [1180, 100], [1180, 400], [900, 400]],
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
        cooldown_seconds=10,
        max_active=50,
        sound_enabled=True,
    )
    
    # Create calibration (simple identity for demo)
    calibrator = CoordinateCalibrator()
    # Add some calibration points for the mock
    calibrator.add_calibration_point(100, 100, 0, 0)
    calibrator.add_calibration_point(1180, 100, 100, 0)
    calibrator.add_calibration_point(1180, 400, 100, 30)
    calibrator.add_calibration_point(100, 400, 0, 30)
    calibrator.compute_homography()
    
    # Create zone manager
    zone_manager = ZoneManager()
    zone_manager.set_calibrator(calibrator)
    
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
        target_fps=10,
    )
    
    # Create and start pipeline
    pipeline_id = str(uuid.uuid4())
    processor = pipeline_manager.create_pipeline(pipeline_id, pipeline_config)
    
    print(f"\nCreated pipeline: {pipeline_id}")
    print("Starting pipeline...")
    
    try:
        await processor.start()
    except Exception as e:
        print(f"Failed to start pipeline: {e}")
        print("\nNote: This demo requires YOLO model weights (yolov8n.pt) to be downloaded.")
        print("Download from: https://github.com/ultralytics/assets/releases")
        print("Place in: /workspace/app/sentinel/data/models/yolov8n.pt")
        return
    
    # Track stats
    frame_count = 0
    start_time = time.time()
    last_stats_time = time.time()
    
    # Set up signal handler for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler():
        print("\nShutdown signal received...")
        shutdown_event.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        while not shutdown_event.is_set():
            # Get result from pipeline
            result = await processor.get_result()
            
            if result:
                frame_count += 1
                
                # Print periodic stats
                current_time = time.time()
                if current_time - last_stats_time >= 2.0:
                    stats = processor.get_stats()
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    
                    print(f"\n--- Frame {frame_count} (FPS: {fps:.1f}, Avg processing: {stats['avg_processing_time']:.1f}ms) ---")
                    print(f"  Detections: {len(result.detections.detections)}")
                    print(f"  Active Tracks: {len(result.tracks.tracks)}")
                    
                    if result.count:
                        for zone_id, count in result.count.items():
                            print(f"  {zone_id}: {count.current_count} people ({count.entries} in, {count.exits} out)")
                    
                    if result.density:
                        for zone_id, density in result.density.items():
                            print(f"  {zone_id} Density: avg={density.avg_density:.2f}/m², max={density.max_density:.2f}/m²")
                    
                    if result.occupancy:
                        for zone_id, occ in result.occupancy.items():
                            print(f"  {zone_id} Occupancy: {occ.current_count}/{occ.capacity} ({occ.occupancy_ratio:.0%}) - {occ.status}")
                    
                    if result.risk:
                        print(f"  Risk: {result.risk.risk_level.value.upper()} (score: {result.risk.risk_score:.1f})")
                        if result.risk.recommended_actions:
                            for action in result.risk.recommended_actions:
                                print(f"    -> {action}")
                    
                    if result.alerts:
                        for alert in result.alerts:
                            print(f"  🚨 ALERT: [{alert.risk_level.value.upper()}] {alert.message}")
                    
                    last_stats_time = current_time
            
            # Small delay to prevent busy loop
            await asyncio.sleep(0.01)
            
    except asyncio.CancelledError:
        pass
    finally:
        print("\nStopping pipeline...")
        await processor.stop()
        
        elapsed = time.time() - start_time
        print(f"\nDemo completed:")
        print(f"  Total frames: {frame_count}")
        print(f"  Elapsed time: {elapsed:.1f}s")
        print(f"  Average FPS: {frame_count/elapsed:.1f}")


if __name__ == "__main__":
    asyncio.run(run_demo())