"""Performance and Latency Benchmark for SENTINEL Pipeline.

Measures stage-by-stage latencies:
- Video Ingestion & Quality Preprocessing
- YOLOv8 Person Detection Inference
- Multi-Object Tracking (ByteTrack)
- Crowd Analytics (Count, Calibrated Density, Movement, Flow, Bottleneck)
- Risk Engine Assessment & Temporal Smoothing
- Alert Generation & Evaluation
"""

import time
import json
import statistics
import sys
from pathlib import Path
import numpy as np
import cv2

# Ensure sentinel root and workspace root are in sys.path
_sentinel_root = Path(__file__).resolve().parent.parent
if str(_sentinel_root) not in sys.path:
    sys.path.insert(0, str(_sentinel_root))
_workspace_root = _sentinel_root.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from app.config import settings
from app.video.adapter import FileVideoAdapter
from app.video.processor import FramePreprocessor, FrameQualityChecker
from app.video.camera_motion import CameraMotionCompensator
from app.detection.yolo_detector import YOLODetector
from app.detection.models import DetectorConfig
from app.detection.detector import DetectionContext
from app.tracking.tracker import Tracker, TrackerConfig, TrackingContext
from app.analytics import (
    PeopleCounter,
    DensityEstimator,
    OccupancyMonitor,
    MovementAnalyzerWrapper,
    FlowAnalyzerWrapper,
    BottleneckDetectorWrapper,
    AnalyticsConfig,
)
from app.analytics.models import AnalyticsContext
from app.analytics.zone_config import ZoneOccupancyConfig
from app.risk.engine import RiskEngineWrapper
from app.risk.models import RiskConfig, AlertConfig
from app.alerts.manager import AlertManagerWrapper
from app.mapping import CoordinateCalibrator, ZoneManager


def run_benchmark(num_frames: int = 40):
    video_path = settings.sample_videos_dir / "demo_drone_crowd.mp4"
    print(f"Starting SENTINEL Performance Benchmark ({num_frames} frames)...")
    print(f"Video Source: {video_path}")

    # Initialize components
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f"Cannot open {video_path}"

    preprocessor = FramePreprocessor()
    quality_checker = FrameQualityChecker()
    cmc = CameraMotionCompensator()

    detector = YOLODetector(DetectorConfig(
        model_path=str(settings.resolved_yolo_model_path),
        confidence_threshold=0.35,
        device="cpu",
    ))
    detector.load_model()

    tracker = Tracker(TrackerConfig())

    analytics_config = AnalyticsConfig()
    density_estimator = DensityEstimator(analytics_config)
    occupancy_monitor = OccupancyMonitor(analytics_config)
    movement_analyzer = MovementAnalyzerWrapper(analytics_config)
    flow_analyzer = FlowAnalyzerWrapper(analytics_config)
    bottleneck_detector = BottleneckDetectorWrapper(analytics_config)

    risk_engine = RiskEngineWrapper(RiskConfig())
    alert_manager = AlertManagerWrapper(AlertConfig())

    zone_poly = [[50, 50], [1200, 50], [1200, 650], [50, 650]]
    people_counter = PeopleCounter("zone_main", zone_poly)

    stage_latencies = {
        "decode_ms": [],
        "preprocess_and_cmc_ms": [],
        "yolo_detection_ms": [],
        "tracking_ms": [],
        "crowd_analytics_ms": [],
        "risk_engine_ms": [],
        "alert_evaluation_ms": [],
        "total_pipeline_ms": [],
    }

    person_counts = []

    for f_idx in range(num_frames):
        t_frame_start = time.perf_counter()

        # 1. Decode
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        t_decode = (time.perf_counter() - t0) * 1000.0

        # 2. Preprocess & Camera Motion Compensation
        t0 = time.perf_counter()
        quality = quality_checker.check_quality(frame)
        processed_frame = preprocessor.preprocess(frame)
        motion = cmc.estimate_motion(frame)
        t_prep = (time.perf_counter() - t0) * 1000.0

        # 3. YOLO Detection
        t0 = time.perf_counter()
        det_ctx = DetectionContext(frame_number=f_idx, timestamp=f_idx * 0.05, original_frame_shape=frame.shape[:2])
        detections = detector.detect(frame, det_ctx)
        t_det = (time.perf_counter() - t0) * 1000.0

        person_counts.append(len(detections.detections))

        # 4. Tracking
        t0 = time.perf_counter()
        track_ctx = TrackingContext(frame_number=f_idx, timestamp=f_idx * 0.05, original_frame_shape=frame.shape[:2])
        tracks = tracker.update(detections, track_ctx)
        t_track = (time.perf_counter() - t0) * 1000.0

        # 5. Crowd Analytics
        t0 = time.perf_counter()
        an_ctx = AnalyticsContext(
            zone_id="zone_main",
            frame_number=f_idx,
            timestamp=f_idx * 0.05,
            zone_polygon=zone_poly,
            zone_capacity=100,
        )
        count_res = people_counter.update(tracks.tracks, an_ctx, 20.0)
        density_res = density_estimator.estimate(tracks.tracks, an_ctx, 20.0)
        occ_res = occupancy_monitor.analyze(count_res.current_count, 100, f_idx * 0.05, "zone_main")
        mov_res = movement_analyzer.analyze(tracks.tracks, an_ctx, 20.0)
        flow_res = flow_analyzer.analyze(tracks.tracks, an_ctx, 20.0, [], [])
        bn_res = bottleneck_detector.analyze(density_res, flow_res, mov_res, an_ctx)
        t_analytics = (time.perf_counter() - t0) * 1000.0

        # 6. Risk Engine
        t0 = time.perf_counter()
        risk_res = risk_engine.assess(
            zone_id="zone_main",
            timestamp=f_idx * 0.05,
            density_result=density_res,
            occupancy_result=occ_res,
            movement_result=mov_res,
            flow_result=flow_res,
            bottleneck_result=bn_res,
        )
        t_risk = (time.perf_counter() - t0) * 1000.0

        # 7. Alerts
        t0 = time.perf_counter()
        alerts = alert_manager.check_alerts(risk_res)
        t_alerts = (time.perf_counter() - t0) * 1000.0

        t_total = (time.perf_counter() - t_frame_start) * 1000.0

        stage_latencies["decode_ms"].append(t_decode)
        stage_latencies["preprocess_and_cmc_ms"].append(t_prep)
        stage_latencies["yolo_detection_ms"].append(t_det)
        stage_latencies["tracking_ms"].append(t_track)
        stage_latencies["crowd_analytics_ms"].append(t_analytics)
        stage_latencies["risk_engine_ms"].append(t_risk)
        stage_latencies["alert_evaluation_ms"].append(t_alerts)
        stage_latencies["total_pipeline_ms"].append(t_total)

    cap.release()

    # Compute statistics
    def calc_stats(vals):
        if not vals:
            return {"mean": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
        s = sorted(vals)
        return {
            "mean": round(statistics.mean(s), 2),
            "p50": round(statistics.median(s), 2),
            "p90": round(s[int(0.9 * len(s))], 2),
            "p95": round(s[int(0.95 * len(s))], 2),
            "max": round(max(s), 2),
        }

    summary = {stage: calc_stats(lats) for stage, lats in stage_latencies.items()}
    total_mean = summary["total_pipeline_ms"]["mean"]
    effective_fps = round(1000.0 / total_mean, 1) if total_mean > 0 else 0.0

    print("\n" + "=" * 65)
    print("SENTINEL LATENCY & PERFORMANCE BREAKDOWN (CPU)")
    print("=" * 65)
    print(f"{'Pipeline Stage':<28} | {'Mean (ms)':<9} | {'P50 (ms)':<8} | {'P95 (ms)':<8}")
    print("-" * 65)
    for stage, stats in summary.items():
        print(f"{stage:<28} | {stats['mean']:<9.2f} | {stats['p50']:<8.2f} | {stats['p95']:<8.2f}")
    print("-" * 65)
    print(f"Average Total Pipeline Latency : {total_mean:.2f} ms")
    print(f"Effective Throughput (FPS)     : {effective_fps:.1f} FPS")
    print(f"Average People per Frame       : {statistics.mean(person_counts):.1f}")
    print("=" * 65)

    # Save to json
    results_path = settings.base_dir / "data" / "benchmark_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "num_frames": len(stage_latencies["total_pipeline_ms"]),
            "effective_fps": effective_fps,
            "stages": summary,
            "device": "cpu",
        }, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    run_benchmark(num_frames=30)
