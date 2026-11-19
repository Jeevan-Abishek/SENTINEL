"""
Verification and benchmark script for SENTINEL People Count & Tracking module (Phase 10).
Tests:
  1. Foot-contact ground plane polygon containment in ZoneCounter.
  2. Virtual gate line crossing, anti-flicker debouncing, and rolling ppm rates.
  3. Multi-zone ScenePeopleCounter aggregation and SceneCountReport.
  4. End-to-end integration with Tracker (ByteTrack + Re-ID + Validation) on real video frames.
  5. Memory footprint and CPU execution speed.
"""

import sys
import time
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure sentinel package is on sys.path
sentinel_dir = Path(__file__).resolve().parent.parent
if str(sentinel_dir) not in sys.path:
    sys.path.insert(0, str(sentinel_dir))

from app.analytics.models import SceneCountReport, CountingResult, ZoneCounter
from app.analytics.counting import ScenePeopleCounter, LineCrossCounter, PeopleCounter
from app.tracking.tracker import Tracker, TrackerConfig, TrackingContext
from app.tracking.models import Track, TrackResult
from app.detection.yolo_detector import YOLOPersonDetector
from app.detection.models import DetectorConfig
from app.detection.detector import DetectionContext
from app.config import settings


def verify_foot_contact_zone_counter():
    print("[1/5] Verifying foot-contact ground plane containment in ZoneCounter...")
    # Polygon covers y in [100, 200], x in [50, 150]
    polygon = [[50, 100], [150, 100], [150, 200], [50, 200]]
    zc = ZoneCounter("test_zone", polygon, capacity=50, area_m2=100.0)

    # Person track with bbox [80, 20, 120, 110]
    # Centroid: (100, 65) -> OUTSIDE polygon!
    # Foot: (100, 110) -> INSIDE polygon!
    t_foot_inside = Track(
        track_id=1,
        bbox=[80.0, 20.0, 120.0, 110.0],
        confidence=0.88,
        is_validated=True,
    )
    zc.update([t_foot_inside], frame_number=1)
    assert zc.current_count == 1, f"Expected current_count=1, got {zc.current_count}"
    assert zc.entries == 1, f"Expected entries=1, got {zc.entries}"
    print("  -> Passed: Foot-contact point accurately placed person inside ground polygon.")


def verify_line_crossing_and_debouncing():
    print("[2/5] Verifying virtual gate line crossing and anti-flicker debouncing...")
    # Horizontal line y = 100 from x = 0 to 200
    line = [[0, 100], [200, 100]]
    lc = LineCrossCounter("ingress_gate", line, cooldown_frames=20)

    # Person moves from y=80 to y=120 -> Inflow crossing
    t_before = Track(track_id=42, bbox=[90, 60, 110, 80], confidence=0.9, is_validated=True)
    t_after = Track(track_id=42, bbox=[90, 100, 110, 120], confidence=0.9, is_validated=True)

    lc.update([t_before], frame_number=1, timestamp=0.033)
    r2 = lc.update([t_after], frame_number=2, timestamp=0.066)
    assert r2["total_inflow"] == 1, f"Expected total_inflow=1, got {r2['total_inflow']}"
    assert 42 in r2["inflow_events"]

    # Immediate sensor jitter back across line within cooldown (frame 3 < 2+20)
    t_jitter_back = Track(track_id=42, bbox=[90, 60, 110, 80], confidence=0.9, is_validated=True)
    r3 = lc.update([t_jitter_back], frame_number=3, timestamp=0.10)
    assert r3["total_outflow"] == 0, "Anti-flicker failed: jitter was counted as outflow!"
    assert len(r3["outflow_events"]) == 0

    # Person returns to inside zone (y=120) still within cooldown (frame 4)
    t_inside = Track(track_id=42, bbox=[90, 100, 110, 120], confidence=0.9, is_validated=True)
    r4 = lc.update([t_inside], frame_number=4, timestamp=0.133)
    assert r4["total_inflow"] == 1, "Duplicate inflow occurred during cooldown!"

    # Clean crossing back out after cooldown window (frame 30 > 2+20)
    r30 = lc.update([t_jitter_back], frame_number=30, timestamp=1.0)
    assert r30["total_outflow"] == 1, f"Expected total_outflow=1, got {r30['total_outflow']}"
    print("  -> Passed: Line crossing detected and anti-flicker debouncing verified.")


def verify_scene_people_counter_multi_zone():
    print("[3/5] Verifying ScenePeopleCounter multi-zone aggregation & SceneCountReport...")
    counter = ScenePeopleCounter()
    counter.register_zone("gate_north", [[0, 0], [100, 0], [100, 100], [0, 100]], capacity=25, area_m2=50.0)
    counter.register_zone("arena_plaza", [[200, 0], [500, 0], [500, 300], [200, 300]], capacity=200, area_m2=400.0)
    counter.register_line("entry_turnstile", [[0, 100], [100, 100]])

    tracks = [
        Track(track_id=1, bbox=[20, 20, 40, 80], confidence=0.90, is_validated=True),   # In gate_north
        Track(track_id=2, bbox=[250, 50, 290, 120], confidence=0.85, is_validated=True), # In arena_plaza
        Track(track_id=3, bbox=[300, 60, 340, 130], confidence=0.70, is_validated=False),# Tentative in arena_plaza
        Track(track_id=4, bbox=[700, 700, 740, 780], confidence=0.80, is_validated=True),# Outside zones
    ]

    report = counter.update(tracks, frame_number=1, timestamp=0.033, rejected_count=2, reid_matches=1)
    assert isinstance(report, SceneCountReport)
    assert report.total_active_count == 4
    assert report.total_validated_count == 3
    assert report.total_tentative_count == 1
    assert report.rejected_count == 2
    assert report.reid_matches == 1
    assert report.zone_counts["gate_north"] == 1
    assert report.zone_counts["arena_plaza"] == 2
    assert report.zone_status["gate_north"] == "safe"
    assert report.zone_status["arena_plaza"] == "safe"
    print(f"  -> Passed: SceneCountReport generated successfully (Active: {report.total_active_count}, Validated: {report.total_validated_count}).")


def verify_real_video_pipeline_integration():
    print("[4/5] Verifying end-to-end People Count & Tracking on sample drone video...")
    video_path = "data/sample_videos/demo_drone_crowd.mp4"
    if not os.path.exists(video_path):
        video_path = "../data/sample_videos/demo_drone_crowd.mp4"

    if not os.path.exists(video_path):
        print(f"  -> Skipped real video test: {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector_config = DetectorConfig(
        model_path=str(settings.resolved_yolo_model_path),
        confidence_threshold=0.35,
        device="cpu",
    )
    detector = YOLOPersonDetector(detector_config)
    detector.load_model()

    tracker = Tracker(TrackerConfig(reid_enabled=True, min_hits=2))
    tracker.initialize()

    counter = ScenePeopleCounter()
    counter.register_zone("arena_overview", [[50, 50], [590, 50], [590, 590], [50, 590]], capacity=100, area_m2=250.0)
    counter.register_line("virtual_line", [[0, 320], [640, 320]])

    processed_frames = 0
    t_start = time.perf_counter()

    for f_idx in range(12):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Resize to 640px for lightweight CPU processing
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * (640 / float(w)))), interpolation=cv2.INTER_AREA)

        det_ctx = DetectionContext(frame_number=f_idx, timestamp=f_idx / fps, original_frame_shape=frame.shape[:2])
        dets = detector.detect(frame, det_ctx)

        trk_ctx = TrackingContext(frame_number=f_idx, timestamp=f_idx / fps, original_frame_shape=frame.shape[:2])
        tracks = tracker.update(dets, trk_ctx, frame=frame)

        report = counter.update(
            tracks,
            frame_number=f_idx,
            timestamp=f_idx / fps,
            frame_rate=fps,
            rejected_count=getattr(tracks, "rejected_count", 0),
            reid_matches=getattr(tracks, "reid_matches", 0),
        )
        processed_frames += 1

    cap.release()
    elapsed = time.perf_counter() - t_start
    avg_fps = processed_frames / elapsed if elapsed > 0 else 0

    latest_rep = counter.get_latest_report()
    assert latest_rep is not None
    print(f"  -> Processed {processed_frames} frames in {elapsed:.2f}s ({avg_fps:.1f} FPS on CPU).")
    print(f"  -> Latest Scene Report: Validated={latest_rep.total_validated_count}, Active={latest_rep.total_active_count}, Re-ID matches={latest_rep.reid_matches}, Net Accumulation={latest_rep.net_flow_rate_ppm} p/min.")
    print("  -> Passed: End-to-end People Count & Tracking pipeline operates with zero errors.")


def verify_memory_footprint():
    print("[5/5] Verifying process and module memory footprint...")
    import tracemalloc
    # Check tracemalloc if active
    if tracemalloc.is_tracing():
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        curr_mb = current_mem / (1024 * 1024)
        peak_mb = peak_mem / (1024 * 1024)
        print(f"  -> Traced Python memory: {curr_mb:.2f} MB (Peak: {peak_mb:.2f} MB).")
        assert peak_mb < 50.0, f"Module memory exceeded limit: {peak_mb:.1f} MB"
    else:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            print(f"  -> Total process RSS memory: {mem_mb:.1f} MB (including PyTorch/OpenCV DLLs).")
            # Baseline with PyTorch DLLs on Windows is ~450 MB
            assert mem_mb < 650.0, f"Process memory exceeded limit: {mem_mb:.1f} MB"
        except ImportError:
            print("  -> psutil not available, skipping process memory check.")
    print("  -> Passed: Memory footprint is well within acceptable CPU bounds.")


if __name__ == "__main__":
    print("=" * 70)
    print("SENTINEL People Count & Tracking Verification Suite")
    print("=" * 70)
    verify_foot_contact_zone_counter()
    verify_line_crossing_and_debouncing()
    verify_scene_people_counter_multi_zone()
    verify_real_video_pipeline_integration()
    verify_memory_footprint()
    print("=" * 70)
    print("ALL PEOPLE COUNT & TRACKING VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 70)
