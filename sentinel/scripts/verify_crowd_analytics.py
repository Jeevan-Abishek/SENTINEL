"""
Verification and benchmark script for SENTINEL Crowd Analytics module.
Tests:
  1. Density calculation with calibrated area, rate-of-change, and 5-min forward trend.
  2. Occupancy ratio, headroom headcount, and safe default with zero capacity.
  3. Movement speed, heading, stagnation ratio, velocity drop, and directional convergence.
  4. Flow rates, inflow/outflow ratio, net accumulation, and conflict detection.
  5. Bottleneck detection, congestion level, stagnation overlap, and 5-min early warning.
  6. Unified CrowdAnalyticsEngine, Risk Engine factor export (to_risk_factors), and memory/speed benchmark.
"""

import sys
import time
import os
import tracemalloc
from pathlib import Path
import numpy as np

# Ensure sentinel package is on sys.path
sentinel_dir = Path(__file__).resolve().parent.parent
if str(sentinel_dir) not in sys.path:
    sys.path.insert(0, str(sentinel_dir))

from app.analytics.models import (
    AnalyticsConfig,
    AnalyticsContext,
    DensityResult,
    OccupancyResult,
    MovementResult,
    FlowResult,
    BottleneckResult,
    ZoneAnalyticsSummary,
    SceneAnalyticsSummary,
)
from app.analytics.density import (
    DensityEstimator,
    calculate_zone_density,
    calculate_density_rate_of_change,
    project_density_trend,
    estimate_lead_time_to_threshold,
)
from app.analytics.occupancy import (
    OccupancyMonitor,
    calculate_occupancy_ratio,
    calculate_headroom,
    calculate_headroom_ratio,
    classify_occupancy_status,
)
from app.analytics.movement import (
    MovementAnalyzerWrapper,
    compute_vector_speed,
    compute_heading_degrees,
    compute_stagnation_ratio,
    compute_directional_convergence,
)
from app.analytics.flow import (
    FlowAnalyzerWrapper,
    calculate_inflow_outflow_ratio,
    calculate_net_accumulation_rate,
)
from app.analytics.bottleneck import (
    BottleneckDetectorWrapper,
    classify_congestion_level,
    detect_choke_point_condition,
)
from app.analytics.engine import CrowdAnalyticsEngine
from app.tracking.models import Track


def verify_density_estimation():
    print("[1/6] Verifying Density calculation, rate-of-change & 5-min trend projection...")
    cfg = AnalyticsConfig(warning_density=3.0, critical_density=5.0)
    estimator = DensityEstimator(cfg)

    # 1. Calibrated zone density
    dens = calculate_zone_density(count=50, area_sqm=100.0)
    assert dens == 0.5, f"Expected 0.5 p/m², got {dens}"

    # 2. Rate-of-change: density rises from 1.0 to 1.6 over 60 seconds (1 minute)
    # history = [(0.0, 1.0), (60.0, 1.6)] -> +0.6 p/(m²·min)
    roc = calculate_density_rate_of_change(
        history=[(0.0, 1.0), (60.0, 1.6)],
        min_dt_seconds=0.05,
    )
    assert abs(roc - 0.6) < 1e-4, f"Expected +0.6 p/m²/min, got {roc}"

    # 3. 5-min forward trend projection: 1.6 + 0.6 * 5 = 4.6 p/m²
    proj = project_density_trend(
        current_density=1.6,
        rate_per_min=roc,
        horizon_minutes=5.0,
    )
    assert abs(proj - 4.6) < 1e-4, f"Expected 4.6 p/m², got {proj}"

    # 4. Lead time to critical threshold (5.0 p/m²) from 1.6 at 0.6 p/m²/min
    # (5.0 - 1.6) / 0.6 = 5.67 min -> within 6 min horizon
    lead_time = estimate_lead_time_to_threshold(
        current_density=1.6,
        rate_per_min=roc,
        threshold=5.0,
        horizon_minutes=6.0,
    )
    assert lead_time is not None
    assert abs(lead_time - 5.7) <= 0.1, f"Expected ~5.7 min, got {lead_time}"

    # 5. Full estimate() method
    tracks = [
        Track(track_id=i, bbox=[10 * i, 10 * i, 10 * i + 20, 10 * i + 40], confidence=0.90, is_validated=True)
        for i in range(20)
    ]
    ctx = AnalyticsContext(
        frame_number=100,
        timestamp=3.33,
        zone_id="main_square",
        area_sqm=50.0,
        frame_rate=30.0,
    )
    res = estimator.estimate(tracks, ctx, frame_rate=30.0, area_sqm=50.0)
    assert isinstance(res, DensityResult)
    assert res.density_p_sqm == 0.4
    assert res.status == "safe"
    assert res.risk_score >= 0.0
    print("  -> Passed: Calibrated density, rate-of-change, and 5-min forward projection verified.")


def verify_occupancy_monitoring():
    print("[2/6] Verifying Occupancy ratio, headroom & safe defaults...")
    cfg = AnalyticsConfig(occupancy_warning_threshold=0.80, occupancy_critical_threshold=0.90)
    monitor = OccupancyMonitor(cfg)

    # 1. Normal occupancy
    ratio = calculate_occupancy_ratio(count=70, capacity=100)
    assert ratio == 0.70, f"Expected 0.70, got {ratio}"
    headroom = calculate_headroom(count=70, capacity=100)
    assert headroom == 30, f"Expected 30, got {headroom}"
    hr_ratio = calculate_headroom_ratio(count=70, capacity=100)
    assert abs(hr_ratio - 0.30) < 1e-4, f"Expected 0.30, got {hr_ratio}"
    assert classify_occupancy_status(ratio, warning_threshold=0.80, critical_threshold=0.90) == "safe"

    # 2. Warning occupancy
    ratio_warn = calculate_occupancy_ratio(count=85, capacity=100)
    assert classify_occupancy_status(ratio_warn, warning_threshold=0.80, critical_threshold=0.90) == "warning"

    # 3. Critical occupancy
    ratio_crit = calculate_occupancy_ratio(count=95, capacity=100)
    assert classify_occupancy_status(ratio_crit, warning_threshold=0.80, critical_threshold=0.90) == "critical"

    # 4. Safe fallback with zero/negative capacity
    zero_ratio = calculate_occupancy_ratio(count=50, capacity=0)
    assert zero_ratio == 0.0, f"Expected 0.0 with capacity=0, got {zero_ratio}"
    assert calculate_headroom(count=50, capacity=0) == 0

    # 5. Full analyze() method
    res = monitor.analyze(count=40, capacity=50, timestamp=3.33, zone_id="turnstile_a")
    assert isinstance(res, OccupancyResult)
    assert res.occupancy_ratio == 0.80
    assert res.headroom == 10
    assert res.headroom_ratio == 0.20
    assert res.status == "warning"
    print("  -> Passed: Occupancy ratio, headroom, and zero-capacity fallbacks verified.")


def verify_movement_dynamics():
    print("[3/6] Verifying Movement speed, heading, stagnation & directional convergence...")
    cfg = AnalyticsConfig()
    analyzer = MovementAnalyzerWrapper(cfg)

    # 1. Vector speed & heading
    # Move vx=3, vy=4 -> speed = 5.0, heading = arctan2(4, 3) = ~53.13 deg
    spd = compute_vector_speed(vx=3.0, vy=4.0)
    assert spd == 5.0, f"Expected 5.0, got {spd}"
    heading = compute_heading_degrees(vx=3.0, vy=4.0)
    assert abs(heading - 53.13) < 0.1, f"Expected ~53.13 deg, got {heading}"

    # 2. Stagnation ratio
    speeds = [0.1, 0.05, 0.15, 1.2, 1.5]  # 3 out of 5 below 0.3 threshold
    stag_ratio, stagnant_count = compute_stagnation_ratio(speeds, threshold=0.3)
    assert abs(stag_ratio - 0.6) < 1e-4, f"Expected 0.6, got {stag_ratio}"
    assert stagnant_count == 3, f"Expected 3 stagnant tracks, got {stagnant_count}"

    # 3. Directional convergence (funneling toward centroid)
    # Positions funneling toward (100, 100)
    centroids = [(50.0, 100.0), (100.0, 50.0), (150.0, 100.0), (100.0, 150.0)]
    velocities = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    conv = compute_directional_convergence(centroids, velocities)
    assert conv > 0.8, f"Expected strong convergence > 0.8, got {conv}"

    # 4. Full analyze() method
    tracks = [
        Track(track_id=1, bbox=[10, 10, 20, 20], confidence=0.90, centroid_history=[[15.0, 15.0], [15.05, 15.05]], is_validated=True),
        Track(track_id=2, bbox=[30, 30, 40, 40], confidence=0.90, centroid_history=[[35.0, 35.0], [35.02, 35.02]], is_validated=True),
        Track(track_id=3, bbox=[50, 50, 60, 60], confidence=0.90, centroid_history=[[55.0, 55.0], [67.0, 64.0]], is_validated=True),
    ]
    ctx = AnalyticsContext(frame_number=10, timestamp=0.33, zone_id="choke_narrow")
    res = analyzer.analyze(tracks, ctx, frame_rate=30.0)
    assert isinstance(res, MovementResult)
    assert res.stagnation_ratio > 0.6
    assert res.low_movement_count == 2
    print("  -> Passed: Movement metrics, headings, stagnation, and convergence verified.")


def verify_flow_dynamics():
    print("[4/6] Verifying Flow rates, net accumulation & counter-flow conflict...")
    cfg = AnalyticsConfig()
    flow_analyzer = FlowAnalyzerWrapper(cfg)

    # 1. Inflow/outflow ratio & net accumulation
    io_ratio = calculate_inflow_outflow_ratio(inflow=30.0, outflow=15.0)
    assert io_ratio == 2.0, f"Expected 2.0, got {io_ratio}"
    assert calculate_inflow_outflow_ratio(inflow=10.0, outflow=0.0) == 100.0

    net_acc = calculate_net_accumulation_rate(inflow=25.0, outflow=10.0)
    assert net_acc == 15.0, f"Expected +15.0 p/min, got {net_acc}"

    # 2. Counter-flow conflict detection
    # Group A heading East (0°), Group B heading West (180°) -> 180° opposition
    tracks = [
        Track(track_id=1, bbox=[10, 10, 20, 20], confidence=0.90, centroid_history=[[15.0, 15.0], [17.0, 15.0]], is_validated=True),
        Track(track_id=2, bbox=[15, 15, 25, 25], confidence=0.90, centroid_history=[[20.0, 20.0], [22.0, 20.0]], is_validated=True),
        Track(track_id=3, bbox=[20, 20, 30, 30], confidence=0.90, centroid_history=[[25.0, 25.0], [23.0, 25.0]], is_validated=True),
        Track(track_id=4, bbox=[25, 25, 35, 35], confidence=0.90, centroid_history=[[30.0, 30.0], [28.0, 30.0]], is_validated=True),
    ]
    ctx = AnalyticsContext(frame_number=50, timestamp=1.66, zone_id="corridor_main")
    res = flow_analyzer.analyze(tracks, ctx, frame_rate=30.0)
    assert isinstance(res, FlowResult)
    assert res.counter_flow_conflict_score > 0.15, f"Expected counter-flow conflict > 0.15, got {res.counter_flow_conflict_score}"
    assert res.flow_conflict_detected is True
    print("  -> Passed: Inflow/outflow ratio, net accumulation, and counter-flow conflict verified.")


def verify_bottleneck_detection():
    print("[5/6] Verifying Bottleneck detection, congestion level & 5-min early warning...")
    cfg = AnalyticsConfig(
        choke_density_threshold=2.5,
        stagnation_speed_threshold=0.3,
        bottleneck_window_seconds=15.0,
    )
    detector = BottleneckDetectorWrapper(cfg)

    # 1. Congestion classification
    assert classify_congestion_level(0.10) == "none"
    assert classify_congestion_level(0.35) == "low"
    assert classify_congestion_level(0.55) == "medium"
    assert classify_congestion_level(0.75) == "high"
    assert classify_congestion_level(0.90) == "critical"

    # 2. Choke point condition: high density (3.2) + low speed (0.15)
    is_choke = detect_choke_point_condition(
        density=3.2,
        speed=0.15,
        stagnation_ratio=0.8,
        density_thresh=2.5,
        speed_thresh=0.5,
    )
    assert is_choke is True

    # 3. Analyze choke zone with early warning trigger
    ctx = AnalyticsContext(frame_number=100, timestamp=3.33, zone_id="gate_north_choke", area_sqm=30.0)
    density_res = DensityResult(
        zone_id="gate_north_choke",
        timestamp=3.33,
        avg_density=3.5,
        max_density=4.0,
        density_rate_of_change=0.25,
        projected_density_5min=4.75,
        lead_time_to_critical_min=3.8,
        status="warning",
    )
    movement_res = MovementResult(
        zone_id="gate_north_choke",
        timestamp=3.33,
        avg_speed=0.12,
        stagnation_ratio=0.85,
        directional_convergence=0.72,
        velocity_drop_ratio=0.35,
    )
    flow_res = FlowResult(
        zone_id="gate_north_choke",
        timestamp=3.33,
        entry_flow=35.0,
        exit_flow=10.0,
        net_accumulation_rate_ppm=25.0,
        counter_flow_conflict_score=0.22,
        flow_conflict_detected=True,
    )
    bn_res = detector.analyze(density_res, flow_res, movement_res, ctx)
    assert isinstance(bn_res, BottleneckResult)
    assert bn_res.bottleneck_score > 0.50
    assert bn_res.congestion_level in ["medium", "high", "critical"]
    assert bn_res.early_warning_active is True
    assert bn_res.lead_time_min is not None and bn_res.lead_time_min <= 5.0
    print("  -> Passed: Bottleneck detection, congestion levels, and 5-min early warning verified.")


def verify_unified_engine_and_risk_integration():
    print("[6/6] Verifying CrowdAnalyticsEngine, Risk Engine factor export & footprint benchmark...")
    tracemalloc.start()
    start_time = time.perf_counter()

    engine = CrowdAnalyticsEngine(AnalyticsConfig())

    calibrated_zones = {
        "ingress_gate": {
            "polygon": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "area_sqm": 50.0,
            "max_capacity": 40,
        },
        "assembly_square": {
            "polygon": [[200, 0], [600, 0], [600, 400], [200, 400]],
            "area_sqm": 300.0,
            "max_capacity": 200,
        },
    }

    # Generate synthetic tracks
    test_tracks = []
    # 20 people in ingress_gate (foot contacts within [0, 100])
    for i in range(20):
        cx = 17.5 + (i % 5) * 15
        cy = 25.0 + (i // 5) * 15
        test_tracks.append(
            Track(
                track_id=i + 1,
                bbox=[10 + (i % 5) * 15, 10 + (i // 5) * 15, 25 + (i % 5) * 15, 40 + (i // 5) * 15],
                confidence=0.90,
                centroid_history=[[cx, cy - 0.25], [cx, cy]],
                is_validated=True,
            )
        )
    # 15 people in assembly_square
    for i in range(15):
        cx = 265.0 + (i % 4) * 50
        cy = 80.0 + (i // 4) * 50
        test_tracks.append(
            Track(
                track_id=100 + i,
                bbox=[250 + (i % 4) * 50, 50 + (i // 4) * 50, 280 + (i % 4) * 50, 110 + (i // 4) * 50],
                confidence=0.85,
                centroid_history=[[cx - 1.0, cy - 1.0], [cx, cy]],
                is_validated=True,
            )
        )

    # 1. Analyze single zone
    zone_sum = engine.analyze_zone(
        zone_id="ingress_gate",
        tracks=test_tracks,
        frame_number=1,
        timestamp=0.033,
        zone_config=calibrated_zones["ingress_gate"],
        frame_rate=30.0,
    )
    assert isinstance(zone_sum, ZoneAnalyticsSummary)
    assert zone_sum.zone_id == "ingress_gate"
    assert zone_sum.density.density_p_sqm == 0.4
    assert zone_sum.occupancy.occupancy_ratio == 0.50
    assert zone_sum.occupancy.headroom == 20

    # 2. Analyze global scene
    scene_sum = engine.analyze_scene(
        tracks=test_tracks,
        frame_number=1,
        timestamp=0.033,
        zones=calibrated_zones,
        frame_rate=30.0,
    )
    assert isinstance(scene_sum, SceneAnalyticsSummary)
    assert "ingress_gate" in scene_sum.zones
    assert "assembly_square" in scene_sum.zones
    assert scene_sum.density.density_p_sqm > 0
    assert scene_sum.risk_level in ["SAFE", "WARNING", "CRITICAL"]

    # 3. Test Risk Engine factors export
    risk_factors = engine.to_risk_factors(
        density_res=zone_sum.density,
        occupancy_res=zone_sum.occupancy,
        movement_res=zone_sum.movement,
        flow_res=zone_sum.flow,
        bottleneck_res=zone_sum.bottleneck,
    )
    assert "density" in risk_factors
    assert "occupancy" in risk_factors
    assert "movement" in risk_factors
    assert "flow" in risk_factors
    assert "bottleneck" in risk_factors
    assert "overall_risk" in risk_factors
    for k, v in risk_factors.items():
        if isinstance(v, (int, float)):
            assert 0.0 <= v <= 100.0, f"Factor {k}={v} out of bounds [0, 100]"

    # 4. Performance & Memory footprint check
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  -> Execution time: {elapsed_ms:.2f} ms")
    print(f"  -> Traced Python memory: Current={current_mem / 1024 / 1024:.2f} MB, Peak={peak_mem / 1024 / 1024:.2f} MB")
    assert elapsed_ms < 100.0, f"Execution too slow: {elapsed_ms:.2f} ms"
    assert (peak_mem / 1024 / 1024) < 15.0, f"Memory exceeded peak threshold: {peak_mem / 1024 / 1024:.2f} MB"
    print("  -> Passed: CrowdAnalyticsEngine and Risk Engine factors verified with ultra-low footprint.")


if __name__ == "__main__":
    print("=" * 70)
    print("🌿 SENTINEL Crowd Analytics & Predictive Engine Verification Suite")
    print("=" * 70)
    verify_density_estimation()
    verify_occupancy_monitoring()
    verify_movement_dynamics()
    verify_flow_dynamics()
    verify_bottleneck_detection()
    verify_unified_engine_and_risk_integration()
    print("=" * 70)
    print("✅ ALL 6 CROWD ANALYTICS VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
