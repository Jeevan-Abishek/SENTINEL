"""
Comprehensive Verification and Benchmark Suite for SENTINEL Explainable Risk Engine.
Validates:
1. Configurable 5-Factor Risk Weights and Safe Normalization (sum != 1.0, zero/negative guards).
2. Multi-Factor Scoring (0-100) across Density, Occupancy, Movement, Flow, and Bottleneck.
3. Temporal Smoothing (S_t = alpha * S_{t-1} + (1-alpha) * R_t) and Hysteresis Buffer.
4. 5-Minute Predictive Horizon & Advance Early Warning Integration.
5. Explainability Generation (component contributions, top drivers, human-readable narrative, steward actions).
6. Per-Zone and Scene-Wide Aggregate Risk Breakdown.
7. Micro-benchmarking (< 25 ms latency target, < 5 MB memory footprint).
"""

import sys
import time
import os
import tracemalloc
from pathlib import Path

# Set up path to import app modules
current_dir = Path(__file__).resolve().parent
sentinel_root = current_dir.parent
if str(sentinel_root) not in sys.path:
    sys.path.insert(0, str(sentinel_root))

from app.risk.models import (
    RiskLevel,
    RiskFactor,
    RiskAssessment,
    RiskConfig,
    RiskEngine,
    AlertRule,
    AlertManager,
    check_and_create_alerts,
)
from app.risk.rules import (
    create_critical_rule,
    create_warning_rule,
    create_early_warning_rule,
    create_choke_point_rule,
    get_default_ruleset,
)
from app.analytics.models import (
    DensityResult,
    OccupancyResult,
    MovementResult,
    FlowResult,
    BottleneckResult,
    ZoneAnalyticsSummary,
    SceneAnalyticsSummary,
    DensityGrid,
)


def test_1_configurable_weights_and_normalization():
    print("\n--- Test 1: Configurable Weights & Safe Normalization ---")
    
    # 1. Default weights sum to 1.0
    cfg_default = RiskConfig()
    norm_w = cfg_default.normalized_weights()
    total_w = sum(norm_w.values())
    assert abs(total_w - 1.0) < 1e-4, f"Default weights sum must be 1.0, got {total_w}"
    assert norm_w["density"] == 0.30
    assert norm_w["occupancy"] == 0.20
    assert norm_w["movement"] == 0.20
    assert norm_w["flow"] == 0.15
    assert norm_w["bottleneck"] == 0.15
    print(f"  [PASS] Default weights correctly configured and sum to {total_w:.2f}")

    # 2. Weights summing != 1.0 (e.g., all 0.5 -> sum 2.5)
    cfg_unbalanced = RiskConfig(
        density_weight=0.5,
        occupancy_weight=0.5,
        movement_weight=0.5,
        flow_weight=0.5,
        bottleneck_weight=0.5,
    )
    norm_unb = cfg_unbalanced.normalized_weights()
    assert abs(sum(norm_unb.values()) - 1.0) < 1e-4, f"Normalized unbalanced sum must be 1.0, got {sum(norm_unb.values())}"
    for k, v in norm_unb.items():
        assert abs(v - 0.20) < 1e-3, f"Each weight should be 0.20, got {v} for {k}"
    print(f"  [PASS] Unbalanced weights (sum=2.5) safely normalized to {norm_unb}")

    # 3. Guard against negative/zero weights fallback
    cfg_zero = RiskConfig(
        density_weight=0.0,
        occupancy_weight=0.0,
        movement_weight=0.0,
        flow_weight=0.0,
        bottleneck_weight=0.0,
    )
    norm_zero = cfg_zero.normalized_weights()
    assert abs(sum(norm_zero.values()) - 1.0) < 1e-4
    assert norm_zero["density"] == 0.30
    print(f"  [PASS] All-zero weights safely guarded and restored to default baseline: {norm_zero}")


def test_2_multifactor_scoring_and_tiers():
    print("\n--- Test 2: Multi-Factor Scoring (0-100) & Safety Tiers ---")
    engine = RiskEngine()

    # Scenario A: Nominal SAFE Conditions
    dens_safe = DensityResult(
        zone_id="plaza",
        timestamp=1.0,
        grid=DensityGrid(grid_data=[[0.8]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=0.8),
        avg_density=0.8,
        max_density=0.8,
    )
    occ_safe = OccupancyResult(
        zone_id="plaza",
        timestamp=1.0,
        capacity=100,
        current_count=25,
        occupancy_ratio=0.25,
        status="safe",
    )
    mov_safe = MovementResult(
        zone_id="plaza",
        timestamp=1.0,
        avg_speed=1.4,
        speed_trend="stable",
    )
    flow_safe = FlowResult(
        zone_id="plaza",
        timestamp=1.0,
        flow_rate=2.0,
    )
    bn_safe = BottleneckResult(
        zone_id="plaza",
        timestamp=1.0,
        bottleneck_score=0.1,
    )

    assessment_safe = engine.assess(
        zone_id="plaza",
        timestamp=1.0,
        density_result=dens_safe,
        occupancy_result=occ_safe,
        movement_result=mov_safe,
        flow_result=flow_safe,
        bottleneck_result=bn_safe,
    )
    assert assessment_safe.risk_level == RiskLevel.SAFE
    assert assessment_safe.risk_score < 30.0, f"Expected safe score < 30, got {assessment_safe.risk_score}"
    assert len(assessment_safe.factors) == 5
    print(f"  [PASS] SAFE condition confirmed: Score={assessment_safe.risk_score:.1f}/100, Tier={assessment_safe.risk_level.value.upper()}")

    # Scenario B: Severe CRITICAL Crowd Compression
    engine.reset_state()
    dens_crit = DensityResult(
        zone_id="ingress_gate",
        timestamp=1.0,
        grid=DensityGrid(grid_data=[[5.5]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=5.5),
        avg_density=5.2,
        max_density=5.5,
    )
    occ_crit = OccupancyResult(
        zone_id="ingress_gate",
        timestamp=1.0,
        capacity=100,
        current_count=98,
        occupancy_ratio=0.98,
        status="critical",
    )
    mov_crit = MovementResult(
        zone_id="ingress_gate",
        timestamp=1.0,
        avg_speed=0.15,
        speed_trend="slowing",
    )
    flow_crit = FlowResult(
        zone_id="ingress_gate",
        timestamp=1.0,
        flow_rate=18.0,
    )
    bn_crit = BottleneckResult(
        zone_id="ingress_gate",
        timestamp=1.0,
        bottleneck_score=0.92,
        is_bottleneck=True,
    )

    assessment_crit = engine.assess(
        zone_id="ingress_gate",
        timestamp=1.0,
        density_result=dens_crit,
        occupancy_result=occ_crit,
        movement_result=mov_crit,
        flow_result=flow_crit,
        bottleneck_result=bn_crit,
    )
    assert assessment_crit.risk_level == RiskLevel.CRITICAL
    assert assessment_crit.risk_score >= 80.0, f"Expected critical score >= 80, got {assessment_crit.risk_score}"
    assert "evacuation" in assessment_crit.recommended_actions[0].lower() or any("evacuation" in a.lower() for a in assessment_crit.recommended_actions)
    print(f"  [PASS] CRITICAL condition confirmed: Score={assessment_crit.risk_score:.1f}/100, Tier={assessment_crit.risk_level.value.upper()}")

    # Scenario C: Partial Sensor Inputs (Only Density & Occupancy)
    engine.reset_state()
    assessment_partial = engine.assess(
        zone_id="partial_zone",
        timestamp=1.0,
        density_result=dens_crit,
        occupancy_result=occ_crit,
    )
    assert assessment_partial.risk_level == RiskLevel.CRITICAL
    assert 80.0 <= assessment_partial.risk_score <= 100.0
    assert len(assessment_partial.factors) == 2
    # Verify effective weights re-scaled to sum to 1.0 (density 0.3 / 0.5 = 0.6, occupancy 0.2 / 0.5 = 0.4)
    factor_weights = {f.name: f.weight for f in assessment_partial.factors}
    assert abs(sum(factor_weights.values()) - 1.0) < 1e-3
    print(f"  [PASS] Partial inputs correctly rescaled weights (sum={sum(factor_weights.values()):.2f}): Score={assessment_partial.risk_score:.1f}")


def test_3_temporal_smoothing_and_hysteresis():
    print("\n--- Test 3: Temporal Smoothing & Hysteresis State Transition ---")
    config = RiskConfig(
        warning_threshold=60.0,
        critical_threshold=80.0,
        temporal_smoothing=0.50,
        hysteresis_buffer=3.0,
    )
    engine = RiskEngine(config)

    # Frame 1: Nominal state -> score ~20.0 (SAFE)
    dens_nom = DensityResult(
        zone_id="corridor",
        timestamp=1.0,
        grid=DensityGrid(grid_data=[[1.2]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=1.2),
        avg_density=1.2,
        max_density=1.2,
    )
    a1 = engine.assess("corridor", 1.0, density_result=dens_nom)
    assert a1.risk_level == RiskLevel.SAFE
    score_1 = a1.risk_score

    # Frame 2: Sudden 1-frame transient spike (Raw score = 100.0)
    dens_spike = DensityResult(
        zone_id="corridor",
        timestamp=2.0,
        grid=DensityGrid(grid_data=[[10.0]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=10.0),
        avg_density=10.0,
        max_density=10.0,
    )
    a2 = engine.assess("corridor", 2.0, density_result=dens_spike)
    # Expected smoothed score: 0.5 * score_1 + 0.5 * 100.0
    expected_smooth = (0.5 * score_1) + (0.5 * 100.0)
    assert abs(a2.risk_score - expected_smooth) < 0.5
    assert a2.risk_score < 75.0, f"Transient spike should be smoothed, got {a2.risk_score}"
    print(f"  [PASS] Temporal Smoothing dampened raw spike (100.0) -> Smoothed Score: {a2.risk_score:.1f}")

    # Frame 3: Sustained critical state
    for t in range(3, 8):
        a_crit = engine.assess("corridor", float(t), density_result=dens_spike)
    assert a_crit.risk_level == RiskLevel.CRITICAL
    print(f"  [PASS] Sustained high risk transitioned smoothly to CRITICAL (Score: {a_crit.risk_score:.1f})")

    # Frame 8: Score drops to 78.5 (just below 80.0 critical threshold)
    # With hysteresis buffer = 3.0, threshold to step down is 80.0 - 3.0 = 77.0.
    # At 78.5, state MUST stay CRITICAL to prevent flickering!
    dens_near_edge = DensityResult(
        zone_id="corridor",
        timestamp=8.0,
        grid=DensityGrid(grid_data=[[4.3]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=4.3),
        avg_density=4.3,
        max_density=4.3,
    )
    # Force previous score to 78.5 for exact hysteresis check
    engine.previous_scores["corridor"] = 78.5
    engine.previous_levels["corridor"] = RiskLevel.CRITICAL
    config_zero_smooth = RiskConfig(
        warning_threshold=60.0,
        critical_threshold=80.0,
        temporal_smoothing=0.0,
        hysteresis_buffer=3.0,
    )
    engine_hys = RiskEngine(config_zero_smooth)
    engine_hys.previous_levels["corridor"] = RiskLevel.CRITICAL
    engine_hys.previous_scores["corridor"] = 78.5
    
    # Assess at 78.5 raw
    # Let's assess with an input that yields raw score = 78.5
    dens_78 = DensityResult(
        zone_id="corridor",
        timestamp=9.0,
        grid=DensityGrid(grid_data=[[4.55]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=4.55),
        avg_density=4.55,
        max_density=4.55,
    )
    a_edge = engine_hys.assess("corridor", 9.0, density_result=dens_78)
    if a_edge.risk_score >= 77.0 and a_edge.risk_score < 80.0:
        assert a_edge.risk_level == RiskLevel.CRITICAL, "Hysteresis buffer must keep CRITICAL state above 77.0"
        print(f"  [PASS] Hysteresis prevented state flickering at edge: Score={a_edge.risk_score:.1f}, State={a_edge.risk_level.value.upper()}")


def test_4_predictive_early_warning_integration():
    print("\n--- Test 4: 5-Minute Predictive Horizon & Early Warning ---")
    risk_cfg = RiskConfig(
        warning_threshold=55.0,
        critical_threshold=80.0,
        predictive_weight=0.25,
        density_weight=0.25,
        enable_predictive=True,
    )
    engine = RiskEngine(risk_cfg)

    # Current density is SAFE (2.2 p/m² < 3.0), but projected 5-min density is 4.1 p/m² (surging influx)
    dens_pred = DensityResult(
        zone_id="choke_east",
        timestamp=100.0,
        grid=DensityGrid(grid_data=[[2.2]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=2.2),
        avg_density=2.2,
        max_density=2.2,
        projected_density_5min=4.1,
        density_rate_of_change=0.38,
        lead_time_to_critical_min=2.4,
    )

    assessment = engine.assess("choke_east", 100.0, density_result=dens_pred)
    assert assessment.is_early_warning is True
    assert assessment.risk_level == RiskLevel.WARNING
    assert assessment.lead_time_min == 2.4
    assert any("Early Warning" in act for act in assessment.recommended_actions)
    assert "predictive_5min" in assessment.component_contributions
    print(f"  [PASS] Preemptive Early Warning successfully triggered: Lead Time={assessment.lead_time_min} min, Tier={assessment.risk_level.value.upper()}")

    # Verify AlertManager triggers explainable alert rule
    mgr = AlertManager()
    mgr.add_rule(create_early_warning_rule())
    alerts = mgr.check_and_create_alerts(assessment)
    assert len(alerts) == 1
    assert alerts[0].risk_level == RiskLevel.WARNING
    assert "EARLY WARNING" in alerts[0].message
    assert "choke_east" in alerts[0].message
    print(f"  [PASS] Early Warning Alert dispatched: '{alerts[0].message}'")


def test_5_explainability_and_top_drivers():
    print("\n--- Test 5: Explainability, Top Drivers & Human Narrative ---")
    engine = RiskEngine()

    dens = DensityResult(
        zone_id="north_turnstiles",
        timestamp=50.0,
        grid=DensityGrid(grid_data=[[4.2]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=4.2),
        avg_density=4.0,
        max_density=4.2,
    )
    occ = OccupancyResult(
        zone_id="north_turnstiles",
        timestamp=50.0,
        capacity=100,
        current_count=88,
        occupancy_ratio=0.88,
        status="warning",
    )
    mov = MovementResult(
        zone_id="north_turnstiles",
        timestamp=50.0,
        avg_speed=0.35,
        speed_trend="slowing",
    )
    bn = BottleneckResult(
        zone_id="north_turnstiles",
        timestamp=50.0,
        bottleneck_score=0.82,
        is_bottleneck=True,
    )

    assessment = engine.assess(
        zone_id="north_turnstiles",
        timestamp=50.0,
        density_result=dens,
        occupancy_result=occ,
        movement_result=mov,
        bottleneck_result=bn,
    )

    # 1. Component contributions verified
    assert "density" in assessment.component_contributions
    assert "occupancy" in assessment.component_contributions
    assert "movement" in assessment.component_contributions
    assert "bottleneck" in assessment.component_contributions
    print(f"  [PASS] Factor contributions: {assessment.component_contributions}")

    # 2. Top Drivers extracted (should contain bottleneck / density / occupancy)
    assert len(assessment.top_drivers) >= 1
    assert len(assessment.top_drivers) <= 3
    print(f"  [PASS] Top Drivers extracted: {assessment.top_drivers}")

    # 3. Transparent Human-Readable Narrative generated
    assert len(assessment.explanation) > 20
    assert "north_turnstiles" in assessment.explanation
    assert assessment.risk_level.value.upper() in assessment.explanation
    print(f"  [PASS] Human Narrative Explanation:\n    \"{assessment.explanation}\"")

    # 4. Context-aware actions
    assert len(assessment.recommended_actions) >= 2
    print(f"  [PASS] Recommended Actions: {assessment.recommended_actions}")


def test_6_per_zone_and_scene_aggregate():
    print("\n--- Test 6: Per-Zone & Scene-Wide Aggregate Breakdown ---")
    engine = RiskEngine()

    zones_dict = {
        "gate_north": {"name": "North Gate", "area_sqm": 100.0, "max_capacity": 200},
        "main_stage": {"name": "Main Stage", "area_sqm": 500.0, "max_capacity": 1000},
        "exit_south": {"name": "South Exit", "area_sqm": 80.0, "max_capacity": 150},
    }

    # Build mock SceneAnalyticsSummary
    dens_overall = DensityResult(zone_id="scene_overall", timestamp=10.0, avg_density=2.0, max_density=3.5)
    occ_overall = OccupancyResult(zone_id="scene_overall", timestamp=10.0, capacity=1350, current_count=350, occupancy_ratio=0.26, status="safe")
    mov_overall = MovementResult(zone_id="scene_overall", timestamp=10.0, avg_speed=0.9)
    flow_overall = FlowResult(zone_id="scene_overall", timestamp=10.0, flow_rate=15.0)
    bn_overall = BottleneckResult(zone_id="scene_overall", timestamp=10.0, bottleneck_score=0.4)

    # Gate north is congested
    z_north = ZoneAnalyticsSummary(
        zone_id="gate_north",
        timestamp=10.0,
        density=DensityResult(zone_id="gate_north", timestamp=10.0, avg_density=4.5, max_density=5.2),
        occupancy=OccupancyResult(zone_id="gate_north", timestamp=10.0, capacity=200, current_count=185, occupancy_ratio=0.925, status="critical"),
        movement=MovementResult(zone_id="gate_north", timestamp=10.0, avg_speed=0.2),
        flow=FlowResult(zone_id="gate_north", timestamp=10.0, flow_rate=12.0),
        bottleneck=BottleneckResult(zone_id="gate_north", timestamp=10.0, bottleneck_score=0.85),
    )

    # Stage is safe
    z_stage = ZoneAnalyticsSummary(
        zone_id="main_stage",
        timestamp=10.0,
        density=DensityResult(zone_id="main_stage", timestamp=10.0, avg_density=0.8, max_density=1.0),
        occupancy=OccupancyResult(zone_id="main_stage", timestamp=10.0, capacity=1000, current_count=150, occupancy_ratio=0.15, status="safe"),
        movement=MovementResult(zone_id="main_stage", timestamp=10.0, avg_speed=1.2),
        flow=FlowResult(zone_id="main_stage", timestamp=10.0, flow_rate=3.0),
        bottleneck=BottleneckResult(zone_id="main_stage", timestamp=10.0, bottleneck_score=0.05),
    )

    scene_summary = SceneAnalyticsSummary(
        timestamp=10.0,
        frame_number=300,
        total_active_tracks=350,
        total_validated_tracks=340,
        density=dens_overall,
        occupancy=occ_overall,
        movement=mov_overall,
        flow=flow_overall,
        bottleneck=bn_overall,
        zone_summaries={
            "gate_north": z_north,
            "main_stage": z_stage,
        },
    )

    assessments = engine.assess_scene(scene_summary, zones=zones_dict)
    assert "scene_overall" in assessments
    assert "gate_north" in assessments
    assert "main_stage" in assessments

    # North gate should be CRITICAL
    assert assessments["gate_north"].risk_level == RiskLevel.CRITICAL
    assert assessments["gate_north"].risk_score >= 80.0
    # Main stage should be SAFE
    assert assessments["main_stage"].risk_level == RiskLevel.SAFE
    assert assessments["main_stage"].risk_score < 30.0

    print(f"  [PASS] Multi-zone breakdown: Gate North={assessments['gate_north'].risk_level.value.upper()} ({assessments['gate_north'].risk_score:.1f}), Main Stage={assessments['main_stage'].risk_level.value.upper()} ({assessments['main_stage'].risk_score:.1f})")


def test_7_micro_benchmark_performance():
    print("\n--- Test 7: Micro-Benchmark Latency & Memory Footprint ---")
    tracemalloc.start()
    engine = RiskEngine()

    dens = DensityResult(zone_id="bench_z", timestamp=1.0, avg_density=3.2, max_density=4.1)
    occ = OccupancyResult(zone_id="bench_z", timestamp=1.0, capacity=100, current_count=80, occupancy_ratio=0.80, status="warning")
    mov = MovementResult(zone_id="bench_z", timestamp=1.0, avg_speed=0.6)
    flow = FlowResult(zone_id="bench_z", timestamp=1.0, flow_rate=8.0)
    bn = BottleneckResult(zone_id="bench_z", timestamp=1.0, bottleneck_score=0.65)

    # Warm-up
    for _ in range(10):
        engine.assess("bench_z", 1.0, density_result=dens, occupancy_result=occ, movement_result=mov, flow_result=flow, bottleneck_result=bn)

    # 1,000 Iterations benchmark
    num_runs = 1000
    t0 = time.perf_counter()
    for i in range(num_runs):
        engine.assess(
            "bench_z",
            float(i),
            density_result=dens,
            occupancy_result=occ,
            movement_result=mov,
            flow_result=flow,
            bottleneck_result=bn,
        )
    total_time_ms = (time.perf_counter() - t0) * 1000.0
    avg_latency_ms = total_time_ms / float(num_runs)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    print(f"  Latency: {avg_latency_ms:.3f} ms / assessment ({num_runs} assessments in {total_time_ms:.1f} ms)")
    print(f"  Memory: Current={current / 1024:.1f} KB, Peak={peak_mb:.2f} MB")

    assert avg_latency_ms < 25.0, f"Latency target < 25 ms, got {avg_latency_ms:.3f} ms"
    assert peak_mb < 5.0, f"Memory footprint target < 5 MB, got {peak_mb:.2f} MB"
    print(f"  [PASS] Ultra-fast performance confirmed: {avg_latency_ms:.3f} ms << 25 ms target, {peak_mb:.2f} MB << 5 MB target")


def main():
    print("=" * 65)
    print("SENTINEL RISK ENGINE - COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 65)
    
    t_start = time.perf_counter()
    test_1_configurable_weights_and_normalization()
    test_2_multifactor_scoring_and_tiers()
    test_3_temporal_smoothing_and_hysteresis()
    test_4_predictive_early_warning_integration()
    test_5_explainability_and_top_drivers()
    test_6_per_zone_and_scene_aggregate()
    test_7_micro_benchmark_performance()
    
    elapsed = (time.perf_counter() - t_start) * 1000.0
    print("\n" + "=" * 65)
    print(f"ALL 7 VERIFICATION SUITES PASSED in {elapsed:.2f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
