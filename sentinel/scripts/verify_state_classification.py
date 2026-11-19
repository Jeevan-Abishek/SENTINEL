"""
Comprehensive Verification Script for Safe / Warning / Critical State Classification.

Tests:
1. Canonical risk level threshold mapping (SAFE < 60, WARNING 60-79.9, CRITICAL >= 80).
2. Hysteresis buffering (prevents oscillating at boundary values like 59.5 or 78.5).
3. Multi-frame consecutive de-escalation confirmation gating (anti-flicker).
4. Preemptive early warning predictive override (SAFE -> WARNING when early warning active).
5. State presentation metadata and HTML badge generation.
6. Zero-lag state filtering and summary counts.
7. End-to-end RiskEngine state transitions and reset.
8. Performance benchmark (< 1 ms latency, negligible memory).
"""

import sys
import os
import time

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from sentinel.app.risk.models import RiskLevel, RiskConfig
from sentinel.app.risk.rules import (
    STATE_PRESENTATION_METADATA,
    get_state_presentation,
    format_state_pill_html,
    classify_risk_state,
    filter_items_by_state,
    summarize_state_counts,
)
from sentinel.app.risk.engine import RiskEngine
from sentinel.app.analytics.models import (
    DensityResult,
    OccupancyResult,
    MovementResult,
    FlowResult,
    BottleneckResult,
)


def test_canonical_mapping():
    print("[TEST 1] Canonical Risk Level Mapping...")
    # SAFE: < 60.0
    state = classify_risk_state(risk_score=45.0, previous_state=RiskLevel.SAFE)
    assert state == RiskLevel.SAFE, f"Expected SAFE, got {state}"

    # WARNING: 60.0 - 79.9
    state = classify_risk_state(risk_score=65.0, previous_state=RiskLevel.SAFE)
    assert state == RiskLevel.WARNING, f"Expected WARNING, got {state}"

    # CRITICAL: >= 80.0
    state = classify_risk_state(risk_score=85.0, previous_state=RiskLevel.SAFE)
    assert state == RiskLevel.CRITICAL, f"Expected CRITICAL, got {state}"
    print("  ✓ Canonical mapping passed!")


def test_hysteresis_buffering():
    print("[TEST 2] Hysteresis Buffering (buffer = 3.0)...")
    # Escalation is immediate
    state = classify_risk_state(
        risk_score=80.5,
        previous_state=RiskLevel.WARNING,
        critical_threshold=80.0,
        hysteresis_buffer=3.0,
    )
    assert state == RiskLevel.CRITICAL, f"Expected immediate escalation to CRITICAL, got {state}"

    # De-escalation buffer test: dropping from CRITICAL (80.0) to 78.0
    # Threshold for leaving CRITICAL is 80.0 - 3.0 = 77.0.
    # At 78.0, it is still within the hysteresis buffer, so it must stay CRITICAL!
    state = classify_risk_state(
        risk_score=78.0,
        previous_state=RiskLevel.CRITICAL,
        critical_threshold=80.0,
        hysteresis_buffer=3.0,
        consecutive_confirmations=1,
        required_deescalation_frames=2,
    )
    assert state == RiskLevel.CRITICAL, f"Expected CRITICAL due to hysteresis buffer, got {state}"

    # Similar for WARNING -> SAFE: warning_threshold = 60.0, buffer = 3.0 -> boundary is 57.0.
    # At 58.5, it must stay WARNING!
    state = classify_risk_state(
        risk_score=58.5,
        previous_state=RiskLevel.WARNING,
        warning_threshold=60.0,
        hysteresis_buffer=3.0,
        consecutive_confirmations=1,
        required_deescalation_frames=2,
    )
    assert state == RiskLevel.WARNING, f"Expected WARNING due to hysteresis buffer, got {state}"
    print("  ✓ Hysteresis buffering passed!")


def test_consecutive_deescalation_confirmation():
    print("[TEST 3] Consecutive De-escalation Confirmation Gating...")
    # Score drops below hysteresis boundary (e.g., 75.0 < 77.0), but required_deescalation_frames = 2.
    # Frame 1: consecutive_confirmations = 1 (< 2) -> stays CRITICAL
    state1 = classify_risk_state(
        risk_score=75.0,
        previous_state=RiskLevel.CRITICAL,
        critical_threshold=80.0,
        hysteresis_buffer=3.0,
        consecutive_confirmations=1,
        required_deescalation_frames=2,
    )
    assert state1 == RiskLevel.CRITICAL, f"Frame 1: Expected CRITICAL holding, got {state1}"

    # Frame 2: consecutive_confirmations = 2 (>= 2) -> now de-escalates to WARNING!
    state2 = classify_risk_state(
        risk_score=75.0,
        previous_state=RiskLevel.CRITICAL,
        critical_threshold=80.0,
        hysteresis_buffer=3.0,
        consecutive_confirmations=2,
        required_deescalation_frames=2,
    )
    assert state2 == RiskLevel.WARNING, f"Frame 2: Expected de-escalation to WARNING, got {state2}"

    # Similarly from WARNING to SAFE with consecutive confirmations
    # Frame 1: score 55.0 (< 57.0), confirmations = 1 (< 2) -> stays WARNING
    state_w1 = classify_risk_state(
        risk_score=55.0,
        previous_state=RiskLevel.WARNING,
        warning_threshold=60.0,
        hysteresis_buffer=3.0,
        consecutive_confirmations=1,
        required_deescalation_frames=2,
    )
    assert state_w1 == RiskLevel.WARNING, f"Expected WARNING holding, got {state_w1}"

    # Frame 2: score 55.0, confirmations = 2 -> de-escalates to SAFE
    state_w2 = classify_risk_state(
        risk_score=55.0,
        previous_state=RiskLevel.WARNING,
        warning_threshold=60.0,
        hysteresis_buffer=3.0,
        consecutive_confirmations=2,
        required_deescalation_frames=2,
    )
    assert state_w2 == RiskLevel.SAFE, f"Expected SAFE after 2 confirmations, got {state_w2}"
    print("  ✓ Consecutive de-escalation confirmation passed!")


def test_early_warning_predictive_override():
    print("[TEST 4] Predictive Early Warning Override...")
    # A low risk score of 45 normally yields SAFE, but with predictive early warning -> WARNING
    state = classify_risk_state(
        risk_score=45.0,
        previous_state=RiskLevel.SAFE,
        is_early_warning=True,
    )
    assert state == RiskLevel.WARNING, f"Expected WARNING via early warning override, got {state}"

    # But if risk score is already CRITICAL (85.0), early warning does not demote it
    state_crit = classify_risk_state(
        risk_score=85.0,
        previous_state=RiskLevel.SAFE,
        is_early_warning=True,
    )
    assert state_crit == RiskLevel.CRITICAL, f"Expected CRITICAL to take precedence, got {state_crit}"
    print("  ✓ Early warning override passed!")


def test_presentation_metadata_and_html():
    print("[TEST 5] State Presentation Metadata and HTML Pills...")
    for level, expected_color in [
        (RiskLevel.SAFE, "#2E7D32"),
        (RiskLevel.WARNING, "#E65100"),
        (RiskLevel.CRITICAL, "#C62828"),
    ]:
        meta = get_state_presentation(level)
        assert meta["color"] == expected_color, f"Mismatch color for {level}: {meta['color']}"
        pill = format_state_pill_html(level)
        assert expected_color in pill, f"Color {expected_color} missing in HTML pill: {pill}"
        assert meta["label"] in pill, f"Label {meta['label']} missing in HTML pill: {pill}"

    # Check string input handling
    meta_str = get_state_presentation("critical")
    assert meta_str["color"] == "#C62828"
    print("  ✓ Presentation metadata and HTML pills passed!")


def test_filtering_and_summaries():
    print("[TEST 6] Filtering and Summaries...")
    items = [
        {"id": "zone_1", "state": "SAFE"},
        {"id": "zone_2", "state": "WARNING"},
        {"id": "zone_3", "state": "CRITICAL"},
        {"id": "zone_4", "state": "SAFE"},
        {"id": "zone_5", "state": "CRITICAL"},
    ]

    all_items = filter_items_by_state(items, "ALL")
    assert len(all_items) == 5, f"Expected 5 items, got {len(all_items)}"

    crit_items = filter_items_by_state(items, "CRITICAL")
    assert len(crit_items) == 2, f"Expected 2 critical items, got {len(crit_items)}"

    warn_items = filter_items_by_state(items, "WARNING")
    assert len(warn_items) == 1, f"Expected 1 warning item, got {len(warn_items)}"

    safe_items = filter_items_by_state(items, "SAFE")
    assert len(safe_items) == 2, f"Expected 2 safe items, got {len(safe_items)}"

    counts = summarize_state_counts(items)
    assert counts["total"] == 5
    assert counts["safe"] == 2
    assert counts["warning"] == 1
    assert counts["critical"] == 2
    print("  ✓ Filtering and summaries passed!")


def test_risk_engine_integration():
    print("[TEST 7] RiskEngine State Lifecycle & Assessment...")
    engine = RiskEngine(config=RiskConfig(temporal_smoothing=0.0, consecutive_deescalation_frames=2))
    assert engine.previous_levels.get("overall", RiskLevel.SAFE) == RiskLevel.SAFE

    t0 = time.time()
    # Assess nominal crowd
    density = DensityResult(
        zone_id="overall",
        timestamp=t0,
        avg_density=0.1,
        max_density=0.1,
        trend_label="stable",
    )
    occupancy = OccupancyResult(
        zone_id="overall",
        timestamp=t0,
        current_count=10,
        capacity=100,
        occupancy_ratio=0.1,
        status="safe",
    )
    movement = MovementResult(
        zone_id="overall",
        timestamp=t0,
        avg_speed=1.2,
        speed_trend="stable",
    )
    flow = FlowResult(
        zone_id="overall",
        timestamp=t0,
        flow_rate=1.0,
        entry_flow=2.0,
        exit_flow=2.0,
        net_flow=0.0,
    )
    bottleneck = BottleneckResult(
        zone_id="overall",
        timestamp=t0,
        bottleneck_score=0.05,
        is_bottleneck=False,
    )

    assessment = engine.assess(
        zone_id="overall",
        timestamp=t0,
        density_result=density,
        occupancy_result=occupancy,
        movement_result=movement,
        flow_result=flow,
        bottleneck_result=bottleneck,
    )
    assert assessment.risk_level == RiskLevel.SAFE, f"Expected SAFE, got {assessment.risk_level}"
    assert engine.previous_levels.get("overall") == RiskLevel.SAFE

    # Trigger critical crowd (all factors elevated)
    t1 = t0 + 1.0
    crit_density = DensityResult(
        zone_id="overall",
        timestamp=t1,
        avg_density=4.5,
        max_density=4.5,
        trend_label="surging",
        risk_score=95.0,
    )
    crit_occupancy = OccupancyResult(
        zone_id="overall",
        timestamp=t1,
        current_count=190,
        capacity=100,
        occupancy_ratio=1.9,
        status="critical",
        risk_score=95.0,
    )
    crit_movement = MovementResult(
        zone_id="overall",
        timestamp=t1,
        avg_speed=0.05,
        speed_trend="slowing",
        risk_score=90.0,
    )
    crit_flow = FlowResult(
        zone_id="overall",
        timestamp=t1,
        flow_rate=18.0,
        entry_flow=45.0,
        exit_flow=2.0,
        net_flow=43.0,
        risk_score=90.0,
    )
    crit_bottleneck = BottleneckResult(
        zone_id="overall",
        timestamp=t1,
        bottleneck_score=0.95,
        is_bottleneck=True,
        risk_score=95.0,
    )

    assessment_crit = engine.assess(
        zone_id="overall",
        timestamp=t1,
        density_result=crit_density,
        occupancy_result=crit_occupancy,
        movement_result=crit_movement,
        flow_result=crit_flow,
        bottleneck_result=crit_bottleneck,
    )
    assert assessment_crit.risk_level == RiskLevel.CRITICAL, f"Expected CRITICAL, got {assessment_crit.risk_level}"
    assert engine.previous_levels.get("overall") == RiskLevel.CRITICAL

    # Reset
    engine.reset_state()
    assert engine.previous_levels.get("overall", RiskLevel.SAFE) == RiskLevel.SAFE
    assert len(engine.deescalation_counters) == 0
    print("  ✓ RiskEngine integration passed!")


def test_performance_microbenchmark():
    print("[TEST 8] Performance Micro-Benchmark (10,000 iterations)...")
    start = time.perf_counter()
    state = RiskLevel.SAFE
    for i in range(10000):
        score = 50.0 + (i % 45)  # Ranges 50 to 95
        state = classify_risk_state(
            risk_score=score,
            previous_state=state,
        )
        _ = get_state_presentation(state)
        _ = format_state_pill_html(state)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print(f"  ✓ 10,000 classifications & HTML generations completed in {elapsed_ms:.2f} ms ({elapsed_ms / 10000.0 * 1000.0:.3f} µs/op)")
    assert elapsed_ms < 500.0, f"Benchmark too slow: {elapsed_ms} ms"


if __name__ == "__main__":
    print("=" * 60)
    print("SENTINEL: Safe / Warning / Critical State Classification Verification")
    print("=" * 60)
    test_canonical_mapping()
    test_hysteresis_buffering()
    test_consecutive_deescalation_confirmation()
    test_early_warning_predictive_override()
    test_presentation_metadata_and_html()
    test_filtering_and_summaries()
    test_risk_engine_integration()
    test_performance_microbenchmark()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! (Zero errors, high throughput)")
    print("=" * 60)
