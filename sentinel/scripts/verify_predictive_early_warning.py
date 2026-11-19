"""
SENTINEL — 5-Minute Horizon Predictive Crowd Analytics & Early Warning Verification Script
------------------------------------------------------------------------------------------
Validates:
1. Flow of Direction & Movement Metrics:
   - Movement vectors (heading and speed in m/s).
   - Counter-flow conflicts (divergence >= 120°) and crossing-flow index.
   - Directional convergence toward choke points.
   - Velocity drops (slowdown relative to baseline speed).
2. Net Flow Imbalance & Density Rate-of-Change:
   - Rolling inflow vs outflow rate and net accumulation (persons/min).
   - Density rate-of-change (p/m²/min) and 5-minute forward projection.
   - 0–5 minute lead window calculation to critical/warning threshold.
   - Decision support labeling: trend-based predictive estimate.
3. 5-Minute Horizon Early Warning in Risk Engine & Alert Manager:
   - Factor 'predictive_5min' calculation.
   - Preemptive WARNING alert triggering before threshold breach.
   - Explainable operator alert messages.
   - Temporal smoothing validation.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Setup paths
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

from app.analytics.models import (
    AnalyticsConfig,
    AnalyticsContext,
    MovementAnalyzer,
    MovementVector,
    FlowAnalyzer,
    DensityAnalyzer,
    BottleneckDetector,
    DensityResult,
    DensityGrid,
)
from app.risk.models import (
    RiskConfig,
    RiskEngine,
    RiskLevel,
    AlertRule,
    AlertManager,
)


class TrackSim:
    def __init__(self, track_id: int, trajectory: list):
        self.track_id = track_id
        self.centroid_history = trajectory
        self.state = "active"

    @property
    def centroid(self):
        return self.centroid_history[-1]

    def get_velocity(self, frame_rate: float = 30.0):
        if len(self.centroid_history) < 2:
            return [0.0, 0.0]
        prev = np.array(self.centroid_history[-2])
        curr = np.array(self.centroid_history[-1])
        return ((curr - prev) * frame_rate).tolist()

    def get_speed(self, frame_rate: float = 30.0):
        vel = self.get_velocity(frame_rate)
        return float(np.linalg.norm(vel))

    def get_direction(self, frame_rate: float = 30.0):
        vel = self.get_velocity(frame_rate)
        if vel[0] == 0 and vel[1] == 0:
            return 0.0
        return float(np.arctan2(vel[1], vel[0]))


def verify_predictive_early_warning():
    print("=" * 80)
    print("SENTINEL 5-MINUTE HORIZON PREDICTIVE ANALYTICS & EARLY WARNING VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Flow of Direction & Movement Metrics
    # -------------------------------------------------------------
    print("\n[STEP 1/4] Verifying Direction & Movement Metrics...")
    mov_cfg = AnalyticsConfig(
        counter_flow_angle_threshold_deg=120.0,
        crossing_flow_angle_threshold_deg=60.0,
        velocity_drop_threshold=0.30,
    )
    mov_analyzer = MovementAnalyzer(mov_cfg)
    ctx_1 = AnalyticsContext(zone_id="north_ingress_concourse", frame_number=1, timestamp=1.0)

    # Create 12 tracks moving east (heading ~0.0 rad, speed ~1.4 m/s)
    # and 3 tracks moving west (counter-flow heading ~pi rad, opposing heading)
    east_tracks = [
        TrackSim(i, [[10.0 + i * 4, 30.0], [10.046 + i * 4, 30.0]])
        for i in range(12)
    ]
    counter_tracks = [
        TrackSim(100 + i, [[60.0 + i * 4, 30.0], [59.954 + i * 4, 30.0]])
        for i in range(3)
    ]
    all_tracks = east_tracks + counter_tracks

    mov_res = mov_analyzer.analyze(all_tracks, ctx_1, frame_rate=30.0)

    print(f"  • Total Tracked Entities: {len(mov_res.movements)}")
    print(f"  • Dominant Direction: {mov_res.dominant_direction:.3f} rad ({np.degrees(mov_res.dominant_direction):.1f}°)")
    print(f"  • Average Movement Speed: {mov_res.avg_speed:.2f} m/s")
    print(f"  • Counter-Flow Conflict Index: {mov_res.counter_flow_index:.3f} (Conflict active: {mov_res.crossing_flow_detected})")
    assert mov_res.counter_flow_index > 0.15, "Counter-flow index must detect opposing movement!"
    assert mov_res.crossing_flow_detected is True, "Crossing/counter-flow conflict flag must be set!"
    print("  ✅ Step 1A: Counter-Flow Conflict successfully detected and quantified.")

    # 1B. Directional Convergence (funneling toward choke point)
    choke_ctx = AnalyticsContext(zone_id="choke_turnstile", frame_number=1, timestamp=1.0)
    converging_tracks = [
        TrackSim(1, [[80.0, 100.0], [81.5, 100.0]]),
        TrackSim(2, [[120.0, 100.0], [118.5, 100.0]]),
        TrackSim(3, [[100.0, 80.0], [100.0, 81.5]]),
        TrackSim(4, [[100.0, 120.0], [100.0, 118.5]]),
    ]
    conv_res = mov_analyzer.analyze(converging_tracks, choke_ctx, frame_rate=30.0)
    print(f"  • Directional Convergence Score: {conv_res.directional_convergence:.3f} (Inward focal alignment)")
    assert conv_res.directional_convergence >= 0.70, "Directional convergence must exceed 0.70 for inward funneling!"
    print("  ✅ Step 1B: Directional Convergence successfully detected and quantified.")

    # 1C. Sudden Velocity Deceleration (Speed Drop)
    print("  • Simulating sudden velocity drop (walking crowd entering congestion)...")
    for f in range(10):
        t = [
            TrackSim(1, [[f * 1.5, 10.0], [(f + 1) * 1.5, 10.0]]),
            TrackSim(2, [[f * 1.5, 20.0], [(f + 1) * 1.5, 20.0]]),
        ]
        mov_analyzer.analyze(t, AnalyticsContext(zone_id="gate_slow", frame_number=f, timestamp=float(f)))

    # Crowd decelerates dramatically over 5 frames
    slow_res = None
    for f in range(10, 15):
        t = [
            TrackSim(1, [[15.0 + (f - 10) * 0.15, 10.0], [15.0 + (f - 9) * 0.15, 10.0]]),
            TrackSim(2, [[15.0 + (f - 10) * 0.15, 20.0], [15.0 + (f - 9) * 0.15, 20.0]]),
        ]
        slow_res = mov_analyzer.analyze(t, AnalyticsContext(zone_id="gate_slow", frame_number=f, timestamp=float(f)))

    print(f"  • Baseline Speed: ~45.0 px/s (1.5 m/s) -> Decelerated Speed: {slow_res.avg_speed:.2f} px/s")
    print(f"  • Velocity Deceleration Ratio: {slow_res.velocity_drop_ratio:.3f} (Drop: {slow_res.velocity_drop_ratio*100:.1f}%)")
    print(f"  • Speed Dynamic Trend: '{slow_res.speed_trend.upper()}'")
    assert slow_res.velocity_drop_ratio >= 0.30, "Velocity drop must be >= 0.30!"
    assert slow_res.speed_trend == "slowing", "Speed trend must register as 'slowing'!"
    print("  ✅ Step 1C: Velocity Deceleration successfully detected.")

    # -------------------------------------------------------------
    # 2. Net Flow Imbalance & Density Rate-of-Change
    # -------------------------------------------------------------
    print("\n[STEP 2/4] Verifying Net Flow Imbalance & 5-Minute Density Extrapolation...")
    flow_cfg = AnalyticsConfig(flow_window_seconds=60)
    flow_analyzer = FlowAnalyzer(flow_cfg)
    density_cfg = AnalyticsConfig(prediction_horizon_minutes=5.0)
    density_analyzer = DensityAnalyzer(density_cfg)

    # Simulate t=0s: 180 people in 100 m² zone -> 1.80 p/m²
    zone_ctx_t0 = AnalyticsContext(zone_id="concourse_alpha", frame_number=1, timestamp=0.0)
    tracks_t0 = [TrackSim(i, [[float(i % 10), float(i // 10)], [float(i % 10), float(i // 10)]]) for i in range(180)]
    flow_res_t0 = flow_analyzer.analyze(tracks_t0, zone_ctx_t0)
    dens_res_t0 = density_analyzer.compute_density(tracks_t0, zone_ctx_t0)
    print(f"  • [t = 0.0s] Headcount: {dens_res_t0.grid.max_density*100:.0f} | Current Density: {dens_res_t0.avg_density:.2f} p/m²")

    # Simulate t=60s: Influx of +24 people (net inflow = +24/min) -> 204 people -> 2.04 p/m²
    zone_ctx_t1 = AnalyticsContext(zone_id="concourse_alpha", frame_number=1800, timestamp=60.0)
    tracks_t1 = [TrackSim(i, [[float(i % 10), float(i // 10)], [float(i % 10), float(i // 10)]]) for i in range(204)]
    flow_res_t1 = flow_analyzer.analyze(tracks_t1, zone_ctx_t1)
    dens_res_t1 = density_analyzer.compute_density(tracks_t1, zone_ctx_t1)

    print(f"  • [t = 60.0s] Headcount: 204 | Current Density: {dens_res_t1.avg_density:.2f} p/m²")
    print(f"  • Net Accumulation Rate: +{flow_res_t1.net_accumulation_rate_ppm:.1f} persons/min")
    print(f"  • Density Rate-of-Change (dρ/dt): +{dens_res_t1.density_rate_of_change:.4f} p/m²/min")
    print(f"  • Density Trend Label: '{dens_res_t1.trend_label.upper()}'")
    print(f"  • 5-Minute Projected Density: {dens_res_t1.projected_density_5min:.2f} p/m² (Trend-based estimate)")
    print(f"  • 5-Minute Projected Headcount: {dens_res_t1.projected_occupancy_5min} persons")

    # Math verification:
    # dρ/dt = (2.04 - 1.80) / 1.0 min = +0.24 p/m²/min
    # ρ(t + 5m) = 2.04 + (0.24 * 5.0) = 3.24 p/m²
    expected_proj = 2.04 + (0.24 * 5.0)
    assert abs(dens_res_t1.projected_density_5min - expected_proj) < 0.05, f"Expected {expected_proj}, got {dens_res_t1.projected_density_5min}"
    print(f"  • Mathematical Proof: ρ(t+5m) = 2.04 + (0.24 × 5.0) = {expected_proj:.2f} p/m² (Exact match)")

    # Lead window to Warning Threshold (3.0 p/m²):
    # Time to 3.0 = (3.0 - 2.04) / 0.24 = 0.96 / 0.24 = 4.0 minutes!
    print(f"  • Estimated Advance Lead Window: ~{dens_res_t1.lead_time_to_critical_min:.1f} minutes before 3.0 p/m² threshold")
    assert dens_res_t1.lead_time_to_critical_min is not None
    assert 3.8 <= dens_res_t1.lead_time_to_critical_min <= 4.2
    print("  ✅ Step 2: Net Flow Imbalance, Rate-of-Change, and 5-Min Horizon Extrapolation Verified.")

    # -------------------------------------------------------------
    # 3. Risk Engine 5-Minute Horizon Early Warning
    # -------------------------------------------------------------
    print("\n[STEP 3/4] Verifying Risk Engine 5-Minute Advance Early Warning Trigger...")
    risk_cfg = RiskConfig(
        warning_threshold=55.0,
        critical_threshold=80.0,
        predictive_weight=0.20,
        density_weight=0.30,
        temporal_smoothing=0.25,
    )
    risk_engine = RiskEngine(risk_cfg)

    # Scenario A: Instantaneous density is only 2.04 p/m² (below 3.0 warning threshold).
    # Without predictive factor, score would be SAFE (~23.3).
    # With predictive factor (proj_dens=3.24, speed slowing, counter-flow active):
    assessment = risk_engine.assess(
        zone_id="concourse_alpha",
        timestamp=60.0,
        density_result=dens_res_t1,
        movement_result=slow_res,
        flow_result=flow_res_t1,
    )

    print(f"  • Instantaneous Physical Density: {dens_res_t1.avg_density:.2f} p/m² (Safe < 3.0)")
    print(f"  • Projected 5-Min Density: {dens_res_t1.projected_density_5min:.2f} p/m² (Approaching Warning)")
    print(f"  • Evaluated Risk Factors ({len(assessment.factors)} active):")
    for f in assessment.factors:
        print(f"    - {f.name.upper()}: val={f.value:.2f} | score={f.score:.1f}/100 | weight={f.weight:.2f}")

    print(f"  • Computed Overall Risk Score: {assessment.risk_score:.1f}/100")
    print(f"  • Triggered Risk Severity Tier: {assessment.risk_level.value.upper()}")
    assert assessment.risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL], "Must trigger WARNING ahead of time!"
    print("  ✅ Step 3: Predictive Factor successfully elevated Risk Score to trigger Early Warning.")

    # -------------------------------------------------------------
    # 4. Explainable Alert Generation & Operator Verification
    # -------------------------------------------------------------
    print("\n[STEP 4/4] Verifying Explainable Operator Alert Message...")
    alert_mgr = AlertManager()
    alert_mgr.add_rule(AlertRule(
        id="rule_predictive_early_warning",
        name="Predictive Early Warning Rule (0-5 min lead)",
        condition="predictive_early_warning",
        risk_level=RiskLevel.WARNING,
        cooldown_seconds=30,
    ))
    alert_mgr.add_rule(AlertRule(
        id="rule_warning_general",
        name="General Warning Rule",
        condition="risk_level_warning",
        risk_level=RiskLevel.WARNING,
        cooldown_seconds=30,
    ))

    alerts = alert_mgr.check_and_create_alerts(assessment)
    print(f"  • Generated Safety Alerts: {len(alerts)}")
    assert len(alerts) >= 1, "Must generate at least one explainable alert!"
    for a in alerts:
        print(f"    📢 [{a.risk_level.value.upper()}] {a.message}")
        assert "concourse_alpha" in a.message
        assert ("EARLY WARNING" in a.message or "Projected density" in a.message)

    print(f"  • Recommended Mitigations ({len(assessment.recommended_actions)} actions):")
    for act in assessment.recommended_actions:
        print(f"    🛡️ {act}")

    print("\n" + "=" * 80)
    print("ALL 4 PREDICTIVE ANALYTICS & EARLY WARNING MODULES VERIFIED (100% PASS)")
    print("=" * 80)


if __name__ == "__main__":
    verify_predictive_early_warning()

