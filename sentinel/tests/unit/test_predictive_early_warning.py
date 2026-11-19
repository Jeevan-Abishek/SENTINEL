"""
Unit tests for SENTINEL 5-Minute Predictive Trend & Early Warning Capabilities.
Validates:
1. Movement vectors, counter-flow conflicts, directional convergence, and velocity drop detection.
2. Net flow imbalance, density rate-of-change, and 5-minute trend extrapolation.
3. Advance early warning triggering in RiskEngine and explainable alerts in AlertManager.
"""

import pytest
import numpy as np
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


class DummyTrack:
    def __init__(self, track_id: int, centroid_history: list, state: str = "active"):
        self.track_id = track_id
        self.centroid_history = centroid_history
        self.state = state

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


class TestMovementDynamics:
    def test_counter_flow_conflict_detection(self):
        """Verify that counter-flow opposing dominant heading is correctly detected."""
        config = AnalyticsConfig()
        analyzer = MovementAnalyzer(config)
        context = AnalyticsContext(zone_id="corridor_north", frame_number=1, timestamp=1.0)

        # 8 tracks moving eastward (heading 0 rad, speed ~1.5 m/s)
        tracks = []
        for i in range(8):
            t = DummyTrack(
                track_id=i,
                centroid_history=[[10.0 + i * 5, 20.0], [10.05 + i * 5, 20.0]],
            )
            tracks.append(t)

        # 2 tracks moving westward (counter-flow: heading pi rad, opposing flow)
        for i in range(8, 10):
            t = DummyTrack(
                track_id=i,
                centroid_history=[[50.0 + (i - 8) * 5, 20.0], [49.95 + (i - 8) * 5, 20.0]],
            )
            tracks.append(t)

        res = analyzer.analyze(tracks, context, frame_rate=30.0)
        assert res.dominant_direction is not None
        assert res.counter_flow_index > 0.15, f"Expected counter-flow index > 0.15, got {res.counter_flow_index}"
        assert res.crossing_flow_detected is True

    def test_directional_convergence_toward_choke(self):
        """Verify that people moving inward toward a bottleneck exhibits high convergence."""
        config = AnalyticsConfig()
        analyzer = MovementAnalyzer(config)
        context = AnalyticsContext(zone_id="choke_exit", frame_number=1, timestamp=1.0)

        # 4 tracks heading inwards toward (100, 100)
        tracks = [
            DummyTrack(1, [[80.0, 100.0], [82.0, 100.0]]),   # moving right
            DummyTrack(2, [[120.0, 100.0], [118.0, 100.0]]), # moving left
            DummyTrack(3, [[100.0, 80.0], [100.0, 82.0]]),   # moving down
            DummyTrack(4, [[100.0, 120.0], [100.0, 118.0]]), # moving up
        ]

        res = analyzer.analyze(tracks, context, frame_rate=30.0)
        assert res.directional_convergence > 0.70, f"Expected convergence > 0.70, got {res.directional_convergence}"

    def test_sudden_velocity_drop_detection(self):
        """Verify that crowd slowdown relative to baseline triggers speed_trend = 'slowing'."""
        config = AnalyticsConfig()
        analyzer = MovementAnalyzer(config)

        # Feed 10 frames of normal walking speed (~1.5 m/s per frame)
        for f in range(10):
            tracks = [
                DummyTrack(1, [[f * 1.5, 10.0], [(f + 1) * 1.5, 10.0]]),
                DummyTrack(2, [[f * 1.5, 20.0], [(f + 1) * 1.5, 20.0]]),
            ]
            ctx = AnalyticsContext(zone_id="gate_a", frame_number=f, timestamp=float(f))
            analyzer.analyze(tracks, ctx, frame_rate=30.0)

        # Feed 4 frames of sudden slowdown to ~0.1 m/s (congestion beginning)
        slow_res = None
        for f in range(10, 15):
            slow_tracks = [
                DummyTrack(1, [[15.0 + (f - 10) * 0.1, 10.0], [15.0 + (f - 9) * 0.1, 10.0]]),
                DummyTrack(2, [[15.0 + (f - 10) * 0.1, 20.0], [15.0 + (f - 9) * 0.1, 20.0]]),
            ]
            slow_ctx = AnalyticsContext(zone_id="gate_a", frame_number=f, timestamp=float(f))
            slow_res = analyzer.analyze(slow_tracks, slow_ctx, frame_rate=30.0)

        assert slow_res is not None
        assert slow_res.velocity_drop_ratio > 0.30
        assert slow_res.speed_trend == "slowing"


class TestDensityAndFlowPredictions:
    def test_density_rate_of_change_and_5min_horizon(self):
        """Verify that accumulating crowd correctly calculates 5-minute projected density and lead window."""
        config = AnalyticsConfig(prediction_horizon_minutes=5.0)
        analyzer = DensityAnalyzer(config)

        # Frame 1: 10 people in 100 m² zone -> 0.10 p/m² at t=0
        tracks_t0 = [DummyTrack(i, [[float(i), float(i)], [float(i), float(i)]]) for i in range(10)]
        ctx_t0 = AnalyticsContext(zone_id="arena_north", frame_number=1, timestamp=0.0)
        res_t0 = analyzer.compute_density(tracks_t0, ctx_t0)
        assert res_t0.avg_density == 0.10

        # Frame 2: 30 people at t=60s (1 min later) -> 0.30 p/m² (+0.20 p/m²/min)
        tracks_t1 = [DummyTrack(i, [[float(i), float(i)], [float(i), float(i)]]) for i in range(30)]
        ctx_t1 = AnalyticsContext(zone_id="arena_north", frame_number=1800, timestamp=60.0)
        res_t1 = analyzer.compute_density(tracks_t1, ctx_t1)

        # Expected rate of change = +0.20 p/m²/min
        assert pytest.approx(res_t1.density_rate_of_change, rel=1e-2) == 0.20
        assert res_t1.trend_label == "rising"

        # Projected density in 5 minutes: 0.30 + (0.20 * 5) = 1.30 p/m²
        assert pytest.approx(res_t1.projected_density_5min, rel=1e-2) == 1.30
        # Projected occupancy: 30 + (20 * 5) = 130 people
        assert res_t1.projected_occupancy_5min == 130

    def test_lead_time_to_critical_threshold(self):
        """Verify accurate lead time estimation before crossing warning threshold (3.0 p/m²)."""
        config = AnalyticsConfig(prediction_horizon_minutes=5.0)
        analyzer = DensityAnalyzer(config)

        # t=0: 200 people -> 2.0 p/m²
        t0 = [DummyTrack(i, [[float(i), float(i)], [float(i), float(i)]]) for i in range(200)]
        analyzer.compute_density(t0, AnalyticsContext(zone_id="choke_z", frame_number=1, timestamp=0.0))

        # t=60s: 230 people -> 2.3 p/m² (rate = +0.30 p/m²/min)
        t1 = [DummyTrack(i, [[float(i), float(i)], [float(i), float(i)]]) for i in range(230)]
        res = analyzer.compute_density(t1, AnalyticsContext(zone_id="choke_z", frame_number=1800, timestamp=60.0))

        assert res.projected_density_5min >= 3.5  # 2.3 + 1.5 = 3.8
        # Lead time to 3.0 threshold: (3.0 - 2.3) / 0.30 = 2.33 minutes (~2.3 min)
        assert res.lead_time_to_critical_min is not None
        assert 2.0 <= res.lead_time_to_critical_min <= 2.6


class TestRiskEngineEarlyWarning:
    def test_predictive_early_warning_trigger(self):
        """Verify that high projected 5-minute density triggers WARNING before current density breaches threshold."""
        risk_cfg = RiskConfig(
            warning_threshold=50.0,
            predictive_weight=0.25,
            density_weight=0.25,
        )
        engine = RiskEngine(risk_cfg)

        # Current density is 2.1 p/m² (normal/safe < 3.0), but projected 5-min density is 3.8 p/m²
        density = DensityResult(
            zone_id="gate_north",
            timestamp=100.0,
            grid=DensityGrid(grid_data=[[2.1]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=2.1),
            avg_density=2.1,
            max_density=2.1,
            projected_density_5min=3.8,
            density_rate_of_change=0.34,
            lead_time_to_critical_min=2.6,
        )

        assessment = engine.assess("gate_north", 100.0, density_result=density)
        factor_names = [f.name for f in assessment.factors]
        assert "predictive_5min" in factor_names

        pred_f = next(f for f in assessment.factors if f.name == "predictive_5min")
        assert pred_f.value == 3.8
        assert pred_f.score > 55.0

        # Actions should include preemptive early warning advice
        assert any("Early Warning" in act for act in assessment.recommended_actions)

    def test_alert_manager_generates_early_warning_message(self):
        """Verify AlertManager triggers explainable early warning alert with lead time info."""
        alert_mgr = AlertManager()
        alert_mgr.add_rule(AlertRule(
            id="rule_pred_warning",
            name="Predictive Early Warning Rule",
            condition="predictive_early_warning",
            risk_level=RiskLevel.WARNING,
        ))

        risk_cfg = RiskConfig(warning_threshold=50.0)
        engine = RiskEngine(risk_cfg)

        density = DensityResult(
            zone_id="sector_7",
            timestamp=200.0,
            grid=DensityGrid(grid_data=[[2.0]], grid_width=1, grid_height=1, cell_size_meters=1.0, max_density=2.0),
            avg_density=2.0,
            max_density=2.0,
            projected_density_5min=3.6,
        )

        assessment = engine.assess("sector_7", 200.0, density_result=density)
        new_alerts = alert_mgr.check_and_create_alerts(assessment)

        assert len(new_alerts) == 1
        alert = new_alerts[0]
        assert alert.risk_level == RiskLevel.WARNING
        assert "EARLY WARNING" in alert.message
        assert "sector_7" in alert.message
        assert "3.60 p/m²" in alert.message
