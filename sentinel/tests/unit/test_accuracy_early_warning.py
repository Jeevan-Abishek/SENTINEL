"""
Unit tests for Accuracy-Gated 5+ Minute Predictive Early Warning & Alert Engine.
Ensures zero false alarms, strict confidence gating, and reliable multi-window trend projections.
"""

import pytest
import numpy as np

from app.alerts.early_warning import (
    EarlyWarningEngine,
    EarlyWarningConfig,
    EarlyWarningAccuracyStatus,
    EarlyWarningSeverity,
    EarlyWarningProjection,
)
from app.alerts.manager import (
    EarlyWarningAlert,
    EarlyWarningAlertManager,
    AlertManagerWrapper,
)


class TestEarlyWarningAccuracyGating:
    """Verifies that speculative alerts are withheld when data is noisy or incomplete."""

    def test_gating_on_insufficient_samples(self):
        engine = EarlyWarningEngine(EarlyWarningConfig(min_history_samples=3))
        # 1 sample
        p1 = engine.assess_early_warning("zone_1", 100.0, max_density=3.5, inflow_rate=20.0, outflow_rate=5.0)
        assert p1.accuracy_status == EarlyWarningAccuracyStatus.INSUFFICIENT_DATA
        assert p1.is_early_warning is False

        # 2 samples
        p2 = engine.assess_early_warning("zone_1", 105.0, max_density=3.6, inflow_rate=20.0, outflow_rate=5.0)
        assert p2.accuracy_status == EarlyWarningAccuracyStatus.INSUFFICIENT_DATA
        assert p2.is_early_warning is False

    def test_gating_on_insufficient_duration(self):
        engine = EarlyWarningEngine(EarlyWarningConfig(min_history_samples=3, min_window_duration_seconds=10.0))
        # 3 samples, but all within 2 seconds
        engine.assess_early_warning("zone_2", 100.0, max_density=3.0)
        engine.assess_early_warning("zone_2", 101.0, max_density=3.2)
        p = engine.assess_early_warning("zone_2", 102.0, max_density=3.4)

        assert p.accuracy_status == EarlyWarningAccuracyStatus.INSUFFICIENT_DATA
        assert p.is_early_warning is False
        assert "below minimum" in p.reason

    def test_gating_on_low_tracking_confidence(self):
        engine = EarlyWarningEngine(EarlyWarningConfig(min_history_samples=3, min_window_duration_seconds=3.0, min_tracking_confidence=0.50))
        # 3 samples, 6 seconds, but very low tracking confidence
        engine.assess_early_warning("zone_3", 100.0, max_density=3.5, avg_track_confidence=0.30)
        engine.assess_early_warning("zone_3", 103.0, max_density=3.8, avg_track_confidence=0.25)
        p = engine.assess_early_warning("zone_3", 106.0, max_density=4.0, avg_track_confidence=0.35)

        assert p.accuracy_status == EarlyWarningAccuracyStatus.INSUFFICIENT_DATA
        assert p.is_early_warning is False
        assert "Tracking quality" in p.reason

    def test_transient_spike_suppressed(self):
        engine = EarlyWarningEngine(EarlyWarningConfig(min_history_samples=3, min_window_duration_seconds=3.0))
        # Baseline
        engine.assess_early_warning("zone_4", 100.0, max_density=1.0, inflow_rate=5.0, outflow_rate=5.0)
        engine.assess_early_warning("zone_4", 103.0, max_density=1.0, inflow_rate=5.0, outflow_rate=5.0)
        # Spike
        engine.assess_early_warning("zone_4", 106.0, max_density=3.5, inflow_rate=5.0, outflow_rate=5.0)
        # Return to normal
        p = engine.assess_early_warning("zone_4", 109.0, max_density=1.1, inflow_rate=5.0, outflow_rate=5.0)

        assert p.is_early_warning is False


class TestEarlyWarningSustainedTrends:
    """Verifies that sustained multi-window surges trigger accurate 5+ min early warnings."""

    def test_sustained_inflow_and_density_surge(self):
        engine = EarlyWarningEngine(EarlyWarningConfig(min_history_samples=3, min_window_duration_seconds=3.0))
        # Sustained rise across 3 windows
        engine.assess_early_warning("gate_a", 100.0, max_density=1.5, inflow_rate=25.0, outflow_rate=10.0)
        engine.assess_early_warning("gate_a", 110.0, max_density=1.7, inflow_rate=28.0, outflow_rate=10.0)
        p = engine.assess_early_warning("gate_a", 120.0, max_density=1.9, inflow_rate=30.0, outflow_rate=10.0)

        assert p.is_early_warning is True
        assert p.accuracy_status in [EarlyWarningAccuracyStatus.HIGH_CONFIDENCE, EarlyWarningAccuracyStatus.MODERATE_CONFIDENCE]
        assert p.confidence_score >= 0.70
        assert p.projected_density_5min >= 2.5
        assert p.net_inflow_surplus >= 15.0
        assert p.severity in [EarlyWarningSeverity.WARNING, EarlyWarningSeverity.CRITICAL]

    def test_bottleneck_persistence_trigger(self):
        engine = EarlyWarningEngine(EarlyWarningConfig(min_history_samples=3, min_window_duration_seconds=3.0))
        # Choke persistence with decelerating speed
        engine.assess_early_warning("turnstile", 100.0, max_density=2.0, bottleneck_score=0.8, mean_speed=0.9)
        engine.assess_early_warning("turnstile", 110.0, max_density=2.2, bottleneck_score=0.85, mean_speed=0.6)
        p = engine.assess_early_warning("turnstile", 120.0, max_density=2.4, bottleneck_score=0.9, mean_speed=0.4)

        assert p.is_early_warning is True
        assert p.bottleneck_persistence >= 0.60
        assert p.speed_trend_slope < 0.0

    def test_lead_time_estimation_bounds(self):
        engine = EarlyWarningEngine(EarlyWarningConfig(min_history_samples=3, density_warning_threshold=2.5))
        # Steady rise of 0.20 p/m²/min starting at 1.50
        engine.assess_early_warning("zone_x", 0.0, max_density=1.50)
        engine.assess_early_warning("zone_x", 30.0, max_density=1.60)
        p = engine.assess_early_warning("zone_x", 60.0, max_density=1.70)

        # Expected lead time to 2.5 p/m² is (2.5 - 1.70) / 0.20 = 4.0 min
        assert 3.5 <= p.estimated_lead_minutes <= 4.5


class TestAlertManagerLifecycle:
    """Verifies deduplication, cooldown, acknowledge, and resolve workflows."""

    def test_deduplication_and_cooldown(self):
        mgr = EarlyWarningAlertManager(EarlyWarningConfig(cooldown_seconds=45.0, min_history_samples=3))
        # Create alert
        mgr.process_telemetry("zone_alpha", 100.0, max_density=1.5, inflow_rate=25.0, outflow_rate=5.0)
        mgr.process_telemetry("zone_alpha", 110.0, max_density=1.8, inflow_rate=25.0, outflow_rate=5.0)
        _, alert1 = mgr.process_telemetry("zone_alpha", 120.0, max_density=2.1, inflow_rate=25.0, outflow_rate=5.0)

        assert alert1 is not None
        assert alert1.status == "ACTIVE"
        assert len(mgr.get_active_alerts()) == 1

        # Follow-up at t=125s (within 45s cooldown) -> should update, not create second alert!
        _, alert2 = mgr.process_telemetry("zone_alpha", 125.0, max_density=2.2, inflow_rate=25.0, outflow_rate=5.0)
        assert alert2.id == alert1.id
        assert len(mgr.get_active_alerts()) == 1

    def test_acknowledgment_and_resolution(self):
        mgr = EarlyWarningAlertManager(EarlyWarningConfig(min_history_samples=3))
        mgr.process_telemetry("zone_beta", 100.0, max_density=1.5, inflow_rate=25.0, outflow_rate=5.0)
        mgr.process_telemetry("zone_beta", 110.0, max_density=1.8, inflow_rate=25.0, outflow_rate=5.0)
        _, alert = mgr.process_telemetry("zone_beta", 120.0, max_density=2.1, inflow_rate=25.0, outflow_rate=5.0)

        assert alert is not None
        aid = alert.id

        # Acknowledge
        ack = mgr.acknowledge_alert(aid, operator_id="op_jane", notes="Dispatched steward 2")
        assert ack is True
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "op_jane"
        assert alert.status == "ACKNOWLEDGED"

        # Resolve
        res = mgr.resolve_alert(aid, operator_id="op_jane", notes="Congestion cleared")
        assert res is True
        assert alert.resolved is True
        assert alert.status == "RESOLVED"
        assert len(mgr.get_active_alerts()) == 0

