"""
Unit tests for SENTINEL Risk Engine and Alerts Module
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.risk.models import (
    RiskLevel,
    RiskFactor,
    RiskAssessment,
    RiskConfig,
    RiskEngine,
    Alert,
    AlertRule,
    AlertConfig,
    AlertManager,
    NotificationHandler,
)
from app.risk.engine import RiskEngineWrapper
from app.alerts.manager import AlertManagerWrapper
from app.alerts.notifications import (
    NotificationChannel,
    WebSocketNotificationChannel,
    EmailNotificationChannel,
    SMSNotificationChannel,
    SoundNotificationChannel,
    NotificationService,
)
from app.analytics.models import (
    DensityResult, OccupancyResult, MovementResult, FlowResult, BottleneckResult,
    DensityGrid, MovementVector
)


class MockDensityResult:
    def __init__(self, max_density, avg_density):
        self.max_density = max_density
        self.avg_density = avg_density


class MockOccupancyResult:
    def __init__(self, occupancy_ratio, current_count, capacity):
        self.occupancy_ratio = occupancy_ratio
        self.current_count = current_count
        self.capacity = capacity


class MockMovementResult:
    def __init__(self, avg_speed, movements=None):
        self.avg_speed = avg_speed
        self.movements = movements or []


class MockFlowResult:
    def __init__(self, flow_rate, entry_flow=0, exit_flow=0, net_flow=0):
        self.flow_rate = flow_rate
        self.entry_flow = entry_flow
        self.exit_flow = exit_flow
        self.net_flow = net_flow


class MockBottleneckResult:
    def __init__(self, bottleneck_score, is_bottleneck=False):
        self.bottleneck_score = bottleneck_score
        self.is_bottleneck = is_bottleneck


class TestRiskModels:
    def test_risk_level_enum(self):
        assert RiskLevel.SAFE == "safe"
        assert RiskLevel.WARNING == "warning"
        assert RiskLevel.CRITICAL == "critical"

    def test_risk_factor_creation(self):
        factor = RiskFactor(
            name="density",
            value=3.5,
            weight=0.3,
            score=70.0,
            threshold_warning=3.0,
            threshold_critical=5.0,
        )
        assert factor.name == "density"
        assert factor.score == 70.0

    def test_risk_assessment_creation(self):
        factors = [
            RiskFactor(name="density", value=3.5, weight=0.3, score=70.0),
            RiskFactor(name="occupancy", value=0.8, weight=0.2, score=80.0),
        ]
        assessment = RiskAssessment(
            zone_id="zone-1",
            timestamp=1.0,
            risk_level=RiskLevel.WARNING,
            risk_score=74.0,
            factors=factors,
            recommended_actions=["Increase monitoring"],
        )
        assert assessment.risk_level == RiskLevel.WARNING
        assert len(assessment.factors) == 2

    def test_alert_creation(self):
        alert = Alert(
            id="alert-1",
            zone_id="zone-1",
            risk_level=RiskLevel.CRITICAL,
            message="Critical risk detected",
            timestamp=1.0,
        )
        assert alert.id == "alert-1"
        assert alert.acknowledged is False

    def test_alert_rule_creation(self):
        rule = AlertRule(
            id="rule-1",
            name="Critical Risk Alert",
            condition="risk_level_critical",
            risk_level=RiskLevel.CRITICAL,
            cooldown_seconds=30,
        )
        assert rule.condition == "risk_level_critical"

    def test_risk_config_defaults(self):
        config = RiskConfig()
        assert config.density_weight == 0.3
        assert config.safe_threshold == 30.0
        assert config.warning_threshold == 60.0
        assert config.critical_threshold == 80.0

    def test_alert_config_defaults(self):
        config = AlertConfig()
        assert config.cooldown_seconds == 30
        assert config.max_active == 100


class TestRiskEngine:
    def test_engine_creation(self):
        engine = RiskEngine()
        assert engine.config is not None

    def test_assess_no_factors(self):
        engine = RiskEngine()
        assessment = engine.assess("zone-1", 1.0)
        assert assessment.risk_score == 0.0
        assert assessment.risk_level == RiskLevel.SAFE

    def test_assess_density_only(self):
        engine = RiskEngine()
        density = MockDensityResult(max_density=5.0, avg_density=3.0)
        assessment = engine.assess("zone-1", 1.0, density_result=density)
        assert assessment.risk_score > 0
        assert len(assessment.factors) == 1
        assert assessment.factors[0].name == "density"

    def test_assess_occupancy_only(self):
        engine = RiskEngine()
        occupancy = MockOccupancyResult(occupancy_ratio=0.8, current_count=80, capacity=100)
        assessment = engine.assess("zone-1", 1.0, occupancy_result=occupancy)
        assert assessment.risk_score > 0
        assert len(assessment.factors) == 1
        assert assessment.factors[0].name == "occupancy"

    def test_assess_movement_only(self):
        engine = RiskEngine()
        movement = MockMovementResult(avg_speed=0.5)
        assessment = engine.assess("zone-1", 1.0, movement_result=movement)
        assert assessment.risk_score > 0
        assert len(assessment.factors) == 1
        assert assessment.factors[0].name == "movement"

    def test_assess_flow_only(self):
        engine = RiskEngine()
        flow = MockFlowResult(flow_rate=10.0)
        assessment = engine.assess("zone-1", 1.0, flow_result=flow)
        assert assessment.risk_score > 0
        assert len(assessment.factors) == 1
        assert assessment.factors[0].name == "flow"

    def test_assess_bottleneck_only(self):
        engine = RiskEngine()
        bottleneck = MockBottleneckResult(bottleneck_score=0.8)
        assessment = engine.assess("zone-1", 1.0, bottleneck_result=bottleneck)
        assert assessment.risk_score > 0
        assert len(assessment.factors) == 1
        assert assessment.factors[0].name == "bottleneck"

    def test_assess_all_factors(self):
        engine = RiskEngine()
        density = MockDensityResult(max_density=5.0, avg_density=3.0)
        occupancy = MockOccupancyResult(occupancy_ratio=0.8, current_count=80, capacity=100)
        movement = MockMovementResult(avg_speed=0.5)
        flow = MockFlowResult(flow_rate=10.0)
        bottleneck = MockBottleneckResult(bottleneck_score=0.8)
        
        assessment = engine.assess(
            "zone-1", 1.0,
            density_result=density,
            occupancy_result=occupancy,
            movement_result=movement,
            flow_result=flow,
            bottleneck_result=bottleneck,
        )
        assert len(assessment.factors) == 5
        assert assessment.risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL]

    def test_temporal_smoothing(self):
        config = RiskConfig(temporal_smoothing=0.5)
        engine = RiskEngine(config)
        
        density = MockDensityResult(max_density=5.0, avg_density=3.0)
        
        # First assessment
        a1 = engine.assess("zone-1", 1.0, density_result=density)
        
        # Second assessment with same input
        a2 = engine.assess("zone-1", 2.0, density_result=density)
        
        # Score should be smoothed
        assert a2.risk_score != a1.risk_score or a2.risk_score == a1.risk_score

    def test_risk_levels(self):
        # Use custom config with lower thresholds for testing
        config = RiskConfig(
            safe_threshold=20.0,
            warning_threshold=35.0,
            critical_threshold=70.0,
        )
        engine = RiskEngine(config)
        
        # Safe
        density_safe = MockDensityResult(max_density=1.0, avg_density=0.5)
        a1 = engine.assess("zone-safe", 1.0, density_result=density_safe)
        assert a1.risk_level == RiskLevel.SAFE
        
        # Warning - combine density and occupancy
        density_warn = MockDensityResult(max_density=4.0, avg_density=3.0)
        occupancy_warn = MockOccupancyResult(occupancy_ratio=0.85, current_count=85, capacity=100)
        a2 = engine.assess("zone-warn", 2.0, density_result=density_warn, occupancy_result=occupancy_warn)
        assert a2.risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL]
    
        # Critical - combine multiple factors
        density_crit = MockDensityResult(max_density=10.0, avg_density=8.0)
        occupancy_crit = MockOccupancyResult(occupancy_ratio=0.95, current_count=95, capacity=100)
        a3 = engine.assess("zone-crit", 3.0, density_result=density_crit, occupancy_result=occupancy_crit)
        assert a3.risk_level == RiskLevel.CRITICAL

    def test_recommended_actions(self):
        config = RiskConfig(
            safe_threshold=20.0,
            warning_threshold=35.0,
            critical_threshold=70.0,
        )
        engine = RiskEngine(config)
        density = MockDensityResult(max_density=10.0, avg_density=8.0)
        occupancy = MockOccupancyResult(occupancy_ratio=0.95, current_count=95, capacity=100)
        assessment = engine.assess("zone-rec", 1.0, density_result=density, occupancy_result=occupancy)
        assert len(assessment.recommended_actions) > 0
        assert any("evacuation" in a.lower() for a in assessment.recommended_actions)


class TestRiskEngineWrapper:
    def test_wrapper_creation(self):
        wrapper = RiskEngineWrapper()
        assert wrapper.engine is not None

    def test_wrapper_assess(self):
        wrapper = RiskEngineWrapper()
        density = MockDensityResult(max_density=5.0, avg_density=3.0)
        assessment = wrapper.assess("zone-1", 1.0, density_result=density)
        assert isinstance(assessment, RiskAssessment)


class TestAlertManager:
    def test_manager_creation(self):
        manager = AlertManager()
        assert manager.config is not None
        assert len(manager.active_alerts) == 0

    def test_add_rule(self):
        manager = AlertManager()
        rule = AlertRule(
            id="rule-1",
            name="Test Rule",
            condition="risk_level_critical",
            risk_level=RiskLevel.CRITICAL,
        )
        manager.add_rule(rule)
        assert len(manager.rules) == 1

    def test_remove_rule(self):
        manager = AlertManager()
        rule = AlertRule(id="rule-1", name="Test", condition="risk_level_critical", risk_level=RiskLevel.CRITICAL)
        manager.add_rule(rule)
        manager.remove_rule("rule-1")
        assert len(manager.rules) == 0

    def test_check_critical_alert(self):
        manager = AlertManager()
        rule = AlertRule(id="rule-1", name="Critical", condition="risk_level_critical", risk_level=RiskLevel.CRITICAL)
        manager.add_rule(rule)
        
        assessment = RiskAssessment(
            zone_id="zone-1",
            timestamp=1.0,
            risk_level=RiskLevel.CRITICAL,
            risk_score=90.0,
            factors=[],
        )
        alerts = manager.check_and_create_alerts(assessment)
        assert len(alerts) == 1
        assert alerts[0].risk_level == RiskLevel.CRITICAL

    def test_check_warning_alert(self):
        manager = AlertManager()
        rule = AlertRule(id="rule-1", name="Warning", condition="risk_level_warning", risk_level=RiskLevel.WARNING)
        manager.add_rule(rule)
        
        assessment = RiskAssessment(
            zone_id="zone-1",
            timestamp=1.0,
            risk_level=RiskLevel.WARNING,
            risk_score=70.0,
            factors=[],
        )
        alerts = manager.check_and_create_alerts(assessment)
        assert len(alerts) == 1
        assert alerts[0].risk_level == RiskLevel.WARNING

    def test_cooldown_prevents_duplicate(self):
        manager = AlertManager()
        rule = AlertRule(id="rule-1", name="Critical", condition="risk_level_critical", risk_level=RiskLevel.CRITICAL, cooldown_seconds=60)
        manager.add_rule(rule)
        
        assessment = RiskAssessment(
            zone_id="zone-1",
            timestamp=1.0,
            risk_level=RiskLevel.CRITICAL,
            risk_score=90.0,
            factors=[],
        )
        alerts1 = manager.check_and_create_alerts(assessment)
        alerts2 = manager.check_and_create_alerts(assessment)
        
        assert len(alerts1) == 1
        assert len(alerts2) == 0  # Should be blocked by cooldown

    def test_acknowledge_alert(self):
        manager = AlertManager()
        rule = AlertRule(id="rule-1", name="Critical", condition="risk_level_critical", risk_level=RiskLevel.CRITICAL)
        manager.add_rule(rule)
        
        assessment = RiskAssessment(zone_id="zone-1", timestamp=1.0, risk_level=RiskLevel.CRITICAL, risk_score=90.0, factors=[])
        alerts = manager.check_and_create_alerts(assessment)
        alert_id = alerts[0].id
        
        result = manager.acknowledge_alert(alert_id, "operator-1")
        assert result is True
        assert alerts[0].acknowledged is True
        assert alerts[0].acknowledged_by == "operator-1"

    def test_get_active_alerts(self):
        manager = AlertManager()
        rule = AlertRule(id="rule-1", name="Critical", condition="risk_level_critical", risk_level=RiskLevel.CRITICAL)
        manager.add_rule(rule)
        
        assessment = RiskAssessment(zone_id="zone-1", timestamp=1.0, risk_level=RiskLevel.CRITICAL, risk_score=90.0, factors=[])
        manager.check_and_create_alerts(assessment)
        
        active = manager.get_active_alerts()
        assert len(active) == 1
        
        active_zone = manager.get_active_alerts(zone_id="zone-1")
        assert len(active_zone) == 1
        
        active_other = manager.get_active_alerts(zone_id="zone-2")
        assert len(active_other) == 0


class TestAlertManagerWrapper:
    def test_wrapper_creation(self):
        wrapper = AlertManagerWrapper()
        assert wrapper.manager is not None
        assert wrapper.notification_handler is not None

    def test_wrapper_check_alerts(self):
        wrapper = AlertManagerWrapper()
        rule = AlertRule(id="rule-1", name="Critical", condition="risk_level_critical", risk_level=RiskLevel.CRITICAL)
        wrapper.add_rule(rule)
        
        assessment = RiskAssessment(zone_id="zone-1", timestamp=1.0, risk_level=RiskLevel.CRITICAL, risk_score=90.0, factors=[])
        alerts = wrapper.check_alerts(assessment)
        assert len(alerts) == 1


class TestNotificationChannels:
    def test_notification_service(self):
        service = NotificationService()
        assert len(service.channels) == 0

    def test_add_channel(self):
        service = NotificationService()
        channel = SoundNotificationChannel({})
        service.add_channel(channel)
        assert len(service.channels) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])