from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskFactor(BaseModel):
    name: str
    value: float
    weight: float
    score: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


class RiskAssessment(BaseModel):
    zone_id: str
    timestamp: float
    risk_level: RiskLevel
    risk_score: float
    factors: List[RiskFactor]
    recommended_actions: List[str] = Field(default_factory=list)


class Alert(BaseModel):
    id: str = Field(..., description="Unique alert identifier")
    zone_id: str
    risk_level: RiskLevel
    message: str
    timestamp: float
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    metadata: dict = Field(default_factory=dict)


class AlertRule(BaseModel):
    id: str
    name: str
    zone_id: Optional[str] = None
    condition: str
    risk_level: RiskLevel
    cooldown_seconds: int = 30
    enabled: bool = True
    metadata: dict = Field(default_factory=dict)


@dataclass
class RiskConfig:
    density_weight: float = 0.3
    occupancy_weight: float = 0.2
    movement_weight: float = 0.2
    flow_weight: float = 0.15
    bottleneck_weight: float = 0.15
    safe_threshold: float = 30.0
    warning_threshold: float = 60.0
    critical_threshold: float = 80.0
    temporal_smoothing: float = 0.3
    occupancy_warning_threshold: float = 0.7
    occupancy_critical_threshold: float = 0.9


@dataclass
class AlertConfig:
    cooldown_seconds: int = 30
    max_active: int = 100
    sound_enabled: bool = True


class RiskEngine:
    def __init__(self, config: Optional[RiskConfig] = None):
        if config is None:
            config = RiskConfig()
        self.config = config
        self.previous_scores: dict[str, float] = {}

    def assess(
        self,
        zone_id: str,
        timestamp: float,
        density_result: Optional["DensityResult"] = None,
        occupancy_result: Optional["OccupancyResult"] = None,
        movement_result: Optional["MovementResult"] = None,
        flow_result: Optional["FlowResult"] = None,
        bottleneck_result: Optional["BottleneckResult"] = None,
    ) -> RiskAssessment:
        from app.analytics.models import (
            DensityResult, OccupancyResult, MovementResult, FlowResult, BottleneckResult
        )
        
        factors = []
        
        # Density factor
        if density_result:
            # Quadratic scaling for density - higher densities get disproportionately higher scores
            density_score = density_result.max_density ** 1.5 * 8
            density_factor = RiskFactor(
                name="density",
                value=density_result.max_density,
                weight=self.config.density_weight,
                score=density_score,
                threshold_warning=3.0,
                threshold_critical=5.0,
            )
            factors.append(density_factor)
        
        # Occupancy factor
        if occupancy_result:
            occupancy_score = occupancy_result.occupancy_ratio * 100
            occupancy_factor = RiskFactor(
                name="occupancy",
                value=occupancy_result.occupancy_ratio,
                weight=self.config.occupancy_weight,
                score=occupancy_score,
                threshold_warning=self.config.occupancy_warning_threshold,
                threshold_critical=self.config.occupancy_critical_threshold,
            )
            factors.append(occupancy_factor)
        
        # Movement factor (low speed = higher risk)
        if movement_result:
            # Risk increases as speed decreases (congestion)
            avg_speed = movement_result.avg_speed
            movement_score = min(100 / (avg_speed + 0.5), 100) if avg_speed > 0 else 100
            movement_factor = RiskFactor(
                name="movement",
                value=avg_speed,
                weight=self.config.movement_weight,
                score=movement_score,
                threshold_warning=1.0,
                threshold_critical=0.5,
            )
            factors.append(movement_factor)
        
        # Flow factor
        if flow_result:
            flow_score = min(flow_result.flow_rate * 5, 100)
            flow_factor = RiskFactor(
                name="flow",
                value=flow_result.flow_rate,
                weight=self.config.flow_weight,
                score=flow_score,
                threshold_warning=5.0,
                threshold_critical=10.0,
            )
            factors.append(flow_factor)
        
        # Bottleneck factor
        if bottleneck_result:
            bottleneck_score = bottleneck_result.bottleneck_score * 100
            bottleneck_factor = RiskFactor(
                name="bottleneck",
                value=bottleneck_result.bottleneck_score,
                weight=self.config.bottleneck_weight,
                score=bottleneck_score,
                threshold_warning=0.5,
                threshold_critical=0.8,
            )
            factors.append(bottleneck_factor)
        
        # Calculate weighted risk score
        if factors:
            raw_score = sum(f.weight * f.score for f in factors)
        else:
            raw_score = 0.0
        
        # Apply temporal smoothing
        if zone_id in self.previous_scores:
            smoothed_score = (
                self.config.temporal_smoothing * self.previous_scores[zone_id] +
                (1 - self.config.temporal_smoothing) * raw_score
            )
        else:
            smoothed_score = raw_score
        
        self.previous_scores[zone_id] = smoothed_score
        
        # Determine risk level
        if smoothed_score >= self.config.critical_threshold:
            risk_level = RiskLevel.CRITICAL
        elif smoothed_score >= self.config.warning_threshold:
            risk_level = RiskLevel.WARNING
        else:
            risk_level = RiskLevel.SAFE
        
        # Generate recommended actions
        recommended_actions = self._generate_actions(risk_level, factors)
        
        return RiskAssessment(
            zone_id=zone_id,
            timestamp=timestamp,
            risk_level=risk_level,
            risk_score=smoothed_score,
            factors=factors,
            recommended_actions=recommended_actions,
        )

    def _generate_actions(self, risk_level: RiskLevel, factors: List[RiskFactor]) -> List[str]:
        actions = []
        
        if risk_level == RiskLevel.CRITICAL:
            actions.append("Immediate evacuation of zone recommended")
            actions.append("Deploy security personnel to manage crowd")
            actions.append("Alert emergency services")
        elif risk_level == RiskLevel.WARNING:
            actions.append("Increase monitoring frequency")
            actions.append("Prepare crowd control measures")
            actions.append("Notify zone operators")
        
        # Factor-specific actions
        for factor in factors:
            if factor.name == "density" and factor.score > 70:
                actions.append("Redirect incoming crowd flow")
            elif factor.name == "occupancy" and factor.score > 80:
                actions.append("Limit entry to zone")
            elif factor.name == "bottleneck" and factor.score > 60:
                actions.append("Open alternative routes")
        
        return list(set(actions))  # Remove duplicates


class AlertManager:
    def __init__(self, config: Optional[AlertConfig] = None):
        if config is None:
            config = AlertConfig()
        self.config = config
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.last_alert_time: dict[str, float] = {}
        self.rules: List[AlertRule] = []

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def remove_rule(self, rule_id: str):
        self.rules = [r for r in self.rules if r.id != rule_id]

    def check_and_create_alerts(self, assessment: RiskAssessment) -> List[Alert]:
        new_alerts = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if rule.zone_id and rule.zone_id != assessment.zone_id:
                continue
            
            if not self._evaluate_condition(rule, assessment):
                continue
            
            # Check cooldown
            rule_key = f"{rule.id}_{assessment.zone_id}"
            now = assessment.timestamp
            if rule_key in self.last_alert_time:
                if now - self.last_alert_time[rule_key] < rule.cooldown_seconds:
                    continue
            
            # Create alert
            alert = Alert(
                id=f"alert_{len(self.alert_history) + len(self.active_alerts) + 1}",
                zone_id=assessment.zone_id,
                risk_level=rule.risk_level,
                message=self._generate_message(rule, assessment),
                timestamp=now,
                metadata={
                    "rule_id": rule.id,
                    "risk_score": assessment.risk_score,
                    "factors": [f.name for f in assessment.factors],
                },
            )
            
            new_alerts.append(alert)
            self.active_alerts.append(alert)
            self.last_alert_time[rule_key] = now
        
        # Clean up old alerts
        self._cleanup_alerts(now)
        
        return new_alerts

    def _evaluate_condition(self, rule: AlertRule, assessment: RiskAssessment) -> bool:
        # Simple condition evaluation - can be extended with a proper expression parser
        if rule.condition == "risk_level_critical":
            return assessment.risk_level == RiskLevel.CRITICAL
        elif rule.condition == "risk_level_warning":
            return assessment.risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL]
        elif rule.condition == "risk_score_above_50":
            return assessment.risk_score > 50
        elif rule.condition == "risk_score_above_70":
            return assessment.risk_score > 70
        elif rule.condition == "any_factor_critical":
            return any(f.score > 80 for f in assessment.factors)
        return False

    def _generate_message(self, rule: AlertRule, assessment: RiskAssessment) -> str:
        if rule.condition == "risk_level_critical":
            return f"CRITICAL risk detected in zone {assessment.zone_id}: Risk score {assessment.risk_score:.1f}"
        elif rule.condition == "risk_level_warning":
            return f"WARNING risk detected in zone {assessment.zone_id}: Risk score {assessment.risk_score:.1f}"
        elif rule.condition == "risk_score_above_50":
            return f"Elevated risk in zone {assessment.zone_id}: Score {assessment.risk_score:.1f}"
        elif rule.condition == "risk_score_above_70":
            return f"High risk in zone {assessment.zone_id}: Score {assessment.risk_score:.1f}"
        elif rule.condition == "any_factor_critical":
            critical_factors = [f.name for f in assessment.factors if f.score > 80]
            return f"Critical factors in zone {assessment.zone_id}: {', '.join(critical_factors)}"
        return f"Alert triggered in zone {assessment.zone_id}"

    def _cleanup_alerts(self, now: float):
        # Move resolved alerts to history
        resolved = []
        for alert in self.active_alerts:
            # Auto-resolve alerts older than 5 minutes if risk level dropped
            if now - alert.timestamp > 300:
                resolved.append(alert)
        
        for alert in resolved:
            self.active_alerts.remove(alert)
            self.alert_history.append(alert)
        
        # Limit history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    def acknowledge_alert(self, alert_id: str, operator_id: str) -> bool:
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = operator_id
                alert.acknowledged_at = datetime.utcnow().timestamp()
                return True
        return False

    def get_active_alerts(self, zone_id: Optional[str] = None) -> List[Alert]:
        alerts = self.active_alerts
        if zone_id:
            alerts = [a for a in alerts if a.zone_id == zone_id]
        return alerts

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        return self.alert_history[-limit:]


class NotificationHandler:
    def __init__(self, sound_enabled: bool = True):
        self.sound_enabled = sound_enabled
        self.handlers: List[callable] = []

    def register_handler(self, handler: callable):
        self.handlers.append(handler)

    def notify(self, alert: Alert):
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Notification handler error: {e}")
        
        if self.sound_enabled:
            self._play_sound(alert.risk_level)

    def _play_sound(self, risk_level: RiskLevel):
        # In production, this would play an actual sound
        # For now, just log
        print(f"ALERT SOUND: {risk_level.value.upper()}")