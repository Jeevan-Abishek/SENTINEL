"""
Risk Models and Data Structures for SENTINEL.
Defines risk levels, factors, assessments, alerts, and configurations.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def label(self) -> str:
        return self.value.upper()

    @property
    def color(self) -> str:
        return "#2E7D32" if self == RiskLevel.SAFE else ("#E65100" if self == RiskLevel.WARNING else "#C62828")

    @property
    def bg_color(self) -> str:
        return "#E8F5E9" if self == RiskLevel.SAFE else ("#FFF3E0" if self == RiskLevel.WARNING else "#FFEBEE")

    @property
    def border_color(self) -> str:
        return "#2E7D32" if self == RiskLevel.SAFE else ("#E65100" if self == RiskLevel.WARNING else "#C62828")

    @property
    def pill_class(self) -> str:
        return f"status-{self.value}"

    @property
    def icon(self) -> str:
        return "🟢" if self == RiskLevel.SAFE else ("⚠️" if self == RiskLevel.WARNING else "🚨")

    @property
    def priority(self) -> int:
        return 0 if self == RiskLevel.SAFE else (1 if self == RiskLevel.WARNING else 2)


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class RiskFactor(BaseModel):
    name: str
    value: float
    weight: float
    score: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    contribution: float = 0.0
    normalized_score: float = 0.0


class RiskAssessment(BaseModel):
    zone_id: str
    timestamp: float
    risk_level: RiskLevel
    risk_score: float
    factors: List[RiskFactor] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    component_contributions: Dict[str, float] = Field(default_factory=dict)
    top_drivers: List[str] = Field(default_factory=list)
    explanation: str = ""
    lead_time_min: Optional[float] = None
    is_early_warning: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    id: str = Field(..., description="Unique alert identifier")
    zone_id: str
    risk_level: RiskLevel
    message: str
    timestamp: float
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertRule(BaseModel):
    id: str
    name: str
    zone_id: Optional[str] = None
    condition: str
    risk_level: RiskLevel
    cooldown_seconds: int = 30
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class RiskConfig:
    density_weight: float = 0.30
    occupancy_weight: float = 0.20
    movement_weight: float = 0.20
    flow_weight: float = 0.15
    bottleneck_weight: float = 0.15
    safe_threshold: float = 30.0
    warning_threshold: float = 60.0
    critical_threshold: float = 80.0
    temporal_smoothing: float = 0.30
    hysteresis_buffer: float = 3.0
    consecutive_escalation_frames: int = 1
    consecutive_deescalation_frames: int = 2
    occupancy_warning_threshold: float = 0.70
    occupancy_critical_threshold: float = 0.90
    warning_density: float = 3.0
    critical_density: float = 5.0
    predictive_weight: float = 0.20
    early_warning_lead_minutes: float = 5.0
    enable_predictive: bool = True

    def normalized_weights(self) -> Dict[str, float]:
        raw = {
            "density": max(0.0, float(self.density_weight)),
            "occupancy": max(0.0, float(self.occupancy_weight)),
            "movement": max(0.0, float(self.movement_weight)),
            "flow": max(0.0, float(self.flow_weight)),
            "bottleneck": max(0.0, float(self.bottleneck_weight)),
        }
        total = sum(raw.values())
        if total <= 1e-6:
            return {
                "density": 0.30,
                "occupancy": 0.20,
                "movement": 0.20,
                "flow": 0.15,
                "bottleneck": 0.15,
            }
        return {k: round(v / total, 4) for k, v in raw.items()}


@dataclass
class AlertConfig:
    cooldown_seconds: int = 30
    max_active: int = 100
    sound_enabled: bool = True


class AlertManager:
    def __init__(self, config: Optional[AlertConfig] = None):
        if config is None:
            config = AlertConfig()
        self.config = config
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.last_alert_time: Dict[str, float] = {}
        self.rules: List[AlertRule] = []

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def remove_rule(self, rule_id: str):
        self.rules = [r for r in self.rules if r.id != rule_id]

    def check_and_create_alerts(self, assessment: RiskAssessment) -> List[Alert]:
        new_alerts = []
        now = assessment.timestamp

        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.zone_id and rule.zone_id != assessment.zone_id:
                continue

            if not self._evaluate_condition(rule, assessment):
                continue

            # Check cooldown
            rule_key = f"{rule.id}_{assessment.zone_id}"
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
        cond = rule.condition.lower().strip()
        if cond == "risk_level_critical":
            return assessment.risk_level == RiskLevel.CRITICAL
        elif cond == "risk_level_warning":
            return assessment.risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL]
        elif cond == "risk_score_above_50":
            return assessment.risk_score > 50
        elif cond == "risk_score_above_70":
            return assessment.risk_score > 70
        elif cond == "any_factor_critical":
            return any(f.score > 80 for f in assessment.factors)
        elif cond == "predictive_early_warning":
            pred_f = next((f for f in assessment.factors if f.name == "predictive_5min"), None)
            return pred_f is not None and (pred_f.score >= 50.0 or pred_f.value >= 3.0)
        elif cond == "bottleneck_compression":
            bn_f = next((f for f in assessment.factors if f.name == "bottleneck"), None)
            return bn_f is not None and (bn_f.score >= 60.0 or bn_f.value >= 0.50)
        elif cond == "counter_flow_conflict":
            mv_f = next((f for f in assessment.factors if f.name == "movement"), None)
            return mv_f is not None and (mv_f.score >= 60.0)
        return False

    def _generate_message(self, rule: AlertRule, assessment: RiskAssessment) -> str:
        cond = rule.condition.lower().strip()
        if cond == "risk_level_critical":
            return f"CRITICAL risk detected in zone {assessment.zone_id}: Risk score {assessment.risk_score:.1f}"
        elif cond == "risk_level_warning":
            pred_f = next((f for f in assessment.factors if f.name == "predictive_5min"), None)
            if pred_f and pred_f.score >= 55:
                return (
                    f"WARNING (~4-5 min advance lead) in zone {assessment.zone_id}: "
                    f"Risk score {assessment.risk_score:.1f} | Projected density {pred_f.value:.2f} p/m²"
                )
            return f"WARNING risk detected in zone {assessment.zone_id}: Risk score {assessment.risk_score:.1f}"
        elif cond == "risk_score_above_50":
            return f"Elevated risk in zone {assessment.zone_id}: Score {assessment.risk_score:.1f}"
        elif cond == "risk_score_above_70":
            return f"High risk in zone {assessment.zone_id}: Score {assessment.risk_score:.1f}"
        elif cond == "any_factor_critical":
            critical_factors = [f.name for f in assessment.factors if f.score > 80]
            return f"Critical factors in zone {assessment.zone_id}: {', '.join(critical_factors)}"
        elif cond == "predictive_early_warning":
            pred_f = next((f for f in assessment.factors if f.name == "predictive_5min"), None)
            val = pred_f.value if pred_f else 0.0
            return (
                f"⏳ EARLY WARNING (~4-5 min lead): Projected congestion build-up in zone {assessment.zone_id} "
                f"(projected density: {val:.2f} p/m²). Preemptive crowd regulation advised."
            )
        elif cond == "bottleneck_compression":
            bn_f = next((f for f in assessment.factors if f.name == "bottleneck"), None)
            val = bn_f.value if bn_f else 0.0
            return f"Turnstile choke bottleneck alert in zone {assessment.zone_id} (compression index: {val:.2f})"
        return f"Alert triggered in zone {assessment.zone_id}"

    def _cleanup_alerts(self, now: float):
        resolved = []
        for alert in self.active_alerts:
            if now - alert.timestamp > 300:
                resolved.append(alert)
        for alert in resolved:
            self.active_alerts.remove(alert)
            self.alert_history.append(alert)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

    def acknowledge_alert(self, alert_id: str, operator_id: str) -> bool:
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = operator_id
                alert.acknowledged_at = datetime.now(timezone.utc).timestamp()
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
        self.handlers: List[Any] = []

    def register_handler(self, handler: Any):
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
        pass


def check_and_create_alerts(
    assessment: RiskAssessment,
    rules: Optional[List[AlertRule]] = None,
    manager: Optional[AlertManager] = None,
) -> List[Alert]:
    """Top-level helper to evaluate rules and create alerts."""
    if manager is not None:
        return manager.check_and_create_alerts(assessment)
    from app.risk.rules import get_default_ruleset
    mgr = AlertManager()
    for r in (rules or get_default_ruleset()):
        mgr.add_rule(r)
    return mgr.check_and_create_alerts(assessment)


# Re-export RiskEngine and RiskEngineWrapper from app.risk.engine
from app.risk.engine import RiskEngine, RiskEngineWrapper

CrowdRiskEngine = RiskEngine

__all__ = [
    "RiskLevel",
    "AlertSeverity",
    "AlertStatus",
    "RiskFactor",
    "RiskAssessment",
    "RiskConfig",
    "RiskEngine",
    "RiskEngineWrapper",
    "CrowdRiskEngine",
    "Alert",
    "AlertRule",
    "AlertConfig",
    "AlertManager",
    "NotificationHandler",
    "check_and_create_alerts",
]
