"""
Alert Manager & Early Warning Lifecycle Coordinator for SENTINEL.
Provides alert deduplication, cooldown management, operator verification,
acknowledgment/resolution workflows, and 5+ minute early warning integration.
"""

from typing import Optional, List, Dict, Any, Union
import time
from loguru import logger
from pydantic import BaseModel, Field

try:
    from app.risk.models import AlertManager, AlertConfig, Alert, AlertRule, NotificationHandler, RiskLevel
except ImportError:
    from sentinel.app.risk.models import AlertManager, AlertConfig, Alert, AlertRule, NotificationHandler, RiskLevel

try:
    from app.alerts.early_warning import (
        EarlyWarningEngine,
        EarlyWarningConfig,
        EarlyWarningProjection,
        EarlyWarningAccuracyStatus,
        EarlyWarningSeverity,
    )
except ImportError:
    from sentinel.app.alerts.early_warning import (
        EarlyWarningEngine,
        EarlyWarningConfig,
        EarlyWarningProjection,
        EarlyWarningAccuracyStatus,
        EarlyWarningSeverity,
    )


class EarlyWarningAlert(BaseModel):
    """Structured, accuracy-gated 5+ minute early warning alert for operator action."""
    model_config = {"extra": "ignore"}

    id: str
    zone_id: str
    severity: EarlyWarningSeverity = EarlyWarningSeverity.WARNING
    confidence_score: float = 0.0           # 0.0 - 1.0
    accuracy_status: EarlyWarningAccuracyStatus = EarlyWarningAccuracyStatus.HIGH_CONFIDENCE
    estimated_lead_minutes: float = 5.0     # Lead time into horizon (>= 5 min)
    title: str = "5+ Min Early Warning"
    reason: str = ""
    steward_action: str = ""
    timestamp: float = Field(default_factory=time.time)
    status: str = "ACTIVE"                  # ACTIVE, ACKNOWLEDGED, RESOLVED
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[float] = None
    operator_notes: str = ""
    metrics: Dict[str, Any] = Field(default_factory=dict)

    @property
    def risk_level(self) -> str:
        """Dashboard compatibility alias."""
        return self.severity.value.lower()


class EarlyWarningAlertManager:
    """
    Accuracy-First Early Warning Alert Coordinator.
    Enforces deduplication, cooldown windows, multi-window confirmation,
    and operator lifecycle management (ACKNOWLEDGE / RESOLVE).
    """

    def __init__(self, config: Optional[EarlyWarningConfig] = None):
        if config is None:
            config = EarlyWarningConfig()
        self.config = config
        self.engine = EarlyWarningEngine(config)
        self.active_alerts: Dict[str, EarlyWarningAlert] = {}       # alert_id -> EarlyWarningAlert
        self.alert_history: List[EarlyWarningAlert] = []            # list of all alerts
        self.last_zone_alert_time: Dict[str, float] = {}            # zone_id -> timestamp of last emitted alert
        self.alert_counter: int = 0

    def process_telemetry(
        self,
        zone_id: str,
        timestamp: float,
        **kwargs,
    ) -> tuple[EarlyWarningProjection, Optional[EarlyWarningAlert]]:
        """
        Assess crowd dynamics for a zone and emit an EarlyWarningAlert if accuracy gates pass.
        Returns: (projection, alert_if_created_or_updated)
        """
        projection = self.engine.assess_early_warning(
            zone_id=zone_id,
            timestamp=timestamp,
            **kwargs,
        )

        alert = self.process_projection(projection)
        return projection, alert

    def process_projection(self, projection: EarlyWarningProjection) -> Optional[EarlyWarningAlert]:
        """
        Evaluate an EarlyWarningProjection. Withholds alerts if accuracy status is INSUFFICIENT_DATA
        or if is_early_warning is False. Deduplicates alerts within cooldown window.
        """
        if not projection.is_early_warning:
            return None

        if projection.accuracy_status == EarlyWarningAccuracyStatus.INSUFFICIENT_DATA:
            logger.debug(f"Withholding early warning for zone {projection.zone_id}: Insufficient data.")
            return None

        zone_id = projection.zone_id
        now = projection.timestamp

        # Check zone cooldown for deduplication
        last_time = self.last_zone_alert_time.get(zone_id, 0.0)
        cooldown = self.config.cooldown_seconds

        # Check if an existing ACTIVE alert exists for this zone
        existing_active = next(
            (a for a in self.active_alerts.values() if a.zone_id == zone_id and a.status == "ACTIVE"),
            None,
        )

        if existing_active:
            # Update telemetry on existing active alert without creating duplicate
            existing_active.confidence_score = projection.confidence_score
            existing_active.estimated_lead_minutes = projection.estimated_lead_minutes
            existing_active.metrics = projection.metrics
            existing_active.reason = projection.reason
            existing_active.steward_action = projection.steward_action
            existing_active.severity = projection.severity
            return existing_active

        if (now - last_time) < cooldown:
            logger.debug(f"Zone {zone_id} early warning within cooldown ({now - last_time:.1f}s < {cooldown:.1f}s). Deduplicating.")
            return None

        # Create new unique early warning alert
        self.alert_counter += 1
        alert_id = f"ew_alert_{int(now)}_{zone_id}_{self.alert_counter}"

        lead_display = f"{projection.estimated_lead_minutes:.1f}"
        alert = EarlyWarningAlert(
            id=alert_id,
            zone_id=zone_id,
            severity=projection.severity,
            confidence_score=projection.confidence_score,
            accuracy_status=projection.accuracy_status,
            estimated_lead_minutes=projection.estimated_lead_minutes,
            title=f"⏳ 5+ Min Early Warning: Zone {zone_id.replace('_', ' ').title()} (~{lead_display}m Lead)",
            reason=projection.reason,
            steward_action=projection.steward_action,
            timestamp=now,
            status="ACTIVE",
            metrics=projection.metrics,
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.last_zone_alert_time[zone_id] = now
        logger.info(f"🚨 New 5+ Min Early Warning Emitted: [{alert_id}] {alert.title} (Conf: {int(alert.confidence_score*100)}%)")
        return alert

    def acknowledge_alert(self, alert_id: str, operator_id: str = "operator_1", notes: str = "") -> bool:
        """Acknowledge an active early warning alert."""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            # Check history
            alert = next((a for a in self.alert_history if a.id == alert_id), None)
            if not alert:
                return False

        alert.acknowledged = True
        alert.acknowledged_by = operator_id
        alert.acknowledged_at = time.time()
        alert.status = "ACKNOWLEDGED"
        if notes:
            alert.operator_notes = f"{alert.operator_notes}\n[Ack Notes]: {notes}".strip()
        logger.info(f"✅ Early Warning Alert {alert_id} acknowledged by {operator_id}")
        return True

    def resolve_alert(self, alert_id: str, operator_id: str = "operator_1", notes: str = "") -> bool:
        """Resolve and archive an early warning alert."""
        alert = self.active_alerts.pop(alert_id, None)
        if not alert:
            alert = next((a for a in self.alert_history if a.id == alert_id), None)
            if not alert:
                return False

        alert.resolved = True
        alert.resolved_by = operator_id
        alert.resolved_at = time.time()
        alert.status = "RESOLVED"
        if notes:
            alert.operator_notes = f"{alert.operator_notes}\n[Resolve Notes]: {notes}".strip()
        logger.info(f"🏁 Early Warning Alert {alert_id} resolved by {operator_id}")
        return True

    def get_active_alerts(self, zone_id: Optional[str] = None) -> List[EarlyWarningAlert]:
        """Retrieve all currently active un-resolved early warning alerts."""
        alerts = list(self.active_alerts.values())
        if zone_id:
            alerts = [a for a in alerts if a.zone_id == zone_id]
        return sorted(alerts, key=lambda a: (a.severity == EarlyWarningSeverity.CRITICAL, a.timestamp), reverse=True)

    def get_alert_history(self, limit: int = 100) -> List[EarlyWarningAlert]:
        """Retrieve recent alert history."""
        return list(reversed(self.alert_history[-limit:]))

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level summary metrics for dashboard display."""
        active = list(self.active_alerts.values())
        return {
            "total_active": len(active),
            "critical_count": sum(1 for a in active if a.severity == EarlyWarningSeverity.CRITICAL),
            "warning_count": sum(1 for a in active if a.severity == EarlyWarningSeverity.WARNING),
            "high_confidence_count": sum(1 for a in active if a.accuracy_status == EarlyWarningAccuracyStatus.HIGH_CONFIDENCE),
            "moderate_confidence_count": sum(1 for a in active if a.accuracy_status == EarlyWarningAccuracyStatus.MODERATE_CONFIDENCE),
            "min_lead_time_min": min([a.estimated_lead_minutes for a in active], default=5.0),
        }

    def clear(self):
        """Clear active alerts and history."""
        self.active_alerts.clear()
        self.alert_history.clear()
        self.last_zone_alert_time.clear()
        self.engine.reset()


class AlertManagerWrapper:
    """Unified wrapper coordinating standard rule alerts and 5+ min early warning alerts."""

    def __init__(self, config: Optional[AlertConfig] = None, ew_config: Optional[EarlyWarningConfig] = None):
        self.manager = AlertManager(config)
        self.notification_handler = NotificationHandler(config.sound_enabled if config else True)
        self.early_warning_manager = EarlyWarningAlertManager(ew_config)

    def add_rule(self, rule: AlertRule):
        self.manager.add_rule(rule)

    def remove_rule(self, rule_id: str):
        self.manager.remove_rule(rule_id)

    def check_alerts(self, assessment) -> List[Alert]:
        alerts = self.manager.check_and_create_alerts(assessment)
        for alert in alerts:
            self.notification_handler.notify(alert)
        return alerts

    def process_early_warning(self, zone_id: str, timestamp: float, **kwargs) -> tuple[EarlyWarningProjection, Optional[EarlyWarningAlert]]:
        return self.early_warning_manager.process_telemetry(zone_id, timestamp, **kwargs)

    def acknowledge(self, alert_id: str, operator_id: str) -> bool:
        if alert_id.startswith("ew_"):
            return self.early_warning_manager.acknowledge_alert(alert_id, operator_id)
        return self.manager.acknowledge_alert(alert_id, operator_id)

    def resolve(self, alert_id: str, operator_id: str, notes: str = "") -> bool:
        if alert_id.startswith("ew_"):
            return self.early_warning_manager.resolve_alert(alert_id, operator_id, notes)
        return False

    def get_active_alerts(self, zone_id: Optional[str] = None) -> List[Alert]:
        return self.manager.get_active_alerts(zone_id)

    def get_active_early_warnings(self, zone_id: Optional[str] = None) -> List[EarlyWarningAlert]:
        return self.early_warning_manager.get_active_alerts(zone_id)

    def get_history(self, limit: int = 100) -> List[Alert]:
        return self.manager.get_alert_history(limit)

    def register_notification_handler(self, handler: callable):
        self.notification_handler.register_handler(handler)


__all__ = [
    "EarlyWarningAlert",
    "EarlyWarningAlertManager",
    "AlertManagerWrapper",
]
