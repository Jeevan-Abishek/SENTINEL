from typing import Optional, List
from loguru import logger

from app.risk.models import AlertManager, AlertConfig, Alert, AlertRule, NotificationHandler


class AlertManagerWrapper:
    def __init__(self, config: Optional[AlertConfig] = None):
        self.manager = AlertManager(config)
        self.notification_handler = NotificationHandler(config.sound_enabled if config else True)

    def add_rule(self, rule: AlertRule):
        self.manager.add_rule(rule)

    def remove_rule(self, rule_id: str):
        self.manager.remove_rule(rule_id)

    def check_alerts(self, assessment) -> List[Alert]:
        alerts = self.manager.check_and_create_alerts(assessment)
        for alert in alerts:
            self.notification_handler.notify(alert)
        return alerts

    def acknowledge(self, alert_id: str, operator_id: str) -> bool:
        return self.manager.acknowledge_alert(alert_id, operator_id)

    def get_active_alerts(self, zone_id: Optional[str] = None) -> List[Alert]:
        return self.manager.get_active_alerts(zone_id)

    def get_history(self, limit: int = 100) -> List[Alert]:
        return self.manager.get_alert_history(limit)

    def register_notification_handler(self, handler: callable):
        self.notification_handler.register_handler(handler)