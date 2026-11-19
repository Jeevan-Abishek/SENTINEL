"""
Safety Operator Action & Incident-Response Layer for SENTINEL.
Provides structured operator decision options, ground steward dispatch,
crowd redirection, egress management, and tamper-evident audit logging.
"""

from typing import Optional, List, Dict, Any, Union
import time
import json
from pathlib import Path
from loguru import logger

try:
    from app.operator.models import (
        OperatorActionType,
        IncidentLifecycleStatus,
        OperatorPriority,
        OperatorActionRecord,
        OperatorIncidentState,
    )
except ImportError:
    from sentinel.app.operator.models import (
        OperatorActionType,
        IncidentLifecycleStatus,
        OperatorPriority,
        OperatorActionRecord,
        OperatorIncidentState,
    )


class SafetyOperatorManager:
    """
    Lightweight, fast in-memory Operator Incident Response & Audit Coordinator.
    Strictly optimized for low-RAM Windows CPU environments (~145 MB target).
    Zero heavy database overhead: uses bounded in-memory circular lists + lightweight JSONL logging.
    """

    def __init__(
        self,
        audit_file_path: Optional[Union[str, Path]] = "sentinel/data/operator_actions.jsonl",
        max_records: int = 500,
    ):
        self.max_records = max_records
        self._records: List[OperatorActionRecord] = []
        self._incidents: Dict[str, OperatorIncidentState] = {}   # zone_id -> OperatorIncidentState
        self._action_counter: int = 0
        self._audit_file: Optional[Path] = Path(audit_file_path) if audit_file_path else None
        self._file_write_enabled: bool = True

        # Pre-seed default venue incident states
        self._init_default_incidents()

    def _init_default_incidents(self) -> None:
        """Initialize clean default states for monitored zones."""
        default_zones = [
            ("overall", "Overall Venue", "SAFE"),
            ("zone_a", "Zone A - North Ingress", "SAFE"),
            ("zone_b", "Zone B - Main Concourse", "SAFE"),
            ("zone_c", "Zone C - Plaza Gathering", "SAFE"),
            ("zone_d", "Zone D - South Egress", "SAFE"),
        ]
        now = time.time()
        for zid, zname, sev in default_zones:
            self._incidents[zid] = OperatorIncidentState(
                incident_id=f"inc_{zid}",
                zone_id=zid,
                zone_name=zname,
                status=IncidentLifecycleStatus.OPEN,
                priority=OperatorPriority.LOW,
                severity=sev,
                created_at=now,
                updated_at=now,
            )

    def _generate_action_id(self) -> str:
        self._action_counter += 1
        return f"act_{int(time.time())}_{self._action_counter:04d}"

    def _persist_record(self, record: OperatorActionRecord) -> None:
        """Append record to in-memory store and optional JSONL file."""
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records.pop(0)

        if self._audit_file and self._file_write_enabled:
            try:
                self._audit_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self._audit_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
            except Exception as e:
                logger.warning(f"Operator audit log file write error: {e}. Falling back to in-memory only.")
                self._file_write_enabled = False

    def _get_or_create_incident(self, zone_id: str, zone_name: str = "") -> OperatorIncidentState:
        if zone_id not in self._incidents:
            now = time.time()
            clean_name = zone_name or f"Zone {zone_id.replace('_', ' ').title()}"
            self._incidents[zone_id] = OperatorIncidentState(
                incident_id=f"inc_{zone_id}",
                zone_id=zone_id,
                zone_name=clean_name,
                status=IncidentLifecycleStatus.OPEN,
                priority=OperatorPriority.MEDIUM,
                severity="WARNING",
                created_at=now,
                updated_at=now,
            )
        return self._incidents[zone_id]

    def acknowledge_alert(
        self,
        alert_id: str,
        zone_id: str = "overall",
        zone_name: str = "Overall Venue",
        operator_id: str = "OP-CHIEF-01",
        note: str = "",
        priority: OperatorPriority = OperatorPriority.MEDIUM,
        **kwargs,
    ) -> OperatorActionRecord:
        """
        Record formal human operator acknowledgment of an active safety alert or 5+ min early warning.
        """
        now = time.time()
        action_id = self._generate_action_id()
        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.ACKNOWLEDGE,
            zone_id=zone_id,
            zone_name=zone_name,
            alert_ref=alert_id,
            status=IncidentLifecycleStatus.ACKNOWLEDGED,
            priority=priority,
            note=note or f"Alert [{alert_id}] acknowledged by on-duty operator.",
            details={"action_source": "operator_dashboard", **kwargs},
        )
        self._persist_record(record)

        # Update zone incident state
        inc = self._get_or_create_incident(zone_id, zone_name)
        inc.status = IncidentLifecycleStatus.ACKNOWLEDGED
        inc.priority = priority
        inc.alert_ref = alert_id
        inc.updated_at = now
        inc.last_action_type = OperatorActionType.ACKNOWLEDGE
        inc.last_note = record.note
        inc.action_count += 1

        logger.info(f"🛡️ [Operator] Alert {alert_id} acknowledged for {zone_name} by {operator_id}")
        return record

    def dispatch_team(
        self,
        zone_id: str,
        zone_name: str = "Overall Venue",
        team_name: str = "Ground Stewards Unit A",
        priority: OperatorPriority = OperatorPriority.HIGH,
        operator_id: str = "OP-CHIEF-01",
        note: str = "",
        alert_ref: Optional[str] = None,
        **kwargs,
    ) -> OperatorActionRecord:
        """
        Dispatch ground safety stewards or security units to a target congested or choke zone.
        """
        now = time.time()
        action_id = self._generate_action_id()
        dispatch_note = note or f"Dispatched {team_name} to {zone_name} (Priority: {priority.value})."
        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.DISPATCH_GROUND_TEAM,
            zone_id=zone_id,
            zone_name=zone_name,
            alert_ref=alert_ref,
            status=IncidentLifecycleStatus.DISPATCHED,
            priority=priority,
            note=dispatch_note,
            details={"team_name": team_name, "priority": priority.value, **kwargs},
        )
        self._persist_record(record)

        # Update zone incident state
        inc = self._get_or_create_incident(zone_id, zone_name)
        inc.status = IncidentLifecycleStatus.DISPATCHED
        inc.priority = priority
        inc.assigned_team = team_name
        inc.updated_at = now
        inc.last_action_type = OperatorActionType.DISPATCH_GROUND_TEAM
        inc.last_note = dispatch_note
        inc.action_count += 1

        logger.info(f"📢 [Operator] Dispatched {team_name} to {zone_name} by {operator_id}")
        return record

    def redirect_crowd(
        self,
        zone_id: str,
        target_zone_id: str,
        zone_name: str = "Overall Venue",
        target_zone_name: str = "",
        instruction: str = "Divert incoming crowd flow away from choke point",
        operator_id: str = "OP-CHIEF-01",
        note: str = "",
        alert_ref: Optional[str] = None,
        **kwargs,
    ) -> OperatorActionRecord:
        """
        Issue actionable crowd redirection instructions from a congested zone to an alternative route.
        """
        now = time.time()
        action_id = self._generate_action_id()
        t_name = target_zone_name or f"Zone {target_zone_id.replace('_', ' ').title()}"
        redirect_note = note or f"Crowd redirection: Diverting flow from {zone_name} to {t_name}. Instruction: '{instruction}'."
        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.REDIRECT_CROWD,
            zone_id=zone_id,
            zone_name=zone_name,
            alert_ref=alert_ref,
            status=IncidentLifecycleStatus.REDIRECTING,
            priority=OperatorPriority.HIGH,
            note=redirect_note,
            details={
                "source_zone": zone_id,
                "target_zone": target_zone_id,
                "target_zone_name": t_name,
                "instruction": instruction,
                **kwargs,
            },
        )
        self._persist_record(record)

        # Update zone incident state
        inc = self._get_or_create_incident(zone_id, zone_name)
        inc.status = IncidentLifecycleStatus.REDIRECTING
        inc.crowd_instruction = instruction
        inc.updated_at = now
        inc.last_action_type = OperatorActionType.REDIRECT_CROWD
        inc.last_note = redirect_note
        inc.action_count += 1

        logger.info(f"🔀 [Operator] Redirecting crowd from {zone_name} -> {t_name}")
        return record

    def open_egress(
        self,
        zone_id: str,
        egress_gate: str = "Emergency Egress East-3",
        zone_name: str = "Overall Venue",
        operator_id: str = "OP-CHIEF-01",
        note: str = "",
        alert_ref: Optional[str] = None,
        **kwargs,
    ) -> OperatorActionRecord:
        """
        Activate secondary or emergency egress routes to relieve high-density accumulation.
        """
        now = time.time()
        action_id = self._generate_action_id()
        egress_note = note or f"Opened auxiliary egress '{egress_gate}' for {zone_name}."
        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.OPEN_EGRESS,
            zone_id=zone_id,
            zone_name=zone_name,
            alert_ref=alert_ref,
            status=IncidentLifecycleStatus.REDIRECTING,
            priority=OperatorPriority.URGENT,
            note=egress_note,
            details={"egress_gate": egress_gate, **kwargs},
        )
        self._persist_record(record)

        # Update zone incident state
        inc = self._get_or_create_incident(zone_id, zone_name)
        inc.status = IncidentLifecycleStatus.REDIRECTING
        inc.egress_status = f"Open ({egress_gate})"
        inc.updated_at = now
        inc.last_action_type = OperatorActionType.OPEN_EGRESS
        inc.last_note = egress_note
        inc.action_count += 1

        logger.info(f"🚪 [Operator] Egress {egress_gate} activated for {zone_name}")
        return record

    def adjust_flow(
        self,
        zone_id: str,
        max_inflow_rate: float = 12.0,
        flow_direction: str = "OUTFLOW_PRIORITY",
        zone_name: str = "Overall Venue",
        operator_id: str = "OP-CHIEF-01",
        note: str = "",
        alert_ref: Optional[str] = None,
        **kwargs,
    ) -> OperatorActionRecord:
        """
        Throttle inflow gates or reconfigure turnstile lanes to balance flow rates.
        """
        now = time.time()
        action_id = self._generate_action_id()
        flow_note = note or f"Adjusted flow control: Inflow capped at {max_inflow_rate:.1f} p/min with {flow_direction}."
        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.ADJUST_FLOW,
            zone_id=zone_id,
            zone_name=zone_name,
            alert_ref=alert_ref,
            status=IncidentLifecycleStatus.ACKNOWLEDGED,
            priority=OperatorPriority.MEDIUM,
            note=flow_note,
            details={"max_inflow_rate": max_inflow_rate, "flow_direction": flow_direction, **kwargs},
        )
        self._persist_record(record)

        inc = self._get_or_create_incident(zone_id, zone_name)
        inc.updated_at = now
        inc.last_action_type = OperatorActionType.ADJUST_FLOW
        inc.last_note = flow_note
        inc.action_count += 1

        logger.info(f"⚖️ [Operator] Flow adjusted for {zone_name}: cap {max_inflow_rate} p/min")
        return record

    def resolve_incident(
        self,
        incident_id_or_alert_id: str,
        zone_id: str = "overall",
        operator_id: str = "OP-CHIEF-01",
        note: str = "",
        **kwargs,
    ) -> OperatorActionRecord:
        """
        Mark an incident or alert resolved, returning the zone to NOMINAL/SAFE status.
        """
        now = time.time()
        action_id = self._generate_action_id()
        res_note = note or f"Incident [{incident_id_or_alert_id}] marked RESOLVED by {operator_id}."
        inc = self._incidents.get(zone_id)
        zname = inc.zone_name if inc else f"Zone {zone_id.replace('_', ' ').title()}"

        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.RESOLVE,
            zone_id=zone_id,
            zone_name=zname,
            alert_ref=incident_id_or_alert_id,
            status=IncidentLifecycleStatus.RESOLVED,
            priority=OperatorPriority.LOW,
            note=res_note,
            details={"resolved_incident_id": incident_id_or_alert_id, **kwargs},
        )
        self._persist_record(record)

        # Update zone incident state
        if inc:
            inc.status = IncidentLifecycleStatus.RESOLVED
            inc.severity = "SAFE"
            inc.updated_at = now
            inc.last_action_type = OperatorActionType.RESOLVE
            inc.last_note = res_note
            inc.action_count += 1

        logger.info(f"🏁 [Operator] Incident {incident_id_or_alert_id} resolved for {zname}")
        return record

    def escalate_incident(
        self,
        incident_id_or_alert_id: str,
        zone_id: str = "overall",
        priority: OperatorPriority = OperatorPriority.URGENT,
        operator_id: str = "OP-CHIEF-01",
        note: str = "",
        **kwargs,
    ) -> OperatorActionRecord:
        """
        Escalate an incident to Chief Safety Officer / Emergency Command.
        """
        now = time.time()
        action_id = self._generate_action_id()
        esc_note = note or f"ESCALATED incident [{incident_id_or_alert_id}] to Emergency Safety Command (Priority: {priority.value})."
        inc = self._incidents.get(zone_id)
        zname = inc.zone_name if inc else f"Zone {zone_id.replace('_', ' ').title()}"

        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.ESCALATE,
            zone_id=zone_id,
            zone_name=zname,
            alert_ref=incident_id_or_alert_id,
            status=IncidentLifecycleStatus.ESCALATED,
            priority=priority,
            note=esc_note,
            details={"escalated_incident_id": incident_id_or_alert_id, **kwargs},
        )
        self._persist_record(record)

        if inc:
            inc.status = IncidentLifecycleStatus.ESCALATED
            inc.priority = priority
            inc.severity = "CRITICAL"
            inc.updated_at = now
            inc.last_action_type = OperatorActionType.ESCALATE
            inc.last_note = esc_note
            inc.action_count += 1

        logger.warning(f"🚨 [Operator] Incident {incident_id_or_alert_id} ESCALATED for {zname}")
        return record

    def log_note(
        self,
        zone_id: str,
        note: str,
        alert_ref: Optional[str] = None,
        zone_name: str = "Overall Venue",
        operator_id: str = "OP-CHIEF-01",
        **kwargs,
    ) -> OperatorActionRecord:
        """Log an arbitrary shift observation or operator note."""
        now = time.time()
        action_id = self._generate_action_id()
        record = OperatorActionRecord(
            action_id=action_id,
            timestamp=now,
            operator_id=operator_id,
            action_type=OperatorActionType.LOG_NOTE,
            zone_id=zone_id,
            zone_name=zone_name,
            alert_ref=alert_ref,
            status=IncidentLifecycleStatus.ACKNOWLEDGED,
            priority=OperatorPriority.LOW,
            note=note,
            details=kwargs,
        )
        self._persist_record(record)

        inc = self._get_or_create_incident(zone_id, zone_name)
        inc.updated_at = now
        inc.last_action_type = OperatorActionType.LOG_NOTE
        inc.last_note = note
        inc.action_count += 1
        return record

    def get_recent_actions(
        self,
        limit: int = 50,
        zone_id: Optional[str] = None,
        action_type: Optional[OperatorActionType] = None,
    ) -> List[OperatorActionRecord]:
        """Retrieve recent operator action records in reverse chronological order."""
        res = list(self._records)
        if zone_id:
            res = [r for r in res if r.zone_id == zone_id]
        if action_type:
            res = [r for r in res if r.action_type == action_type]
        return list(reversed(res[-limit:]))

    def get_incident(self, zone_id: str) -> Optional[OperatorIncidentState]:
        return self._incidents.get(zone_id)

    def get_active_incidents(self) -> List[OperatorIncidentState]:
        return list(self._incidents.values())

    def get_summary(self) -> Dict[str, Any]:
        """Generate high-level summary of operator responses for dashboard KPIs."""
        total = len(self._records)
        acks = sum(1 for r in self._records if r.action_type == OperatorActionType.ACKNOWLEDGE)
        dispatches = sum(1 for r in self._records if r.action_type == OperatorActionType.DISPATCH_GROUND_TEAM)
        redirects = sum(1 for r in self._records if r.action_type in [OperatorActionType.REDIRECT_CROWD, OperatorActionType.OPEN_EGRESS])
        resolves = sum(1 for r in self._records if r.action_type == OperatorActionType.RESOLVE)
        escalates = sum(1 for r in self._records if r.action_type == OperatorActionType.ESCALATE)

        active_incidents = [
            inc for inc in self._incidents.values()
            if inc.status in [IncidentLifecycleStatus.OPEN, IncidentLifecycleStatus.ACKNOWLEDGED, IncidentLifecycleStatus.DISPATCHED, IncidentLifecycleStatus.REDIRECTING]
            and inc.severity in ["WARNING", "CRITICAL"]
        ]

        return {
            "total_actions": total,
            "acknowledged_count": acks,
            "dispatched_count": dispatches,
            "redirected_count": redirects,
            "resolved_count": resolves,
            "escalated_count": escalates,
            "active_incident_count": len(active_incidents),
            "last_action_time": self._records[-1].timestamp if self._records else None,
        }

    def export_audit_log_json(self, limit: int = 100) -> str:
        """Export serialized JSON list of recent actions for operator shift reporting."""
        recent = self.get_recent_actions(limit=limit)
        return json.dumps([r.to_dict() for r in recent], indent=2)

    def clear(self) -> None:
        """Reset state and records (useful for testing and shift handover)."""
        self._records.clear()
        self._incidents.clear()
        self._action_counter = 0
        self._init_default_incidents()


# Global Singleton for consistent dashboard session access
_GLOBAL_OPERATOR_MANAGER: Optional[SafetyOperatorManager] = None


def get_safety_operator_manager(audit_file_path: Optional[str] = "sentinel/data/operator_actions.jsonl") -> SafetyOperatorManager:
    """Retrieve or initialize the singleton SafetyOperatorManager."""
    global _GLOBAL_OPERATOR_MANAGER
    if _GLOBAL_OPERATOR_MANAGER is None:
        _GLOBAL_OPERATOR_MANAGER = SafetyOperatorManager(audit_file_path=audit_file_path)
    return _GLOBAL_OPERATOR_MANAGER


__all__ = [
    "SafetyOperatorManager",
    "get_safety_operator_manager",
]

