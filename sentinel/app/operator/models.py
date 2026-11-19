"""
Safety Operator Data Models for SENTINEL.
Defines typed models, lifecycle states, priority levels, and audit trail records
for human-in-the-loop incident response in the Eco-Safety Command Center.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
import time
from pydantic import BaseModel, Field


class OperatorActionType(str, Enum):
    """Categorization of operator response actions."""
    ACKNOWLEDGE = "ACKNOWLEDGE"
    DISPATCH_GROUND_TEAM = "DISPATCH_GROUND_TEAM"
    REDIRECT_CROWD = "REDIRECT_CROWD"
    OPEN_EGRESS = "OPEN_EGRESS"
    ADJUST_FLOW = "ADJUST_FLOW"
    RESOLVE = "RESOLVE"
    ESCALATE = "ESCALATE"
    LOG_NOTE = "LOG_NOTE"


class IncidentLifecycleStatus(str, Enum):
    """State lifecycle of a safety incident or alert under operator management."""
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISPATCHED = "DISPATCHED"
    REDIRECTING = "REDIRECTING"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class OperatorPriority(str, Enum):
    """Priority level for ground response and steward actions."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class OperatorActionRecord(BaseModel):
    """
    Immutable audit record of a specific safety action executed by an operator.
    Optimized for low-RAM in-memory storage (~145 MB target) and lightweight JSON serialization.
    """
    model_config = {"extra": "ignore"}

    action_id: str
    timestamp: float = Field(default_factory=time.time)
    timestamp_str: str = ""
    operator_id: str = "OP-CHIEF-01"
    action_type: OperatorActionType
    zone_id: str = "overall"
    zone_name: str = "Overall Venue"
    alert_ref: Optional[str] = None
    status: IncidentLifecycleStatus = IncidentLifecycleStatus.ACKNOWLEDGED
    priority: OperatorPriority = OperatorPriority.MEDIUM
    note: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.timestamp_str:
            try:
                self.timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))
            except Exception:
                self.timestamp_str = str(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to primitive dictionary for JSON audit logging and UI display."""
        return {
            "action_id": self.action_id,
            "timestamp": self.timestamp,
            "timestamp_str": self.timestamp_str,
            "operator_id": self.operator_id,
            "action_type": self.action_type.value if hasattr(self.action_type, "value") else str(self.action_type),
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "alert_ref": self.alert_ref,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "priority": self.priority.value if hasattr(self.priority, "value") else str(self.priority),
            "note": self.note,
            "details": self.details,
        }


class OperatorIncidentState(BaseModel):
    """
    Current live operational status for an active zone or incident under operator supervision.
    """
    model_config = {"extra": "ignore"}

    incident_id: str
    zone_id: str
    zone_name: str = "Overall Venue"
    status: IncidentLifecycleStatus = IncidentLifecycleStatus.OPEN
    priority: OperatorPriority = OperatorPriority.MEDIUM
    severity: str = "SAFE"                  # SAFE, WARNING, CRITICAL
    alert_ref: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    last_action_type: Optional[OperatorActionType] = None
    last_note: str = ""
    action_count: int = 0
    assigned_team: Optional[str] = None
    crowd_instruction: Optional[str] = None
    egress_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "zone_id": self.zone_id,
            "zone_name": self.zone_name,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "priority": self.priority.value if hasattr(self.priority, "value") else str(self.priority),
            "severity": self.severity,
            "alert_ref": self.alert_ref,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_action_type": self.last_action_type.value if hasattr(self.last_action_type, "value") else str(self.last_action_type),
            "last_note": self.last_note,
            "action_count": self.action_count,
            "assigned_team": self.assigned_team,
            "crowd_instruction": self.crowd_instruction,
            "egress_status": self.egress_status,
        }


__all__ = [
    "OperatorActionType",
    "IncidentLifecycleStatus",
    "OperatorPriority",
    "OperatorActionRecord",
    "OperatorIncidentState",
]

