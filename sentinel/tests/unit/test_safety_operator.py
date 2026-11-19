"""
Unit tests for the Safety Operator Action & Incident-Response Layer in SENTINEL.
Verifies models, action lifecycles, ground dispatches, crowd redirections,
egress control, audit trails, and low-RAM memory bounds (~145 MB target).
"""

import pytest
import time
import json
from pathlib import Path

from sentinel.app.operator.models import (
    OperatorActionType,
    IncidentLifecycleStatus,
    OperatorPriority,
    OperatorActionRecord,
    OperatorIncidentState,
)
from sentinel.app.operator.actions import (
    SafetyOperatorManager,
    get_safety_operator_manager,
)


@pytest.fixture
def clean_operator_mgr(tmp_path):
    audit_file = tmp_path / "test_operator_actions.jsonl"
    mgr = SafetyOperatorManager(audit_file_path=audit_file, max_records=50)
    yield mgr
    mgr.clear()


def test_operator_models_and_serialization():
    record = OperatorActionRecord(
        action_id="act_test_001",
        timestamp=time.time(),
        operator_id="OP-TEST-01",
        action_type=OperatorActionType.ACKNOWLEDGE,
        zone_id="zone_a",
        zone_name="Zone A - North Ingress",
        alert_ref="alert_101",
        status=IncidentLifecycleStatus.ACKNOWLEDGED,
        priority=OperatorPriority.MEDIUM,
        note="Verified visually via camera 1.",
        details={"camera_id": "cam_01"},
    )
    d = record.to_dict()
    assert d["action_id"] == "act_test_001"
    assert d["action_type"] == "ACKNOWLEDGE"
    assert d["status"] == "ACKNOWLEDGED"
    assert d["operator_id"] == "OP-TEST-01"
    assert d["zone_id"] == "zone_a"
    assert d["note"] == "Verified visually via camera 1."
    assert d["details"]["camera_id"] == "cam_01"


def test_acknowledge_alert(clean_operator_mgr):
    mgr = clean_operator_mgr
    rec = mgr.acknowledge_alert(
        alert_id="ew_alert_123",
        zone_id="zone_a",
        zone_name="Zone A - North Ingress",
        operator_id="OP-CHIEF-01",
        note="Confirmed crowd buildup at turnstiles.",
        priority=OperatorPriority.HIGH,
    )
    assert rec.action_id.startswith("act_")
    assert rec.action_type == OperatorActionType.ACKNOWLEDGE
    assert rec.status == IncidentLifecycleStatus.ACKNOWLEDGED
    assert rec.zone_id == "zone_a"
    assert "Confirmed crowd buildup" in rec.note

    # Check incident state
    inc = mgr.get_incident("zone_a")
    assert inc is not None
    assert inc.status == IncidentLifecycleStatus.ACKNOWLEDGED
    assert inc.priority == OperatorPriority.HIGH
    assert inc.alert_ref == "ew_alert_123"
    assert inc.action_count >= 1


def test_dispatch_team(clean_operator_mgr):
    mgr = clean_operator_mgr
    rec = mgr.dispatch_team(
        zone_id="zone_b",
        zone_name="Zone B - Main Concourse",
        team_name="Ground Stewards Unit Bravo",
        priority=OperatorPriority.URGENT,
        operator_id="OP-DUTY-02",
        note="Deploy stewards to clear central concourse corridor.",
    )
    assert rec.action_type == OperatorActionType.DISPATCH_GROUND_TEAM
    assert rec.status == IncidentLifecycleStatus.DISPATCHED
    assert rec.priority == OperatorPriority.URGENT
    assert rec.details["team_name"] == "Ground Stewards Unit Bravo"

    inc = mgr.get_incident("zone_b")
    assert inc.status == IncidentLifecycleStatus.DISPATCHED
    assert inc.assigned_team == "Ground Stewards Unit Bravo"
    assert inc.priority == OperatorPriority.URGENT


def test_redirect_crowd(clean_operator_mgr):
    mgr = clean_operator_mgr
    rec = mgr.redirect_crowd(
        zone_id="zone_b",
        target_zone_id="zone_c",
        zone_name="Zone B - Main Concourse",
        target_zone_name="Zone C - Plaza Gathering",
        instruction="Hold ingress turnstiles 3-5 and direct flow via east concourse",
        operator_id="OP-CHIEF-01",
    )
    assert rec.action_type == OperatorActionType.REDIRECT_CROWD
    assert rec.status == IncidentLifecycleStatus.REDIRECTING
    assert rec.details["source_zone"] == "zone_b"
    assert rec.details["target_zone"] == "zone_c"

    inc = mgr.get_incident("zone_b")
    assert inc.status == IncidentLifecycleStatus.REDIRECTING
    assert "Hold ingress turnstiles" in inc.crowd_instruction


def test_open_egress(clean_operator_mgr):
    mgr = clean_operator_mgr
    rec = mgr.open_egress(
        zone_id="zone_d",
        egress_gate="Emergency Auxiliary Egress East-3",
        zone_name="Zone D - South Egress",
        operator_id="OP-CHIEF-01",
        note="Auxiliary egress unlatched to relieve 3.8 p/m2 accumulation.",
    )
    assert rec.action_type == OperatorActionType.OPEN_EGRESS
    assert rec.status == IncidentLifecycleStatus.REDIRECTING
    assert rec.priority == OperatorPriority.URGENT

    inc = mgr.get_incident("zone_d")
    assert "East-3" in inc.egress_status


def test_adjust_flow(clean_operator_mgr):
    mgr = clean_operator_mgr
    rec = mgr.adjust_flow(
        zone_id="zone_a",
        max_inflow_rate=8.5,
        flow_direction="OUTFLOW_PRIORITY",
        zone_name="Zone A - North Ingress",
        operator_id="OP-CHIEF-01",
    )
    assert rec.action_type == OperatorActionType.ADJUST_FLOW
    assert rec.details["max_inflow_rate"] == 8.5
    assert rec.details["flow_direction"] == "OUTFLOW_PRIORITY"


def test_incident_lifecycle_and_escalate_resolve(clean_operator_mgr):
    mgr = clean_operator_mgr

    # 1. Open / Acknowledge
    mgr.acknowledge_alert("alert_999", "zone_c", "Zone C - Plaza")
    assert mgr.get_incident("zone_c").status == IncidentLifecycleStatus.ACKNOWLEDGED

    # 2. Dispatch
    mgr.dispatch_team("zone_c", "Zone C - Plaza", "Steward Team Alpha")
    assert mgr.get_incident("zone_c").status == IncidentLifecycleStatus.DISPATCHED

    # 3. Escalate
    mgr.escalate_incident("alert_999", "zone_c", OperatorPriority.URGENT, note="Density exceeded 4.2 p/m2.")
    assert mgr.get_incident("zone_c").status == IncidentLifecycleStatus.ESCALATED
    assert mgr.get_incident("zone_c").severity == "CRITICAL"

    # 4. Resolve
    mgr.resolve_incident("alert_999", "zone_c", note="Flow normalized below 1.5 p/m2.")
    assert mgr.get_incident("zone_c").status == IncidentLifecycleStatus.RESOLVED
    assert mgr.get_incident("zone_c").severity == "SAFE"


def test_audit_trail_and_json_export(clean_operator_mgr):
    mgr = clean_operator_mgr
    mgr.acknowledge_alert("al_1", "zone_a", "Zone A")
    mgr.dispatch_team("zone_a", "Zone A", "Steward Alpha")
    mgr.log_note("zone_a", "Shift observation: lighting nominal.")

    actions = mgr.get_recent_actions(limit=10)
    assert len(actions) == 3
    # Check reverse chronological order
    assert actions[0].action_type == OperatorActionType.LOG_NOTE
    assert actions[1].action_type == OperatorActionType.DISPATCH_GROUND_TEAM
    assert actions[2].action_type == OperatorActionType.ACKNOWLEDGE

    # JSON Export
    json_str = mgr.export_audit_log_json()
    data = json.loads(json_str)
    assert len(data) == 3
    assert data[0]["action_type"] == "LOG_NOTE"


def test_summary_metrics(clean_operator_mgr):
    mgr = clean_operator_mgr
    mgr.acknowledge_alert("al_1", "zone_a")
    mgr.dispatch_team("zone_b", "Zone B")
    mgr.redirect_crowd("zone_b", "zone_c")
    mgr.open_egress("zone_d")
    mgr.resolve_incident("al_1", "zone_a")

    summary = mgr.get_summary()
    assert summary["total_actions"] == 5
    assert summary["acknowledged_count"] == 1
    assert summary["dispatched_count"] == 1
    assert summary["redirected_count"] == 2  # redirect_crowd + open_egress
    assert summary["resolved_count"] == 1


def test_bounded_memory_footprint(clean_operator_mgr):
    # Enforces max_records bound to protect ~145 MB memory limit
    mgr = clean_operator_mgr
    for i in range(100):
        mgr.log_note("zone_a", f"Test note {i}")

    # Maximum is configured to 50 in fixture
    assert len(mgr._records) == 50
    recent = mgr.get_recent_actions(limit=100)
    assert len(recent) == 50
    assert "Test note 99" in recent[0].note

