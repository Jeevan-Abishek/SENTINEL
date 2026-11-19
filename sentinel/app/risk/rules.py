"""
Risk Rules and Alert Evaluation Module for SENTINEL.
Defines transparent rule definitions, condition evaluators, and early warning criteria.
"""

from typing import List, Optional, Dict, Any
from app.risk.models import (
    RiskLevel,
    RiskAssessment,
    RiskFactor,
    AlertRule,
    Alert,
    AlertSeverity,
    AlertStatus,
)


def create_critical_rule(
    rule_id: str = "rule_critical_overall",
    name: str = "Critical Overall Crowd Risk",
    zone_id: Optional[str] = None,
    cooldown_seconds: int = 30,
) -> AlertRule:
    """Create a standardized rule for critical risk level trigger."""
    return AlertRule(
        id=rule_id,
        name=name,
        zone_id=zone_id,
        condition="risk_level_critical",
        risk_level=RiskLevel.CRITICAL,
        cooldown_seconds=cooldown_seconds,
        enabled=True,
    )


def create_warning_rule(
    rule_id: str = "rule_warning_overall",
    name: str = "Warning Crowd Risk Condition",
    zone_id: Optional[str] = None,
    cooldown_seconds: int = 45,
) -> AlertRule:
    """Create a standardized rule for warning risk level trigger."""
    return AlertRule(
        id=rule_id,
        name=name,
        zone_id=zone_id,
        condition="risk_level_warning",
        risk_level=RiskLevel.WARNING,
        cooldown_seconds=cooldown_seconds,
        enabled=True,
    )


def create_early_warning_rule(
    rule_id: str = "rule_pred_warning",
    name: str = "Predictive 5-Minute Advance Early Warning",
    zone_id: Optional[str] = None,
    cooldown_seconds: int = 60,
) -> AlertRule:
    """Create a rule triggering on projected congestion build-up (0-5 min lead window)."""
    return AlertRule(
        id=rule_id,
        name=name,
        zone_id=zone_id,
        condition="predictive_early_warning",
        risk_level=RiskLevel.WARNING,
        cooldown_seconds=cooldown_seconds,
        enabled=True,
        metadata={"advance_lead_time": 5.0},
    )


def create_choke_point_rule(
    rule_id: str = "rule_choke_bottleneck",
    name: str = "Severe Turnstile / Choke Point Congestion",
    zone_id: Optional[str] = None,
    cooldown_seconds: int = 45,
) -> AlertRule:
    """Create a rule triggering on high bottleneck compression index."""
    return AlertRule(
        id=rule_id,
        name=name,
        zone_id=zone_id,
        condition="bottleneck_compression",
        risk_level=RiskLevel.WARNING,
        cooldown_seconds=cooldown_seconds,
        enabled=True,
    )


def create_counter_flow_rule(
    rule_id: str = "rule_flow_conflict",
    name: str = "Counter-Flow Collision Conflict",
    zone_id: Optional[str] = None,
    cooldown_seconds: int = 60,
) -> AlertRule:
    """Create a rule triggering on opposing crowd headings."""
    return AlertRule(
        id=rule_id,
        name=name,
        zone_id=zone_id,
        condition="counter_flow_conflict",
        risk_level=RiskLevel.WARNING,
        cooldown_seconds=cooldown_seconds,
        enabled=True,
    )


def get_default_ruleset() -> List[AlertRule]:
    """Return comprehensive public-safety default alert ruleset."""
    return [
        create_critical_rule(),
        create_warning_rule(),
        create_early_warning_rule(),
        create_choke_point_rule(),
        create_counter_flow_rule(),
        AlertRule(
            id="rule_score_70",
            name="High Aggregate Risk Score",
            condition="risk_score_above_70",
            risk_level=RiskLevel.CRITICAL,
            cooldown_seconds=30,
            enabled=True,
        ),
        AlertRule(
            id="rule_score_50",
            name="Elevated Aggregate Risk Score",
            condition="risk_score_above_50",
            risk_level=RiskLevel.WARNING,
            cooldown_seconds=45,
            enabled=True,
        ),
        AlertRule(
            id="rule_factor_critical",
            name="Single Factor Critical Outlier",
            condition="any_factor_critical",
            risk_level=RiskLevel.CRITICAL,
            cooldown_seconds=30,
            enabled=True,
        ),
    ]


def evaluate_rule_condition(rule: AlertRule, assessment: RiskAssessment) -> bool:
    """
    Evaluate boolean rule condition against a structured RiskAssessment.
    Provides explainable triggers for reactive and predictive alerts.
    """
    cond = rule.condition.lower().strip()

    if cond == "risk_level_critical":
        return assessment.risk_level == RiskLevel.CRITICAL
    elif cond == "risk_level_warning":
        return assessment.risk_level in [RiskLevel.WARNING, RiskLevel.CRITICAL]
    elif cond == "risk_score_above_70":
        return assessment.risk_score >= 70.0
    elif cond == "risk_score_above_50":
        return assessment.risk_score >= 50.0
    elif cond == "any_factor_critical":
        return any(f.score >= 80.0 for f in assessment.factors)
    elif cond == "predictive_early_warning":
        pred_f = next((f for f in assessment.factors if f.name == "predictive_5min"), None)
        return (
            getattr(assessment, "is_early_warning", False)
            or (pred_f is not None and (pred_f.score >= 50.0 or pred_f.value >= 3.0))
        )
    elif cond == "bottleneck_compression":
        bn_f = next((f for f in assessment.factors if f.name == "bottleneck"), None)
        return bn_f is not None and bn_f.score >= 70.0
    elif cond == "counter_flow_conflict":
        fl_f = next((f for f in assessment.factors if f.name in ["flow", "movement"]), None)
        return fl_f is not None and fl_f.score >= 65.0
    return False


def generate_rule_message(rule: AlertRule, assessment: RiskAssessment) -> str:
    """Generate clear, explainable operator alert message for an activated rule."""
    cond = rule.condition.lower().strip()
    zid = assessment.zone_id
    score = assessment.risk_score

    if cond == "predictive_early_warning":
        pred_f = next((f for f in assessment.factors if f.name == "predictive_5min"), None)
        val = pred_f.value if pred_f else 0.0
        lead = getattr(assessment, "lead_time_min", 4.2) or 4.2
        return (
            f"⏳ EARLY WARNING (~{lead:.1f} min lead): Projected congestion build-up in zone {zid} "
            f"(projected density: {val:.2f} p/m²). Preemptive crowd regulation advised."
        )
    elif cond == "risk_level_critical":
        return f"CRITICAL risk detected in zone {zid}: Risk score {score:.1f}/100. Immediate intervention advised."
    elif cond == "risk_level_warning":
        pred_f = next((f for f in assessment.factors if f.name == "predictive_5min"), None)
        if pred_f and pred_f.score >= 55:
            return (
                f"WARNING (~4-5 min advance lead) in zone {zid}: "
                f"Risk score {score:.1f} | Projected density {pred_f.value:.2f} p/m²"
            )
        return f"WARNING risk detected in zone {zid}: Risk score {score:.1f}/100. Monitor actively."
    elif cond == "risk_score_above_70":
        return f"High risk in zone {zid}: Score {score:.1f}/100"
    elif cond == "risk_score_above_50":
        return f"Elevated risk in zone {zid}: Score {score:.1f}/100"
    elif cond == "any_factor_critical":
        crit_factors = [f.name for f in assessment.factors if f.score >= 80.0]
        return f"Critical factors in zone {zid}: {', '.join(crit_factors)}"
    elif cond == "bottleneck_compression":
        return f"Turnstile choke bottleneck alert in zone {zid}: Compression index elevated."
    elif cond == "counter_flow_conflict":
        return f"Opposing counter-flow conflict detected in zone {zid}: Cross-stream turbulence forming."

    return f"Alert triggered in zone {zid}: {rule.name} (Risk score {score:.1f})"


# -----------------------------------------------------------------------------
# Safe / Warning / Critical State Presentation & Classification
# -----------------------------------------------------------------------------

STATE_PRESENTATION_METADATA: Dict[str, Dict[str, Any]] = {
    "safe": {
        "state": RiskLevel.SAFE,
        "label": "SAFE",
        "color": "#2E7D32",          # Eco-Friendly Forest Green
        "bg_color": "#E8F5E9",       # Light Green
        "border_color": "#2E7D32",
        "pill_class": "status-safe",
        "badge_text": "🟢 SAFE",
        "icon": "🟢",
        "priority": 0,
        "description": "Normal crowd conditions, free movement, nominal density (< 1.5 p/m²)",
    },
    "warning": {
        "state": RiskLevel.WARNING,
        "label": "WARNING",
        "color": "#E65100",          # Eco-Amber / Warning Orange
        "bg_color": "#FFF3E0",       # Light Amber
        "border_color": "#E65100",
        "pill_class": "status-warning",
        "badge_text": "⚠️ WARNING",
        "icon": "⚠️",
        "priority": 1,
        "description": "Elevated density / inflow surplus or 0-5 min predictive congestion surge",
    },
    "critical": {
        "state": RiskLevel.CRITICAL,
        "label": "CRITICAL",
        "color": "#C62828",          # High-Contrast Crimson
        "bg_color": "#FFEBEE",       # Light Crimson
        "border_color": "#C62828",
        "pill_class": "status-critical",
        "badge_text": "🚨 CRITICAL",
        "icon": "🚨",
        "priority": 2,
        "description": "Severe physical crowd compression, gate bottleneck, or safety breach",
    },
}


def get_state_presentation(level: Any) -> Dict[str, Any]:
    """
    Retrieve standardized Eco-Safety presentation metadata for a risk level or state string.
    Guarantees non-null dictionary output with label, colors, pill_class, and icon.
    """
    if isinstance(level, RiskLevel):
        key = level.value.lower()
    else:
        key = str(level).lower().strip()
        if "crit" in key:
            key = "critical"
        elif "warn" in key:
            key = "warning"
        else:
            key = "safe"

    return STATE_PRESENTATION_METADATA.get(key, STATE_PRESENTATION_METADATA["safe"])


def format_state_pill_html(level: Any, extra_style: str = "") -> str:
    """Format a consistent HTML status pill badge."""
    meta = get_state_presentation(level)
    style = (
        f"background-color: {meta['bg_color']}; color: {meta['color']}; "
        f"border: 1px solid {meta['border_color']}; font-weight: 700; "
        f"padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; "
        f"display: inline-flex; align-items: center; gap: 4px; {extra_style}"
    )
    return f"<span class='status-pill {meta['pill_class']}' style='{style}'>{meta['icon']} {meta['label']}</span>"


def classify_risk_state(
    risk_score: float,
    previous_state: Optional[Any] = None,
    warning_threshold: float = 60.0,
    critical_threshold: float = 80.0,
    hysteresis_buffer: float = 3.0,
    is_early_warning: bool = False,
    consecutive_confirmations: int = 1,
    required_deescalation_frames: int = 1,
) -> RiskLevel:
    """
    Classify a continuous risk score (0-100) into canonical SAFE / WARNING / CRITICAL states
    with hysteresis buffer and optional confirmation gating to eliminate state flickering.
    """
    h_buf = max(0.0, float(hysteresis_buffer))
    score = float(risk_score)

    # Normalize previous state
    prev = None
    if previous_state:
        if isinstance(previous_state, RiskLevel):
            prev = previous_state
        else:
            s = str(previous_state).lower()
            if "crit" in s:
                prev = RiskLevel.CRITICAL
            elif "warn" in s:
                prev = RiskLevel.WARNING
            else:
                prev = RiskLevel.SAFE

    # 1. Previous state was CRITICAL
    if prev == RiskLevel.CRITICAL:
        # Step-down threshold requires dropping by at least hysteresis_buffer
        step_down_thresh = critical_threshold - h_buf
        if score < step_down_thresh:
            if consecutive_confirmations >= required_deescalation_frames:
                if score < (warning_threshold - h_buf):
                    return RiskLevel.SAFE
                return RiskLevel.WARNING
            return RiskLevel.CRITICAL
        return RiskLevel.CRITICAL

    # 2. Previous state was WARNING
    if prev == RiskLevel.WARNING:
        if score >= critical_threshold:
            return RiskLevel.CRITICAL
        step_down_thresh = warning_threshold - h_buf
        if score < step_down_thresh:
            if consecutive_confirmations >= required_deescalation_frames:
                return RiskLevel.SAFE
            return RiskLevel.WARNING
        return RiskLevel.WARNING

    # 3. Previous state was SAFE or uninitialized
    if score >= critical_threshold:
        return RiskLevel.CRITICAL
    if score >= warning_threshold or is_early_warning:
        return RiskLevel.WARNING
    return RiskLevel.SAFE


def filter_items_by_state(
    items: List[Dict[str, Any]],
    filter_state: str = "ALL",
    state_key: str = "risk_status",
) -> List[Dict[str, Any]]:
    """
    Filter list of dictionary objects (e.g. zones or alerts) by canonical state.
    Accepts filter strings such as 'ALL', 'CRITICAL', 'WARNING', 'SAFE' (and with emoji prefixes).
    """
    filt = filter_state.upper().strip()
    if filt in ["ALL", "", "ALL ZONES", "ALL STATES"]:
        return list(items)

    target = "CRITICAL" if "CRIT" in filt else ("WARNING" if "WARN" in filt else ("SAFE" if "SAFE" in filt else filt))
    matched = []
    for it in items:
        raw_val = str(it.get(state_key, it.get("risk_level", it.get("status", it.get("state", "SAFE"))))).upper().strip()
        if target in raw_val:
            matched.append(it)
    return matched


def summarize_state_counts(
    items: List[Dict[str, Any]],
    state_key: str = "risk_status",
) -> Dict[str, int]:
    """
    Summarize counts of items across total, safe, warning, and critical states.
    Returns: {'total': N, 'safe': S, 'warning': W, 'critical': C}
    """
    counts = {"total": len(items), "safe": 0, "warning": 0, "critical": 0}
    for it in items:
        raw_val = str(it.get(state_key, it.get("risk_level", it.get("status", it.get("state", "SAFE"))))).upper().strip()
        if "CRIT" in raw_val:
            counts["critical"] += 1
        elif "WARN" in raw_val:
            counts["warning"] += 1
        else:
            counts["safe"] += 1
    return counts


__all__ = [
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "STATE_PRESENTATION_METADATA",
    "create_critical_rule",
    "create_warning_rule",
    "create_early_warning_rule",
    "create_choke_point_rule",
    "create_counter_flow_rule",
    "get_default_ruleset",
    "evaluate_rule_condition",
    "generate_rule_message",
    "classify_risk_state",
    "get_state_presentation",
    "format_state_pill_html",
    "filter_items_by_state",
    "summarize_state_counts",
]

