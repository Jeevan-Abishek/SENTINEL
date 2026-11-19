"""
Fast, Explainable, and Resilient Crowd Risk Engine for SENTINEL.
Calculates multi-factor weighted risk scores (0-100), classifies safety tiers
(SAFE, WARNING, CRITICAL) with temporal smoothing and hysteresis, and generates
transparent, human-readable explanations and top driver decompositions.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
from loguru import logger

from app.risk.models import (
    RiskLevel,
    RiskFactor,
    RiskAssessment,
    RiskConfig,
)
from app.analytics.models import (
    DensityResult,
    OccupancyResult,
    MovementResult,
    FlowResult,
    BottleneckResult,
    ZoneAnalyticsSummary,
    SceneAnalyticsSummary,
)


class RiskEngine:
    """
    High-Performance Explainable Risk Assessment Engine.
    Converts 5-factor crowd dynamics into transparent, actionable risk assessments.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        if config is None:
            config = RiskConfig()
        self.config = config
        self.previous_scores: Dict[str, float] = {}
        self.previous_levels: Dict[str, RiskLevel] = {}
        self.deescalation_counters: Dict[str, int] = {}
        self.escalation_counters: Dict[str, int] = {}

    def get_normalized_weights(self) -> Dict[str, float]:
        """Safely retrieve normalized component weights summing to 1.0."""
        return self.config.normalized_weights()

    def assess(
        self,
        zone_id: str,
        timestamp: float,
        density_result: Optional[Union[DensityResult, Any]] = None,
        occupancy_result: Optional[Union[OccupancyResult, Any]] = None,
        movement_result: Optional[Union[MovementResult, Any]] = None,
        flow_result: Optional[Union[FlowResult, Any]] = None,
        bottleneck_result: Optional[Union[BottleneckResult, Any]] = None,
    ) -> RiskAssessment:
        """
        Assess risk for a surveillance zone across all 5 core metrics plus predictive trends.
        Returns a structured, explainable RiskAssessment with component contributions.
        """
        weights = self.get_normalized_weights()
        factors: List[RiskFactor] = []
        contributions: Dict[str, float] = {}
        lead_time_min: Optional[float] = None
        is_early_warning: bool = False

        # 1. Density Component (0 - 100)
        if density_result is not None:
            max_dens = getattr(density_result, "max_density", getattr(density_result, "max_density_p_sqm", 0.0))
            # Quadratic scaling for high density crowd compression
            base_dens_score = min(100.0, (float(max_dens) ** 1.5) * 8.0)
            if hasattr(density_result, "risk_score") and getattr(density_result, "risk_score", 0.0) > 0:
                density_score = max(base_dens_score, float(density_result.risk_score))
            else:
                density_score = base_dens_score

            w_dens = weights["density"]
            dens_contrib = round(w_dens * density_score, 2)
            contributions["density"] = dens_contrib
            factors.append(
                RiskFactor(
                    name="density",
                    value=float(round(max_dens, 2)),
                    weight=w_dens,
                    score=float(round(density_score, 2)),
                    threshold_warning=getattr(self.config, "warning_density", 3.0),
                    threshold_critical=getattr(self.config, "critical_density", 5.0),
                    contribution=dens_contrib,
                    normalized_score=min(1.0, max(0.0, round(density_score / 100.0, 4))),
                )
            )

        # 2. Occupancy Component (0 - 100)
        if occupancy_result is not None:
            occ_ratio = getattr(occupancy_result, "occupancy_ratio", 0.0)
            base_occ_score = min(100.0, float(occ_ratio) * 100.0)
            if hasattr(occupancy_result, "risk_score") and getattr(occupancy_result, "risk_score", 0.0) > 0:
                occ_score = max(base_occ_score, float(occupancy_result.risk_score))
            else:
                occ_score = base_occ_score

            w_occ = weights["occupancy"]
            occ_contrib = round(w_occ * occ_score, 2)
            contributions["occupancy"] = occ_contrib
            factors.append(
                RiskFactor(
                    name="occupancy",
                    value=float(round(occ_ratio, 3)),
                    weight=w_occ,
                    score=float(round(occ_score, 2)),
                    threshold_warning=self.config.occupancy_warning_threshold,
                    threshold_critical=self.config.occupancy_critical_threshold,
                    contribution=occ_contrib,
                    normalized_score=min(1.0, max(0.0, round(occ_score / 100.0, 4))),
                )
            )

        # 3. Movement & Heading Field Component (0 - 100)
        if movement_result is not None:
            avg_spd = getattr(movement_result, "avg_speed", getattr(movement_result, "avg_speed_mps", 0.0))
            if avg_spd > 0:
                base_mov_score = min(100.0, 100.0 / (float(avg_spd) + 0.5))
            else:
                base_mov_score = 100.0

            if hasattr(movement_result, "risk_score") and getattr(movement_result, "risk_score", 0.0) > 0:
                mov_score = max(base_mov_score, float(movement_result.risk_score))
            else:
                mov_score = base_mov_score

            w_mov = weights["movement"]
            mov_contrib = round(w_mov * mov_score, 2)
            contributions["movement"] = mov_contrib
            factors.append(
                RiskFactor(
                    name="movement",
                    value=float(round(avg_spd, 2)),
                    weight=w_mov,
                    score=float(round(mov_score, 2)),
                    threshold_warning=1.0,
                    threshold_critical=0.5,
                    contribution=mov_contrib,
                    normalized_score=min(1.0, max(0.0, round(mov_score / 100.0, 4))),
                )
            )

        # 4. Flow & Net Accumulation Component (0 - 100)
        if flow_result is not None:
            fl_rate = getattr(flow_result, "flow_rate", 0.0)
            base_flow_score = min(100.0, float(fl_rate) * 5.0)
            if hasattr(flow_result, "risk_score") and getattr(flow_result, "risk_score", 0.0) > 0:
                flow_score = max(base_flow_score, float(flow_result.risk_score))
            else:
                flow_score = base_flow_score

            w_fl = weights["flow"]
            flow_contrib = round(w_fl * flow_score, 2)
            contributions["flow"] = flow_contrib
            factors.append(
                RiskFactor(
                    name="flow",
                    value=float(round(fl_rate, 2)),
                    weight=w_fl,
                    score=float(round(flow_score, 2)),
                    threshold_warning=5.0,
                    threshold_critical=10.0,
                    contribution=flow_contrib,
                    normalized_score=min(1.0, max(0.0, round(flow_score / 100.0, 4))),
                )
            )

        # 5. Bottleneck & Choke Point Component (0 - 100)
        if bottleneck_result is not None:
            bn_score_raw = getattr(bottleneck_result, "bottleneck_score", 0.0)
            base_bn_score = min(100.0, float(bn_score_raw) * 100.0)
            if hasattr(bottleneck_result, "risk_score") and getattr(bottleneck_result, "risk_score", 0.0) > 0:
                bn_score = max(base_bn_score, float(bottleneck_result.risk_score))
            else:
                bn_score = base_bn_score

            w_bn = weights["bottleneck"]
            bn_contrib = round(w_bn * bn_score, 2)
            contributions["bottleneck"] = bn_contrib
            factors.append(
                RiskFactor(
                    name="bottleneck",
                    value=float(round(bn_score_raw, 3)),
                    weight=w_bn,
                    score=float(round(bn_score, 2)),
                    threshold_warning=0.50,
                    threshold_critical=0.80,
                    contribution=bn_contrib,
                    normalized_score=min(1.0, max(0.0, round(bn_score / 100.0, 4))),
                )
            )

            # Check lead time
            bn_lead = getattr(bottleneck_result, "lead_time_min", getattr(bottleneck_result, "early_warning_lead_min", None))
            if bn_lead is not None and bn_lead <= self.config.early_warning_lead_minutes:
                lead_time_min = bn_lead
                is_early_warning = True

        # 6. Predictive 0-5 Minute Forward Horizon Factor
        proj_dens = getattr(density_result, "projected_density_5min", getattr(density_result, "projected_density_5m", None)) if density_result else None
        dens_lead = getattr(density_result, "lead_time_to_critical_min", getattr(density_result, "lead_time_min", None)) if density_result else None
        if dens_lead is not None and (lead_time_min is None or dens_lead < lead_time_min):
            lead_time_min = dens_lead

        if getattr(self.config, "enable_predictive", True) and proj_dens is not None and proj_dens > 0:
            pred_score = min(100.0, (float(proj_dens) ** 1.5) * 8.0)

            # Compound dynamics: deceleration + counter-flow + accumulation
            if movement_result and getattr(movement_result, "speed_trend", "") == "slowing":
                pred_score = min(100.0, pred_score * 1.25)
            if movement_result and getattr(movement_result, "counter_flow_index", 0.0) > 0.20:
                pred_score = min(100.0, pred_score + float(movement_result.counter_flow_index) * 15.0)
            if flow_result and getattr(flow_result, "net_accumulation_rate_ppm", 0.0) > 12.0:
                pred_score = min(100.0, pred_score + 10.0)

            pred_weight = self.config.predictive_weight
            pred_contrib = round(pred_weight * pred_score, 2)
            contributions["predictive_5min"] = pred_contrib
            factors.append(
                RiskFactor(
                    name="predictive_5min",
                    value=float(round(proj_dens, 2)),
                    weight=pred_weight,
                    score=float(round(pred_score, 1)),
                    threshold_warning=3.0,
                    threshold_critical=5.0,
                    contribution=pred_contrib,
                    normalized_score=min(1.0, max(0.0, round(pred_score / 100.0, 4))),
                )
            )
            if pred_score >= self.config.warning_threshold or proj_dens >= 3.0:
                is_early_warning = True

        # Compute Raw Weighted Score
        # If not all factors are present, normalize weights across active core factors
        # so the risk score remains properly calibrated to 0-100 even with partial inputs.
        if factors:
            core_factors = [f for f in factors if f.name != "predictive_5min"]
            total_active_core_weight = sum(f.weight for f in core_factors)
            if 0.0 < total_active_core_weight < 0.99 or total_active_core_weight > 1.01:
                scale_factor = 1.0 / total_active_core_weight
                for f in core_factors:
                    f.weight = round(f.weight * scale_factor, 4)
                    f.contribution = round(f.weight * f.score, 2)
                    contributions[f.name] = f.contribution

            raw_score = sum(f.weight * f.score for f in factors)
            raw_score = min(100.0, max(0.0, raw_score))
        else:
            raw_score = 0.0

        # Temporal Smoothing: S_t = alpha * S_{t-1} + (1 - alpha) * R_t
        alpha = max(0.0, min(0.9, float(self.config.temporal_smoothing)))
        if zone_id in self.previous_scores:
            smoothed_score = (alpha * self.previous_scores[zone_id]) + ((1.0 - alpha) * raw_score)
        else:
            smoothed_score = raw_score

        # Hysteresis Safety Tier Classification & Consecutive Confirmation
        prev_level = self.previous_levels.get(zone_id, RiskLevel.SAFE)
        h_buf = max(0.0, getattr(self.config, "hysteresis_buffer", 3.0))
        req_deesc = max(1, getattr(self.config, "consecutive_deescalation_frames", 1))

        from app.risk.rules import classify_risk_state
        candidate_level = classify_risk_state(
            risk_score=smoothed_score,
            previous_state=prev_level,
            warning_threshold=self.config.warning_threshold,
            critical_threshold=self.config.critical_threshold,
            hysteresis_buffer=h_buf,
            is_early_warning=is_early_warning,
            consecutive_confirmations=self.deescalation_counters.get(zone_id, 0) + 1,
            required_deescalation_frames=req_deesc,
        )

        # Track consecutive confirmations for de-escalation
        if prev_level == RiskLevel.CRITICAL and candidate_level != RiskLevel.CRITICAL:
            cur_count = self.deescalation_counters.get(zone_id, 0) + 1
            self.deescalation_counters[zone_id] = cur_count
            if cur_count < req_deesc:
                risk_level = RiskLevel.CRITICAL
            else:
                risk_level = candidate_level
                self.deescalation_counters[zone_id] = 0
        elif prev_level == RiskLevel.WARNING and candidate_level == RiskLevel.SAFE:
            cur_count = self.deescalation_counters.get(zone_id, 0) + 1
            self.deescalation_counters[zone_id] = cur_count
            if cur_count < req_deesc:
                risk_level = RiskLevel.WARNING
            else:
                risk_level = candidate_level
                self.deescalation_counters[zone_id] = 0
        else:
            self.deescalation_counters[zone_id] = 0
            risk_level = candidate_level

        # Preemptive Early Warning override: If predictive factor is warning and current is safe, escalate
        if risk_level == RiskLevel.SAFE and is_early_warning:
            pred_f = next((f for f in factors if f.name == "predictive_5min"), None)
            if pred_f and (pred_f.score >= self.config.warning_threshold or pred_f.value >= 3.0):
                risk_level = RiskLevel.WARNING
                smoothed_score = max(smoothed_score, self.config.warning_threshold)

        # Update persistent history
        self.previous_scores[zone_id] = float(round(smoothed_score, 2))
        self.previous_levels[zone_id] = risk_level

        # Extract Top Drivers
        top_drivers = self._extract_top_drivers(factors)

        # Generate Explainable Narrative
        explanation = self._generate_explanation(
            zone_id=zone_id,
            risk_level=risk_level,
            risk_score=smoothed_score,
            factors=factors,
            top_drivers=top_drivers,
            is_early_warning=is_early_warning,
            lead_time_min=lead_time_min,
        )

        # Generate Recommended Actions
        recommended_actions = self._generate_actions(risk_level, factors, is_early_warning)

        return RiskAssessment(
            zone_id=zone_id,
            timestamp=timestamp,
            risk_level=risk_level,
            risk_score=float(round(smoothed_score, 2)),
            factors=factors,
            recommended_actions=recommended_actions,
            component_contributions=contributions,
            top_drivers=top_drivers,
            explanation=explanation,
            lead_time_min=lead_time_min,
            is_early_warning=is_early_warning,
        )

    def assess_summary(self, summary: Union[ZoneAnalyticsSummary, SceneAnalyticsSummary, Any], zones: Optional[Dict[str, Any]] = None) -> RiskAssessment:
        """Convenience method directly evaluating a ZoneAnalyticsSummary or SceneAnalyticsSummary from Crowd Analytics Engine."""
        zid = getattr(summary, "zone_id", "scene_overall")
        return self.assess(
            zone_id=zid,
            timestamp=summary.timestamp,
            density_result=summary.density,
            occupancy_result=summary.occupancy,
            movement_result=summary.movement,
            flow_result=summary.flow,
            bottleneck_result=summary.bottleneck,
        )

    def assess_scene(self, scene_summary: SceneAnalyticsSummary, zones: Optional[Dict[str, Any]] = None) -> Dict[str, RiskAssessment]:
        """Assess all calibrated zones in the scene plus global scene summary."""
        assessments: Dict[str, RiskAssessment] = {}
        for zid, zsum in scene_summary.zones.items():
            assessments[zid] = self.assess_summary(zsum, zones=zones)

        # Also create overall scene assessment
        overall_assessment = self.assess(
            zone_id="scene_overall",
            timestamp=scene_summary.timestamp,
            density_result=scene_summary.density,
            occupancy_result=scene_summary.occupancy,
            movement_result=scene_summary.movement,
            flow_result=scene_summary.flow,
            bottleneck_result=scene_summary.bottleneck,
        )
        assessments["scene_overall"] = overall_assessment
        return assessments

    def _extract_top_drivers(self, factors: List[RiskFactor]) -> List[str]:
        """Rank and return the top 1-3 dominant factors driving the risk score."""
        if not factors:
            return []
        # Sort by contribution (weight * score) descending
        sorted_factors = sorted(factors, key=lambda f: (f.weight * f.score), reverse=True)
        # Select top factors with significant contribution (> 5.0) or top 2
        drivers = [f.name for f in sorted_factors if (f.weight * f.score) >= 5.0]
        if not drivers:
            drivers = [f.name for f in sorted_factors[:2]]
        return drivers[:3]

    def _generate_explanation(
        self,
        zone_id: str,
        risk_level: RiskLevel,
        risk_score: float,
        factors: List[RiskFactor],
        top_drivers: List[str],
        is_early_warning: bool = False,
        lead_time_min: Optional[float] = None,
    ) -> str:
        """Generate concise, professional human-readable explanation of risk level and root causes."""
        if not factors:
            return f"Zone {zone_id} is SAFE with zero active risk factors reported."

        # Map factors
        f_map = {f.name: f for f in factors}
        lead_str = f" (~{lead_time_min:.1f} min advance lead)" if lead_time_min else ""

        if is_early_warning and risk_level == RiskLevel.WARNING:
            pred_f = f_map.get("predictive_5min")
            val_str = f"projected density {pred_f.value:.2f} p/m²" if pred_f else "surging influx"
            return (
                f"⏳ PREEMPTIVE EARLY WARNING in {zone_id} (Score: {risk_score:.1f}/100){lead_str}: "
                f"Current scene is nominal, but trend analysis indicates rapid congestion build-up ({val_str}). "
                f"Early steward intervention recommended before threshold breach."
            )

        if risk_level == RiskLevel.CRITICAL:
            driver_phrases = []
            for d in top_drivers:
                f = f_map.get(d)
                if not f:
                    continue
                if d == "density":
                    driver_phrases.append(f"severe physical density ({f.value:.1f} p/m², score: {f.score:.0f})")
                elif d == "occupancy":
                    driver_phrases.append(f"critical occupancy ratio ({f.value*100:.0f}%, score: {f.score:.0f})")
                elif d == "movement":
                    driver_phrases.append(f"stagnant crowd velocity ({f.value:.2f} m/s, score: {f.score:.0f})")
                elif d == "flow":
                    driver_phrases.append(f"heavy inflow rate ({f.value:.1f} p/min, score: {f.score:.0f})")
                elif d == "bottleneck":
                    driver_phrases.append(f"high choke compression (index: {f.value:.2f}, score: {f.score:.0f})")

            drivers_text = "; ".join(driver_phrases) if driver_phrases else "multiple compound stressors"
            return (
                f"🚨 CRITICAL risk detected in {zone_id} (Score: {risk_score:.1f}/100): "
                f"Driven by {drivers_text}. Immediate crowd dispersion and steward deployment required."
            )

        if risk_level == RiskLevel.WARNING:
            driver_phrases = []
            for d in top_drivers:
                f = f_map.get(d)
                if not f:
                    continue
                if d == "density":
                    driver_phrases.append(f"elevated density ({f.value:.1f} p/m²)")
                elif d == "occupancy":
                    driver_phrases.append(f"occupancy at {f.value*100:.0f}%")
                elif d == "movement":
                    driver_phrases.append(f"slowing forward speed ({f.value:.2f} m/s)")
                elif d == "bottleneck":
                    driver_phrases.append(f"turnstile compression ({f.value:.2f})")
                elif d == "flow":
                    driver_phrases.append(f"flow accumulation ({f.value:.1f} p/min)")

            drivers_text = ", ".join(driver_phrases) if driver_phrases else "approaching threshold indicators"
            return (
                f"⚠️ WARNING condition in {zone_id} (Score: {risk_score:.1f}/100){lead_str}: "
                f"Elevated stress due to {drivers_text}. Active operator monitoring and standby controls recommended."
            )

        # SAFE
        return (
            f"✅ SAFE nominal conditions in {zone_id} (Score: {risk_score:.1f}/100): "
            f"All crowd density, occupancy, movement speeds, and gate flows are well within safe thresholds."
        )

    def _generate_actions(
        self,
        risk_level: RiskLevel,
        factors: List[RiskFactor],
        is_early_warning: bool = False,
    ) -> List[str]:
        """Generate clear, context-aware operational recommendations for ground safety stewards."""
        actions: List[str] = []

        if risk_level == RiskLevel.CRITICAL:
            actions.append("Immediate evacuation / diversion of zone recommended")
            actions.append("Deploy security personnel to manage crowd")
            actions.append("Alert emergency services")
        elif risk_level == RiskLevel.WARNING:
            actions.append("Increase monitoring frequency")
            actions.append("Prepare crowd control measures")
            actions.append("Notify zone operators")

        if is_early_warning:
            actions.append("Preemptive Early Warning (~4-5 min lead): Regulate turnstiles and open auxiliary routes")

        # Factor-specific actions
        for factor in factors:
            if factor.name == "density" and factor.score > 70:
                actions.append("Redirect incoming crowd flow")
            elif factor.name == "occupancy" and factor.score > 80:
                actions.append("Limit entry to zone")
            elif factor.name == "bottleneck" and factor.score > 60:
                actions.append("Open alternative routes")
            elif factor.name == "predictive_5min" and factor.score >= 55:
                actions.append("Preemptive Early Warning (~4-5 min lead): Regulate turnstiles and open auxiliary routes")

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                deduped.append(a)
        return deduped

    def reset_state(self, zone_id: Optional[str] = None):
        """Reset temporal smoothing memory and confirmation counters for testing or calibration updates."""
        if zone_id:
            self.previous_scores.pop(zone_id, None)
            self.previous_levels.pop(zone_id, None)
            self.deescalation_counters.pop(zone_id, None)
            self.escalation_counters.pop(zone_id, None)
        else:
            self.previous_scores.clear()
            self.previous_levels.clear()
            self.deescalation_counters.clear()
            self.escalation_counters.clear()


class RiskEngineWrapper:
    """Convenience wrapper maintaining backward-compatibility with pipeline callers."""

    def __init__(self, config: Optional[RiskConfig] = None):
        self.engine = RiskEngine(config)

    def assess(
        self,
        zone_id: str,
        timestamp: float,
        density_result: Optional[Union[DensityResult, Any]] = None,
        occupancy_result: Optional[Union[OccupancyResult, Any]] = None,
        movement_result: Optional[Union[MovementResult, Any]] = None,
        flow_result: Optional[Union[FlowResult, Any]] = None,
        bottleneck_result: Optional[Union[BottleneckResult, Any]] = None,
    ) -> RiskAssessment:
        return self.engine.assess(
            zone_id=zone_id,
            timestamp=timestamp,
            density_result=density_result,
            occupancy_result=occupancy_result,
            movement_result=movement_result,
            flow_result=flow_result,
            bottleneck_result=bottleneck_result,
        )


__all__ = ["RiskEngine", "RiskEngineWrapper"]
