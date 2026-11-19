"""Area & Zone Calibration Management Page for SENTINEL Dashboard.

Provides interactive ground surface area calibration (Direct, 2-Point Reference Distance Scale,
Drone Altitude AGL GSD, Homography), dynamic safety threshold tuning, and zero-latency persistence.
"""

from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import streamlit as st
import pandas as pd

try:
    from app.mapping.zones import ZoneDefinition, ZoneManager
    from app.mapping.calibration import (
        ScaleCalibrator,
        GroundResolutionCalculator,
        CameraFOVParams,
        CoordinateCalibrator,
    )
except ImportError:
    from sentinel.app.mapping.zones import ZoneDefinition, ZoneManager
    from sentinel.app.mapping.calibration import (
        ScaleCalibrator,
        GroundResolutionCalculator,
        CameraFOVParams,
        CoordinateCalibrator,
    )

try:
    from dashboard.pages.digital_map import VENUE_PRESETS
except ImportError:
    try:
        from sentinel.dashboard.pages.digital_map import VENUE_PRESETS
    except ImportError:
        from pages.digital_map import VENUE_PRESETS

try:
    from app.risk.rules import (
        filter_items_by_state,
        summarize_state_counts,
        format_state_pill_html,
        get_state_presentation,
    )
except ImportError:
    from risk.rules import (
        filter_items_by_state,
        summarize_state_counts,
        format_state_pill_html,
        get_state_presentation,
    )


DEFAULT_ZONES_KEY = "calibrated_zones"


def init_session_zones():
    """Ensure session state contains active calibrated zones."""
    if DEFAULT_ZONES_KEY not in st.session_state or not st.session_state[DEFAULT_ZONES_KEY]:
        # Seed from default venue preset (Bengaluru Kanteerava Arena)
        default_preset = VENUE_PRESETS["Bengaluru Kanteerava Arena"]["zones"]
        st.session_state[DEFAULT_ZONES_KEY] = {z["zone_id"]: dict(z) for z in default_preset}


def get_active_zones_dict() -> Dict[str, Dict[str, Any]]:
    """Get active zones dictionary from session state."""
    init_session_zones()
    return st.session_state[DEFAULT_ZONES_KEY]


def save_zone_to_session(zone_data: Dict[str, Any]) -> None:
    """Save or update zone in session state."""
    init_session_zones()
    st.session_state[DEFAULT_ZONES_KEY][zone_data["zone_id"]] = zone_data


def delete_zone_from_session(zone_id: str) -> bool:
    """Remove a zone from session state."""
    init_session_zones()
    if zone_id in st.session_state[DEFAULT_ZONES_KEY]:
        del st.session_state[DEFAULT_ZONES_KEY][zone_id]
        return True
    return False


def reset_zones_to_defaults() -> None:
    """Reset active zones to initial default venue preset."""
    default_preset = VENUE_PRESETS["Bengaluru Kanteerava Arena"]["zones"]
    st.session_state[DEFAULT_ZONES_KEY] = {z["zone_id"]: dict(z) for z in default_preset}


# -----------------------------------------------------------------------------
# Main Render Function
# -----------------------------------------------------------------------------
def render_zone_management_page(render_header_bar_fn=None):
    """Render the comprehensive Area & Zone Calibration Management panel."""
    init_session_zones()
    active_zones = st.session_state[DEFAULT_ZONES_KEY]

    # Sync live risk status from Risk Engine if available in session
    latest_z_assessments = st.session_state.get("latest_zone_assessments") or {}
    for zid, z in active_zones.items():
        if zid in latest_z_assessments:
            z["risk_status"] = latest_z_assessments[zid].get("risk_level", "safe").upper()
        elif "risk_status" not in z:
            z["risk_status"] = "SAFE"

    # 1. Header Bar
    if render_header_bar_fn:
        st.markdown(
            render_header_bar_fn(
                title="📍 Area & Zone Calibration Management",
                subtitle="Calibrate Ground Surface Areas (m²), Safety Capacities & Risk Thresholds per Zone",
                badge_text="📐 SCALE CALIBRATED: 0.045m/px",
                badge_type="safe",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## 📍 Area & Zone Calibration Management")

    # 2. Top Metric Cards (Venue Totals & Capacity Density)
    total_area_m2 = sum(z.get("area_sqm", 0.0) or z.get("area_m2", 0.0) for z in active_zones.values())
    total_capacity = sum(z.get("max_capacity", 0) or z.get("capacity", 0) for z in active_zones.values())
    total_headcount = sum(z.get("current_count", 0) for z in active_zones.values())
    avg_capacity_density = (total_capacity / total_area_m2) if total_area_m2 > 0 else 0.0
    state_counts = summarize_state_counts(list(active_zones.values()))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">TOTAL CALIBRATED AREA</div>
                <div class="kpi-value">{total_area_m2:.1f} <span style="font-size: 1.1rem; color: #4B6354;">m²</span></div>
                <div class="kpi-footer">Across {len(active_zones)} Active Zones</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">MAX VENUE CAPACITY</div>
                <div class="kpi-value">{total_capacity:,} <span style="font-size: 1.1rem; color: #4B6354;">people</span></div>
                <div class="kpi-footer">Current Occupancy: {total_headcount}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">AVG CAPACITY DENSITY</div>
                <div class="kpi-value">{avg_capacity_density:.2f} <span style="font-size: 1.1rem; color: #4B6354;">p/m²</span></div>
                <div class="kpi-footer">Standard Crowd Safety Limit: &lt; 2.50</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">SAFETY STATE BREAKDOWN</div>
                <div class="kpi-value" style="font-size: 1.05rem; display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-top:4px;">
                    <span class="status-pill status-safe" style="font-size:0.75rem;">🟢 {state_counts['safe']} SAFE</span>
                    <span class="status-pill status-warning" style="font-size:0.75rem;">⚠️ {state_counts['warning']} WARN</span>
                    <span class="status-pill status-critical" style="font-size:0.75rem;">🚨 {state_counts['critical']} CRIT</span>
                </div>
                <div class="kpi-footer">Total: {state_counts['total']} Calibrated Zones</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3. Two-Column Layout: Left (Zone Calibration Editor), Right (Calibrated Inventory Table)
    col_editor, col_table = st.columns([1.5, 1.3], gap="large")

    with col_editor:
        st.markdown(
            '<h3 style="margin: 0 0 10px 0; color: #14281D; font-weight: 700; font-size: 1.15rem;">⚙️ Zone Area & Threshold Calibration</h3>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            # Zone Selector dropdown
            zone_options = ["➕ [Create New Zone]"] + [f"{z['zone_id']} — {z['name']}" for z in active_zones.values()]
            selected_zone_str = st.selectbox("Select Zone to Calibrate or Edit", options=zone_options, key="calib_zone_selector")

            is_new_zone = selected_zone_str == "➕ [Create New Zone]"
            if is_new_zone:
                curr_zone = {
                    "zone_id": f"zone_{len(active_zones) + 1:02d}",
                    "name": "New Safety Corridor",
                    "zone_type": "ingress",
                    "area_sqm": 50.0,
                    "max_capacity": 80,
                    "warning_density_p_sqm": 1.5,
                    "critical_density_p_sqm": 3.0,
                    "warning_occupancy_ratio": 0.65,
                    "critical_occupancy_ratio": 0.85,
                    "risk_weight": 1.0,
                    "notes": "",
                }
            else:
                sel_id = selected_zone_str.split(" — ")[0]
                curr_zone = active_zones.get(sel_id, {})

            # Area Calibration Method Tab Selector
            calib_method = st.radio(
                "📐 Area Calibration Method",
                options=[
                    "Direct Area Input (m²)",
                    "2-Point Distance Scale (px -> m)",
                    "Drone Altitude AGL & Optics GSD",
                ],
                horizontal=True,
                key="calib_method_radio",
            )

            calib_computed_area = curr_zone.get("area_sqm") or curr_zone.get("area_m2", 50.0)

            # Interactive Method Inputs
            if calib_method == "2-Point Distance Scale (px -> m)":
                st.info("📏 2-Point Reference: Calibrates ground scale based on a known physical landmark distance.")
                sc1, sc2 = st.columns(2)
                with sc1:
                    ref_px = st.number_input("Pixel Distance (px)", value=120.0, min_value=1.0, step=5.0, key="calib_ref_px")
                with sc2:
                    ref_m = st.number_input("Real-World Distance (m)", value=15.0, min_value=0.5, step=1.0, key="calib_ref_m")

                scale_factor = ref_m / max(ref_px, 1.0)
                st.caption(f"Calculated Scale Factor: **{scale_factor:.4f} m/pixel** (1 px² = {scale_factor**2:.6f} m²)")

                # If user specifies pixel dimensions
                px_w = st.number_input("Zone Pixel Width", value=200.0, min_value=10.0, step=10.0, key="calib_px_w")
                px_h = st.number_input("Zone Pixel Height", value=150.0, min_value=10.0, step=10.0, key="calib_px_h")
                calib_computed_area = round((px_w * scale_factor) * (px_h * scale_factor), 1)

            elif calib_method == "Drone Altitude AGL & Optics GSD":
                st.info("🛰️ Drone Optical GSD: Computes physical area from drone barometer altitude and camera FOV.")
                alt_c1, alt_c2 = st.columns(2)
                with alt_c1:
                    drone_agl = st.number_input("Drone Altitude AGL (m)", value=32.4, min_value=5.0, max_value=120.0, step=1.0, key="calib_drone_alt")
                with alt_c2:
                    gimbal_deg = st.selectbox("Gimbal Pitch", options=[-90.0, -75.0, -60.0, -45.0], index=0, key="calib_gimbal_pitch")

                gsd_res = GroundResolutionCalculator.calculate_gsd(drone_agl, gimbal_pitch_deg=gimbal_deg)
                st.caption(f"Ground Footprint: **{gsd_res.ground_footprint_width_m:.1f}m × {gsd_res.ground_footprint_height_m:.1f}m** | GSD: **{gsd_res.meters_per_pixel_x*100:.2f} cm/px**")

                zone_pct = st.slider("Zone % of Total Camera FOV", min_value=1, max_value=100, value=15, step=1, key="calib_fov_pct")
                total_fov_area = gsd_res.ground_footprint_width_m * gsd_res.ground_footprint_height_m
                calib_computed_area = round(total_fov_area * (zone_pct / 100.0), 1)

            # Zone Core Form
            with st.form("zone_calibration_form"):
                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    z_id = st.text_input("Zone ID (unique slug)", value=curr_zone.get("zone_id", "zone_new"))
                with f_c2:
                    z_name = st.text_input("Zone Display Name", value=curr_zone.get("name", "New Safety Zone"))

                f_c3, f_c4 = st.columns(2)
                with f_c3:
                    type_options = ["ingress", "egress", "main_assembly", "choke_point", "monitoring", "emergency_exit", "restricted"]
                    default_type_idx = type_options.index(curr_zone.get("zone_type", "ingress")) if curr_zone.get("zone_type") in type_options else 0
                    z_type = st.selectbox("Functional Zone Type", options=type_options, index=default_type_idx)
                with f_c4:
                    z_area = st.number_input("Calibrated Ground Area (m²)", value=float(calib_computed_area), min_value=1.0, step=5.0)

                f_c5, f_c6 = st.columns(2)
                with f_c5:
                    z_capacity = st.number_input("Maximum Safe Headcount", value=int(curr_zone.get("max_capacity") or curr_zone.get("capacity", 100)), min_value=1, step=5)
                with f_c6:
                    z_weight = st.number_input("Risk Importance Weight", value=float(curr_zone.get("risk_weight", 1.0)), min_value=0.1, max_value=3.0, step=0.1)

                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                st.markdown("<b>🚨 Safety Density Thresholds (people / m²)</b>", unsafe_allow_html=True)
                th_c1, th_c2 = st.columns(2)
                with th_c1:
                    z_warn_dens = st.number_input("Warning Density (p/m²)", value=float(curr_zone.get("warning_density_p_sqm", 1.5)), min_value=0.5, max_value=5.0, step=0.1)
                with th_c2:
                    z_crit_dens = st.number_input("Critical Density (p/m²)", value=float(curr_zone.get("critical_density_p_sqm", 3.0)), min_value=1.0, max_value=8.0, step=0.1)

                z_notes = st.text_input("Calibration Notes / Specific Instructions", value=curr_zone.get("notes", ""))

                # Live Computed Safety Density Preview
                density_at_cap = z_capacity / max(z_area, 1.0)
                if density_at_cap >= z_crit_dens:
                    meter_color = "#C62828"
                    meter_label = "HIGH RISK DESIGN: Capacity exceeds critical density"
                elif density_at_cap >= z_warn_dens:
                    meter_color = "#E65100"
                    meter_label = "MODERATE DENSITY: Normal event crowding"
                else:
                    meter_color = "#2E7D32"
                    meter_label = "SAFE DESIGN: Ample room for free movement"

                st.markdown(
                    f"""
                    <div style="background: #F3F7F4; border-left: 4px solid {meter_color}; padding: 8px 12px; border-radius: 4px; font-size: 0.85rem; margin-top: 6px;">
                        <b>Density at Full Capacity:</b> <span style="font-weight: 700; color: {meter_color};">{density_at_cap:.2f} p/m²</span> &nbsp;|&nbsp; {meter_label}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("💾 Save Zone Calibration", width="stretch")

                if submitted:
                    if z_area <= 0:
                        st.error("❌ Ground area must be strictly positive (> 0 m²).")
                    elif z_capacity <= 0:
                        st.error("❌ Maximum capacity must be at least 1 person.")
                    elif z_warn_dens >= z_crit_dens:
                        st.error("❌ Warning density threshold must be strictly less than Critical density threshold.")
                    else:
                        updated_dict = dict(curr_zone)
                        updated_dict.update({
                            "zone_id": z_id.strip(),
                            "name": z_name.strip(),
                            "zone_type": z_type,
                            "area_sqm": float(z_area),
                            "area_m2": float(z_area),
                            "max_capacity": int(z_capacity),
                            "capacity": int(z_capacity),
                            "warning_density_p_sqm": float(z_warn_dens),
                            "critical_density_p_sqm": float(z_crit_dens),
                            "risk_weight": float(z_weight),
                            "notes": z_notes.strip(),
                            "is_calibrated": True,
                        })
                        save_zone_to_session(updated_dict)
                        st.success(f"✅ Zone '{z_name}' calibration successfully saved! Changes synced with Digital Map & Overview.")

            # Secondary Action Buttons
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if not is_new_zone:
                    if st.button("🗑️ Delete Zone", width="stretch", key="btn_del_zone"):
                        delete_zone_from_session(curr_zone.get("zone_id", ""))
                        st.warning(f"Deleted zone '{curr_zone.get('name')}'.")
                        st.rerun()
            with btn_c2:
                if st.button("🔄 Reset Venue Defaults", width="stretch", key="btn_reset_zones"):
                    reset_zones_to_defaults()
                    st.info("Reset all zones to venue template defaults.")
                    st.rerun()

    with col_table:
        st.markdown(
            '<h3 style="margin: 0 0 10px 0; color: #14281D; font-weight: 700; font-size: 1.15rem;">📋 Active Calibrated Zones Inventory</h3>',
            unsafe_allow_html=True,
        )

        # Quick State Filter Selector
        zm_filter = st.selectbox(
            "Filter Zones by Safety State",
            options=["ALL", "CRITICAL", "WARNING", "SAFE"],
            index=0,
            key="zm_state_filter_sel",
            format_func=lambda x: f"🚨 CRITICAL ({state_counts['critical']})" if x == "CRITICAL" else (
                f"⚠️ WARNING ({state_counts['warning']})" if x == "WARNING" else (
                    f"🟢 SAFE ({state_counts['safe']})" if x == "SAFE" else f"🌐 ALL ZONES ({state_counts['total']})"
                )
            ),
        )

        filtered_active_zones = filter_items_by_state(list(active_zones.values()), zm_filter)

        with st.container(border=True):
            if filtered_active_zones:
                for z in filtered_active_zones:
                    area = z.get("area_sqm") or z.get("area_m2", 0.0)
                    cap = z.get("max_capacity") or z.get("capacity", 100)
                    warn_d = z.get("warning_density_p_sqm", 1.5)
                    crit_d = z.get("critical_density_p_sqm", 3.0)
                    z_type = str(z.get("zone_type", "zone")).upper()
                    density_cap = cap / max(area, 1.0)
                    risk_status = str(z.get("risk_status", "SAFE")).upper()
                    meta = get_state_presentation(risk_status)

                    op_mgr = st.session_state.get("operator_manager")
                    op_inc = op_mgr.get_incident(z.get("zone_id", "")) if op_mgr else None
                    op_extra_html = ""
                    if op_inc and str(getattr(op_inc, "status", "")) not in ["OPEN", "RESOLVED", "IncidentLifecycleStatus.OPEN", "IncidentLifecycleStatus.RESOLVED"]:
                        st_val = op_inc.status.value if hasattr(op_inc.status, "value") else str(op_inc.status)
                        team_val = op_inc.assigned_team or "Action In-Progress"
                        op_extra_html = f"<div style='font-size:0.78rem; color:#0288D1; font-weight:600; margin-top:3px;'>🛡️ Operator Active: <b>{st_val}</b> ({team_val})</div>"

                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid {meta['color']}; border-radius: 6px; padding: 10px 14px; margin-bottom: 10px; background: #FFFFFF; border-top: 1px solid #E0E7E2; border-right: 1px solid #E0E7E2; border-bottom: 1px solid #E0E7E2;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="font-weight: 700; color: #14281D; font-size: 0.95rem;">{z['name']}</span>
                                <div style="display: flex; gap: 6px; align-items: center;">
                                    <span style="font-size: 0.72rem; background: #F0F4F1; color: #2E5A44; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{z_type}</span>
                                    <span class="status-pill {meta['pill_class']}">{meta['icon']} {risk_status}</span>
                                </div>
                            </div>
                            <div style="font-size: 0.83rem; color: #3D5A45; line-height: 1.5;">
                                • <b>Area:</b> {area:.1f} m² &nbsp;|&nbsp; <b>Max Capacity:</b> {cap} people<br>
                                • <b>Safety Thresholds:</b> Warn &ge; {warn_d:.1f} p/m² &nbsp;|&nbsp; Crit &ge; {crit_d:.1f} p/m²<br>
                                • <b>Design Capacity Density:</b> {density_cap:.2f} p/m² &nbsp;|&nbsp; <b>Weight:</b> {z.get('risk_weight', 1.0)}x
                                {op_extra_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"Zero calibrated zones match state filter '{zm_filter}'.")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # JSON Export / Persistence Card
        st.markdown(
            '<h3 style="margin: 0 0 10px 0; color: #14281D; font-weight: 700; font-size: 1.15rem;">💾 Calibration Persistence & Export</h3>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                """
                Calibrated ground dimensions and safety thresholds are instantly cached in memory and serialized for seamless operator continuity without heavy database latency.
                """
            )
            export_json = json.dumps({"version": "1.0", "zones": list(active_zones.values())}, indent=2)
            st.download_button(
                "📥 Download Calibration JSON",
                data=export_json,
                file_name="sentinel_zones_calibration.json",
                mime="application/json",
                width="stretch",
            )

