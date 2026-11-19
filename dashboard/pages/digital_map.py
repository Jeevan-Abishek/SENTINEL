"""Digital Map & Spatial Geospatial Overview Page for SENTINEL Dashboard.

Provides high-speed interactive Folium mapping with graceful offline fallback
and zero-latency Plotly vector graphics optimized for low-RAM Windows CPU environments.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

try:
    from app.mapping.map_renderer import MapRenderer
    from app.mapping.calibration import VenueCoordinates, GPSCoordinateConverter
except ImportError:
    from sentinel.app.mapping.map_renderer import MapRenderer
    from sentinel.app.mapping.calibration import VenueCoordinates, GPSCoordinateConverter

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


# -----------------------------------------------------------------------------
# Venue Presets with Calibrated Zones
# -----------------------------------------------------------------------------
VENUE_PRESETS = {
    "Bengaluru Kanteerava Arena": {
        "lat": 12.9716,
        "lon": 77.5946,
        "zoom": 18,
        "radius_m": 160.0,
        "drone_alt": 32.4,
        "drone_heading": 45.0,
        "zones": [
            {
                "zone_id": "bengaluru_gate_north",
                "name": "Ingress Gate North",
                "zone_type": "ingress",
                "risk_status": "WARNING",
                "current_count": 88,
                "max_capacity": 100,
                "density_p_sqm": 2.20,
                "area_sqm": 40.0,
                "flow_vector": (135.0, 1.4),
                "gps_polygon": [
                    (12.97205, 77.59420),
                    (12.97205, 77.59500),
                    (12.97175, 77.59500),
                    (12.97175, 77.59420),
                ],
            },
            {
                "zone_id": "bengaluru_plaza_central",
                "name": "Central Assembly Plaza",
                "zone_type": "main_assembly",
                "risk_status": "SAFE",
                "current_count": 134,
                "max_capacity": 300,
                "density_p_sqm": 0.89,
                "area_sqm": 150.0,
                "flow_vector": (45.0, 0.9),
                "gps_polygon": [
                    (12.97165, 77.59425),
                    (12.97165, 77.59495),
                    (12.97125, 77.59495),
                    (12.97125, 77.59425),
                ],
            },
            {
                "zone_id": "bengaluru_corridor_choke",
                "name": "Choke Corridor B",
                "zone_type": "choke_point",
                "risk_status": "CRITICAL",
                "current_count": 46,
                "max_capacity": 45,
                "density_p_sqm": 3.07,
                "area_sqm": 15.0,
                "flow_vector": (180.0, 0.4),
                "gps_polygon": [
                    (12.97125, 77.59445),
                    (12.97125, 77.59475),
                    (12.97105, 77.59475),
                    (12.97105, 77.59445),
                ],
            },
            {
                "zone_id": "bengaluru_exit_south",
                "name": "South Egress Concourse",
                "zone_type": "egress",
                "risk_status": "SAFE",
                "current_count": 32,
                "max_capacity": 150,
                "density_p_sqm": 0.53,
                "area_sqm": 60.0,
                "flow_vector": (180.0, 1.3),
                "gps_polygon": [
                    (12.97100, 77.59430),
                    (12.97100, 77.59490),
                    (12.97075, 77.59490),
                    (12.97075, 77.59430),
                ],
            },
        ],
        "flows": [
            {
                "start_lat": 12.97190,
                "start_lon": 77.59460,
                "end_lat": 12.97145,
                "end_lon": 77.59460,
                "rate_ppm": 24,
                "is_choke": False,
            },
            {
                "start_lat": 12.97125,
                "start_lon": 77.59460,
                "end_lat": 12.97105,
                "end_lon": 77.59460,
                "rate_ppm": 12,
                "is_choke": True,
            },
        ],
    },
    "Wembley Safety Concourse": {
        "lat": 51.5560,
        "lon": -0.2795,
        "zoom": 18,
        "radius_m": 180.0,
        "drone_alt": 38.0,
        "drone_heading": 90.0,
        "zones": [
            {
                "zone_id": "wembley_north_turnstiles",
                "name": "North Turnstiles",
                "zone_type": "ingress",
                "risk_status": "SAFE",
                "current_count": 65,
                "max_capacity": 150,
                "density_p_sqm": 1.08,
                "area_sqm": 60.0,
                "flow_vector": (90.0, 1.1),
                "gps_polygon": [
                    (51.55640, -0.28020),
                    (51.55640, -0.27900),
                    (51.55615, -0.27900),
                    (51.55615, -0.28020),
                ],
            },
            {
                "zone_id": "wembley_main_concourse",
                "name": "Olympic Promenade Concourse",
                "zone_type": "main_assembly",
                "risk_status": "WARNING",
                "current_count": 210,
                "max_capacity": 250,
                "density_p_sqm": 1.75,
                "area_sqm": 120.0,
                "flow_vector": (120.0, 0.8),
                "gps_polygon": [
                    (51.55610, -0.28040),
                    (51.55610, -0.27860),
                    (51.55580, -0.27860),
                    (51.55580, -0.28040),
                ],
            },
            {
                "zone_id": "wembley_egress_ramp",
                "name": "South Egress Ramp",
                "zone_type": "egress",
                "risk_status": "SAFE",
                "current_count": 45,
                "max_capacity": 200,
                "density_p_sqm": 0.56,
                "area_sqm": 80.0,
                "flow_vector": (180.0, 1.5),
                "gps_polygon": [
                    (51.55575, -0.27980),
                    (51.55575, -0.27920),
                    (51.55545, -0.27920),
                    (51.55545, -0.27980),
                ],
            },
        ],
        "flows": [
            {
                "start_lat": 51.55625,
                "start_lon": -0.27960,
                "end_lat": 51.55595,
                "end_lon": -0.27950,
                "rate_ppm": 35,
                "is_choke": False,
            }
        ],
    },
    "Central Promenade & Festival": {
        "lat": 37.7749,
        "lon": -122.4194,
        "zoom": 18,
        "radius_m": 150.0,
        "drone_alt": 35.0,
        "drone_heading": 0.0,
        "zones": [
            {
                "zone_id": "promenade_main_stage",
                "name": "Main Stage Gathering",
                "zone_type": "main_assembly",
                "risk_status": "WARNING",
                "current_count": 185,
                "max_capacity": 220,
                "density_p_sqm": 1.85,
                "area_sqm": 100.0,
                "flow_vector": (0.0, 0.5),
                "gps_polygon": [
                    (37.77530, -122.41990),
                    (37.77530, -122.41890),
                    (37.77490, -122.41890),
                    (37.77490, -122.41990),
                ],
            },
            {
                "zone_id": "promenade_exit_choke",
                "name": "West Gate Funnel",
                "zone_type": "choke_point",
                "risk_status": "CRITICAL",
                "current_count": 52,
                "max_capacity": 50,
                "density_p_sqm": 3.47,
                "area_sqm": 15.0,
                "flow_vector": (270.0, 0.3),
                "gps_polygon": [
                    (37.77485, -122.42010),
                    (37.77485, -122.41970),
                    (37.77465, -122.41970),
                    (37.77465, -122.42010),
                ],
            },
        ],
        "flows": [
            {
                "start_lat": 37.77500,
                "start_lon": -122.41940,
                "end_lat": 37.77475,
                "end_lon": -122.41980,
                "rate_ppm": 18,
                "is_choke": True,
            }
        ],
    },
}


# -----------------------------------------------------------------------------
# Caching Functions for Snappy, Low-RAM Renders (< 50ms)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_folium_map_html(
    venue_key: str,
    lat: float,
    lon: float,
    zoom: int,
    radius_m: float,
    drone_alt: float,
    drone_heading: float,
    show_heatmap: bool,
    zones_tuple: Tuple,
    flows_tuple: Tuple,
) -> str:
    """Generate and cache Folium HTML map representation."""
    renderer = MapRenderer(
        center_lat=lat,
        center_lon=lon,
        zoom=zoom,
        venue_name=venue_key,
        venue_radius_m=radius_m,
    )
    zones_list = [dict(z) for z in zones_tuple]
    flows_list = [dict(f) for f in flows_tuple]
    return renderer.render_folium_map(
        zones=zones_list,
        drone_lat=lat,
        drone_lon=lon,
        drone_alt_m=drone_alt,
        drone_heading_deg=drone_heading,
        crowd_flows=flows_list,
        show_heatmap=show_heatmap,
    )


@st.cache_data(show_spinner=False)
def get_cached_vector_map_figure(
    venue_key: str,
    lat: float,
    lon: float,
    radius_m: float,
    drone_alt: float,
    drone_heading: float,
    zones_tuple: Tuple,
    flows_tuple: Tuple,
) -> go.Figure:
    """Generate and cache high-speed Plotly vector figure."""
    renderer = MapRenderer(
        center_lat=lat,
        center_lon=lon,
        zoom=18,
        venue_name=venue_key,
        venue_radius_m=radius_m,
    )
    zones_list = [dict(z) for z in zones_tuple]
    flows_list = [dict(f) for f in flows_tuple]
    return renderer.render_plotly_vector_map(
        zones=zones_list,
        drone_lat=lat,
        drone_lon=lon,
        drone_alt_m=drone_alt,
        drone_heading_deg=drone_heading,
        crowd_flows=flows_list,
        height=530,
    )


def serialize_for_cache(data: List[Dict[str, Any]]) -> Tuple:
    """Helper to convert list of dicts to a hashable tuple for st.cache_data."""
    return tuple(tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items())) for d in data)


# -----------------------------------------------------------------------------
# Main Page Render Function
# -----------------------------------------------------------------------------
def render_digital_map_page(render_header_bar_fn=None):
    """Render the complete GPS & Digital Map Page."""
    # 1. Page Header Bar
    if render_header_bar_fn:
        st.markdown(
            render_header_bar_fn(
                title="🗺️ Digital Map & Spatial Geospatial Overview",
                subtitle="Ground-Plane Calibrated Boundaries, Ingress/Egress Corridors & Drone GPS Coordinates",
                badge_text="🛰️ RTK GPS SYNCED (±0.02m)",
                badge_type="safe",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## 🗺️ Digital Map & Spatial Geospatial Overview")

    # 2. Venue Selection & Map View Mode Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.6, 1.2, 1.2], gap="medium")

    with ctrl_col1:
        selected_venue = st.selectbox(
            "📍 Event Venue Preset",
            options=list(VENUE_PRESETS.keys()) + ["⚙️ Custom GPS Anchor"],
            index=0,
            key="digital_map_venue_select",
        )

    with ctrl_col2:
        map_view_mode = st.selectbox(
            "🗺️ Map View Engine",
            options=["⚡ High-Speed Vector Map (Offline / Instant)", "🛰️ OpenStreetMap / Satellite (Folium)"],
            index=0,
            key="digital_map_view_mode",
        )

    with ctrl_col3:
        drone_mode = st.selectbox(
            "🛰️ Drone Position Anchor",
            options=["Active Telemetry (Live)", "Nadir Hover (Center)", "Patrol Perimeter"],
            index=0,
            key="digital_map_drone_mode",
        )

    # Resolve Venue Configuration
    if selected_venue == "⚙️ Custom GPS Anchor":
        cust_c1, cust_c2, cust_c3, cust_c4 = st.columns(4)
        with cust_c1:
            venue_lat = st.number_input("Venue Latitude (°N)", value=12.9716, format="%.5f", key="dm_cust_lat")
        with cust_c2:
            venue_lon = st.number_input("Venue Longitude (°E)", value=77.5946, format="%.5f", key="dm_cust_lon")
        with cust_c3:
            venue_alt = st.number_input("Drone Altitude AGL (m)", value=32.0, min_value=5.0, max_value=120.0, key="dm_cust_alt")
        with cust_c4:
            venue_zoom = st.number_input("Map Zoom Level", value=18, min_value=14, max_value=20, key="dm_cust_zoom")
        venue_radius = 150.0
        venue_heading = 45.0
        zones_data = VENUE_PRESETS["Bengaluru Kanteerava Arena"]["zones"]
        flows_data = VENUE_PRESETS["Bengaluru Kanteerava Arena"]["flows"]
    else:
        v_info = VENUE_PRESETS[selected_venue]
        venue_lat = v_info["lat"]
        venue_lon = v_info["lon"]
        venue_zoom = v_info["zoom"]
        venue_radius = v_info["radius_m"]
        venue_alt = v_info["drone_alt"]
        venue_heading = v_info["drone_heading"]
        if selected_venue == "Bengaluru Kanteerava Arena" and "calibrated_zones" in st.session_state and st.session_state["calibrated_zones"]:
            zones_data = list(st.session_state["calibrated_zones"].values())
        else:
            zones_data = v_info["zones"]
        flows_data = v_info["flows"]

    # Sync live risk assessments from Risk Engine if available in session
    latest_z_assessments = st.session_state.get("latest_zone_assessments") or {}
    if latest_z_assessments:
        synced_zones = []
        for z in zones_data:
            z_copy = dict(z)
            zid = z_copy.get("zone_id")
            if zid in latest_z_assessments:
                z_ass = latest_z_assessments[zid]
                z_copy["risk_status"] = z_ass.get("risk_level", "safe").upper()
                if "risk_score" in z_ass:
                    z_copy["risk_score"] = z_ass["risk_score"]
            synced_zones.append(z_copy)
        zones_data = synced_zones

    # Offset drone if in Patrol Perimeter mode
    drone_heading: float = float(venue_heading) if "venue_heading" in locals() else 0.0
    if drone_mode == "Patrol Perimeter":
        drone_lat = venue_lat + (70.0 / 111139.0)
        drone_lon = venue_lon + (70.0 / (111139.0 * math.cos(math.radians(venue_lat))))
        drone_heading = 135.0
    else:
        drone_lat = venue_lat
        drone_lon = venue_lon
        drone_heading = float(venue_heading) if "venue_heading" in locals() else 0.0

    # Quick Risk State Filter Strip
    state_counts = summarize_state_counts(zones_data)
    filt_col1, filt_col2 = st.columns([2.2, 1.8], gap="medium")
    with filt_col1:
        map_state_filter = st.selectbox(
            "Filter Zones by Safety State",
            options=["ALL", "CRITICAL", "WARNING", "SAFE"],
            index=0,
            key="digital_map_state_filter",
            format_func=lambda x: f"🚨 CRITICAL ({state_counts['critical']})" if x == "CRITICAL" else (
                f"⚠️ WARNING ({state_counts['warning']})" if x == "WARNING" else (
                    f"🟢 SAFE ({state_counts['safe']})" if x == "SAFE" else f"🌐 ALL ZONES ({state_counts['total']})"
                )
            ),
        )
    with filt_col2:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; align-items:center; height:100%; gap:8px; padding-top:24px;">
                <span class="status-pill status-safe">🟢 {state_counts['safe']} SAFE</span>
                <span class="status-pill status-warning">⚠️ {state_counts['warning']} WARN</span>
                <span class="status-pill status-critical">🚨 {state_counts['critical']} CRIT</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Filtered zones for list presentation
    display_zones = filter_items_by_state(zones_data, map_state_filter)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 3. Two-Column Layout: Left (Map Canvas), Right (Layers & Telemetry)
    col_map, col_info = st.columns([2.5, 1.3], gap="large")

    with col_map:
        with st.container(border=True):
            if "Vector" in map_view_mode:
                # Instant Plotly Vector Map (Offline, Zero network lag)
                cached_zones = serialize_for_cache(zones_data)
                cached_flows = serialize_for_cache(flows_data)
                fig = get_cached_vector_map_figure(
                    selected_venue,
                    venue_lat,
                    venue_lon,
                    venue_radius,
                    venue_alt,
                    drone_heading,
                    cached_zones,
                    cached_flows,
                )
                st.plotly_chart(fig, width="stretch")
            else:
                # Interactive Folium Map
                cached_zones = serialize_for_cache(zones_data)
                cached_flows = serialize_for_cache(flows_data)
                map_html = get_cached_folium_map_html(
                    selected_venue,
                    venue_lat,
                    venue_lon,
                    venue_zoom,
                    venue_radius,
                    venue_alt,
                    drone_heading,
                    False,
                    cached_zones,
                    cached_flows,
                )
                components.html(map_html, height=530)

        # Quick Live Telemetry Strip under Map
        st.markdown(
            f"""
            <div style="background-color: #E8F3EC; border: 1px solid #234734; border-radius: 8px;
                        padding: 8px 14px; font-size: 0.85rem; color: #14281D; display: flex;
                        justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>📍 <b>Anchor:</b> {venue_lat:.5f}°N, {venue_lon:.5f}°E &nbsp;|&nbsp; <b>Zoom:</b> {venue_zoom}x</div>
                <div>🛰️ <b>Drone AGL:</b> {venue_alt:.1f}m &nbsp;|&nbsp; <b>Heading:</b> {drone_heading:.0f}° &nbsp;|&nbsp; <b>Gimbal:</b> -90.0°</div>
                <div>📡 <b>RTK Status:</b> <span style="color: #2E7D32; font-weight: 700;">FIXED (19 Sats)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_info:
        # Layer Control
        st.markdown(
            '<h3 style="margin: 0 0 10px 0; color: #14281D; font-weight: 700; font-size: 1.10rem;">🛰️ Geospatial Layers</h3>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.checkbox("Drone Position & Heading Vector", value=True, key="layer_drone")
            st.checkbox("Ground Calibrated Boundaries", value=True, key="layer_zones")
            st.checkbox("Real-Time Risk Level Coloring", value=True, key="layer_risk_color")
            st.checkbox("Crowd Optical Flow Vectors", value=True, key="layer_flow")
            st.checkbox("Bottleneck Choke Indicators", value=True, key="layer_chokes")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Live Calibrated Zones Summary
        st.markdown(
            f'<h3 style="margin: 0 0 10px 0; color: #14281D; font-weight: 700; font-size: 1.10rem;">📍 Calibrated Zones ({len(display_zones)} shown)</h3>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            if display_zones:
                for z in display_zones:
                    risk = z.get("risk_status", "SAFE").upper()
                    meta = get_state_presentation(risk)
                    border_c = meta["color"]
                    pill_class = meta["pill_class"]

                    op_mgr = st.session_state.get("operator_manager")
                    op_inc = op_mgr.get_incident(z.get("zone_id", "")) if op_mgr else None
                    op_info = ""
                    if op_inc and str(getattr(op_inc, "status", "")) not in ["OPEN", "RESOLVED", "IncidentLifecycleStatus.OPEN", "IncidentLifecycleStatus.RESOLVED"]:
                        st_val = op_inc.status.value if hasattr(op_inc.status, "value") else str(op_inc.status)
                        op_info = f"<div style='font-size:0.75rem; color:#0288D1; font-weight:600; margin-top:2px;'>🛡️ Operator: {st_val} ({op_inc.assigned_team or 'Active'})</div>"

                    st.markdown(
                        f"""
                        <div style="border-left: 3px solid {border_c}; padding: 6px 10px; margin-bottom: 8px; background: #FFFFFF; border-radius: 4px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-weight: 700; font-size: 0.88rem; color: #14281D;">{z['name']}</span>
                                <span class="status-pill {pill_class}">{meta['icon']} {risk}</span>
                            </div>
                            <div style="font-size: 0.80rem; color: #4B6354; margin-top: 2px;">
                                👥 {z.get('current_count', 0)} / {z.get('max_capacity', 100)} &nbsp;|&nbsp; 📐 {z.get('density_p_sqm', 0.0):.2f} p/m² &nbsp;|&nbsp; 📦 {z.get('area_sqm', 0.0):.0f} m²
                                {op_info}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info(f"Zero zones matching filter '{map_state_filter}'.")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Safety Calibration Legend
        st.markdown(
            '<h3 style="margin: 0 0 10px 0; color: #14281D; font-weight: 700; font-size: 1.10rem;">🏷️ Risk Density Legend</h3>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("<span class='status-pill status-safe'>SAFE</span> &nbsp;&lt; 1.5 people/m² (Free movement)", unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            st.markdown("<span class='status-pill status-warning'>WARNING</span> &nbsp;1.5 – 3.0 people/m² (Dense crowd)", unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            st.markdown("<span class='status-pill status-critical'>CRITICAL</span> &nbsp;&gt; 3.0 people/m² (Surge risk)", unsafe_allow_html=True)

