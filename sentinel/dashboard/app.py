import streamlit as st
import requests
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SENTINEL - Crowd Risk Monitoring",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"


def fetch_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def fetch_events():
    try:
        r = requests.get(f"{API_BASE}/api/v1/events", timeout=2)
        return r.json().get("events", []) if r.status_code == 200 else []
    except Exception:
        return []


def fetch_zones():
    try:
        r = requests.get(f"{API_BASE}/api/v1/zones", timeout=2)
        return r.json().get("zones", []) if r.status_code == 200 else []
    except Exception:
        return []


def fetch_drones():
    try:
        r = requests.get(f"{API_BASE}/api/v1/drones", timeout=2)
        return r.json().get("drones", []) if r.status_code == 200 else []
    except Exception:
        return []


def fetch_analytics():
    try:
        r = requests.get(f"{API_BASE}/api/v1/analytics/summary", timeout=2)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def fetch_alerts():
    try:
        r = requests.get(f"{API_BASE}/api/v1/alerts", timeout=2)
        return r.json().get("alerts", []) if r.status_code == 200 else []
    except Exception:
        return []


def main():
    st.sidebar.title("🚁 SENTINEL")
    st.sidebar.caption("Drone Crowd Risk Detection")

    health = fetch_health()
    if health:
        st.sidebar.success(f"✅ API Connected")
        st.sidebar.caption(f"Version: {health.get('version', 'N/A')}")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.sidebar.caption("Start backend on port 8000")

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Overview",
            "🎯 Live Monitoring",
            "🗺️ Digital Map",
            "📍 Zone Management",
            "👥 Crowd Analytics",
            "⚠️ Risk & Alerts",
            "⚙️ System Settings",
        ],
    )

    if page == "📊 Overview":
        render_overview()
    elif page == "🎯 Live Monitoring":
        render_live_monitoring()
    elif page == "🗺️ Digital Map":
        render_digital_map()
    elif page == "📍 Zone Management":
        render_zone_management()
    elif page == "👥 Crowd Analytics":
        render_crowd_analytics()
    elif page == "⚠️ Risk & Alerts":
        render_risk_alerts()
    elif page == "⚙️ System Settings":
        render_settings()


def render_overview():
    st.title("📊 Overview Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    analytics = fetch_analytics()
    zones = fetch_zones()

    with col1:
        st.metric("Total Crowd Count", analytics.get("total_count", 650), delta="+12")
    with col2:
        st.metric("Active Zones", analytics.get("active_zones", 3))
    with col3:
        st.metric("Warning Zones", analytics.get("warning_zones", 1), delta="+1", delta_color="inverse")
    with col4:
        st.metric("Critical Zones", analytics.get("critical_zones", 1), delta="+1", delta_color="inverse")

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Zone Occupancy")
        if zones:
            df = pd.DataFrame(zones)
            fig = px.bar(df, x="name", y="current_count", color="type",
                         title="Current Occupancy by Zone",
                         labels={"current_count": "People Count", "name": "Zone"})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Distribution")
        risk_data = pd.DataFrame({
            "Level": ["Safe", "Warning", "Critical"],
            "Count": [1, 1, 1]
        })
        fig = px.pie(risk_data, values="Count", names="Level",
                     color="Level",
                     color_discrete_map={"Safe": "green", "Warning": "orange", "Critical": "red"},
                     title="Zone Risk Levels")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Alerts")
    alerts = fetch_alerts()
    if alerts:
        for alert in alerts:
            level_color = {"warning": "🟠", "critical": "🔴", "info": "🔵"}.get(alert.get("risk_level", ""), "⚪")
            st.write(f"{level_color} **{alert.get('zone_id', 'N/A')}** - {alert.get('message', 'No message')} ({alert.get('timestamp', '')[:19]})")
    else:
        st.info("No active alerts")


def render_live_monitoring():
    st.title("🎯 Live Drone Monitoring")

    drones = fetch_drones()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Video Feed")
        st.info("📹 Video stream placeholder - Connect RTSP stream or upload video file")
        st.image("https://via.placeholder.com/800x450/1a1a2e/ffffff?text=Drone+Video+Feed", use_container_width=True)

        st.caption("Frame: 1,234 | FPS: 29.8 | Latency: 45ms | Resolution: 1920x1080")

    with col2:
        st.subheader("Drone Status")
        for drone in drones:
            status_color = "🟢" if drone.get("status") == "streaming" else "🔴"
            st.write(f"{status_color} **{drone.get('name')}** ({drone.get('id')})")
            st.caption(f"Status: {drone.get('status')}")
            st.caption(f"Event: {drone.get('event_id')}")

        st.divider()
        st.subheader("Detection Stats")
        st.metric("People Detected", "47", delta="+3")
        st.metric("Active Tracks", "42", delta="+2")
        st.metric("Processing FPS", "28.5")
        st.metric("Confidence Threshold", "0.50")

    st.divider()
    st.subheader("Telemetry")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latitude", "37.7749°")
        st.metric("Longitude", "-122.4194°")
    with col2:
        st.metric("Altitude", "120.5 m")
        st.metric("Battery", "87%")
    with col3:
        st.metric("Gimbal Pitch", "-89.2°")
        st.metric("Flight Mode", "GUIDED")


def render_digital_map():
    st.title("🗺️ Digital Map")

    st.info("🗺️ Interactive map with Folium - Shows event venue, zones, drone position, and risk levels")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        **Map Features (to be implemented):**
        - Event boundary polygon
        - Monitoring zones with color-coded risk
        - Entry/exit gates
        - Emergency exits
        - Restricted areas
        - Real-time drone position
        - Crowd density heatmap overlay
        - Alert markers
        """)

        st.image("https://via.placeholder.com/800x500/0d1b2a/ffffff?text=Interactive+Map+View", use_container_width=True)

    with col2:
        st.subheader("Map Layers")
        st.checkbox("Event Boundary", value=True)
        st.checkbox("Monitoring Zones", value=True)
        st.checkbox("Entry/Exit Gates", value=True)
        st.checkbox("Emergency Exits", value=True)
        st.checkbox("Restricted Areas", value=True)
        st.checkbox("Drone Position", value=True)
        st.checkbox("Density Heatmap", value=False)
        st.checkbox("Alert Markers", value=True)

        st.divider()
        st.subheader("Zone Legend")
        st.markdown("🟢 Safe - Normal conditions")
        st.markdown("🟠 Warning - Elevated risk")
        st.markdown("🔴 Critical - Immediate attention")


def render_zone_management():
    st.title("📍 Zone Management")

    zones = fetch_zones()

    if zones:
        df = pd.DataFrame(zones)
        st.dataframe(
            df[["zone_id", "name", "type", "capacity", "current_count"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Add New Zone")
        with st.form("add_zone"):
            zone_id = st.text_input("Zone ID")
            name = st.text_input("Zone Name")
            zone_type = st.selectbox("Type", ["monitoring", "entry", "exit", "emergency_exit", "restricted"])
            capacity = st.number_input("Capacity", min_value=0, value=500)
            warning_thresh = st.slider("Warning Threshold", 0.0, 1.0, 0.7)
            critical_thresh = st.slider("Critical Threshold", 0.0, 1.0, 0.9)
            submitted = st.form_submit_button("Create Zone")
            if submitted:
                st.success(f"Zone '{name}' created (demo)")

    with col2:
        st.subheader("Zone Configuration")
        st.info("Draw polygon boundaries on map to define zone areas")
        st.info("Set entry/exit lines for flow counting")
        st.info("Configure calibration data for ground-plane mapping")


def render_crowd_analytics():
    st.title("👥 Crowd Analytics")

    tabs = st.tabs(["Density", "Occupancy", "Movement", "Flow", "Bottlenecks"])

    with tabs[0]:
        st.subheader("Density Analysis")
        st.info("People per unit ground area (requires calibration)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Density", "2.3 people/m²")
            st.metric("Max Density", "5.8 people/m²")
        with col2:
            st.metric("Low Density Areas", "45%")
            st.metric("High Density Areas", "12%")
        st.image("https://via.placeholder.com/700x350/1a1a2e/ffffff?text=Density+Heatmap", use_container_width=True)

    with tabs[1]:
        st.subheader("Occupancy Analysis")
        zones = fetch_zones()
        if zones:
            for zone in zones:
                occupancy = (zone.get("current_count", 0) / max(zone.get("capacity", 1), 1)) * 100
                color = "green" if occupancy < 70 else "orange" if occupancy < 90 else "red"
                st.write(f"**{zone['name']}**: {zone['current_count']}/{zone['capacity']} ({occupancy:.0f}%)")
                st.progress(min(occupancy / 100, 1.0))

    with tabs[2]:
        st.subheader("Movement Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Speed", "1.2 m/s")
            st.metric("Dominant Direction", "NE (45°)")
        with col2:
            st.metric("Abnormal Movement", "3 tracks")
            st.metric("Sudden Changes", "1 event")

    with tabs[3]:
        st.subheader("Flow Analysis")
        st.metric("Entry Flow", "45 people/min")
        st.metric("Exit Flow", "38 people/min")
        st.metric("Net Flow", "+7 people/min")
        st.caption("Positive = accumulation, Negative = evacuation")

    with tabs[4]:
        st.subheader("Bottleneck Detection")
        st.warning("⚠️ Bottleneck detected at Exit Gate A")
        st.write("- Accumulation: 80 people")
        st.write("- Flow reduction: 65%")
        st.write("- Persistence: 45 seconds")


def render_risk_alerts():
    st.title("⚠️ Risk Monitoring & Alerts")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Zone Risk Scores")
        risk_data = pd.DataFrame({
            "Zone": ["Main Entrance", "Central Plaza", "Exit Gate A"],
            "Risk Score": [25, 68, 82],
            "Level": ["Safe", "Warning", "Critical"]
        })
        fig = px.bar(risk_data, x="Zone", y="Risk Score", color="Level",
                     color_discrete_map={"Safe": "green", "Warning": "orange", "Critical": "red"})
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Safe Threshold")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Active Alerts")
        alerts = fetch_alerts()
        for alert in alerts:
            with st.container(border=True):
                level = alert.get("risk_level", "info")
                icon = {"warning": "🟠", "critical": "🔴", "info": "🔵"}.get(level, "⚪")
                st.write(f"{icon} **{level.upper()}** - {alert.get('zone_id')}")
                st.caption(alert.get("message", ""))
                st.caption(f"Score: {alert.get('risk_score', 0):.0f}")
                if st.button("Acknowledge", key=f"ack_{alert.get('id')}"):
                    st.success("Acknowledged")

    st.divider()
    st.subheader("Alert History")
    history_data = pd.DataFrame({
        "Time": ["10:15:23", "10:12:45", "10:08:12", "10:05:33"],
        "Zone": ["Central Plaza", "Exit Gate A", "Main Entrance", "Central Plaza"],
        "Level": ["🟠 Warning", "🔴 Critical", "🟢 Safe", "🟠 Warning"],
        "Message": ["High density", "Bottleneck", "Normalized", "Flow imbalance"],
        "Status": ["Active", "Acknowledged", "Resolved", "Active"]
    })
    st.dataframe(history_data, use_container_width=True, hide_index=True)


def render_settings():
    st.title("⚙️ System Settings")

    tabs = st.tabs(["General", "Detection", "Tracking", "Analytics", "Risk Engine", "Alerts"])

    with tabs[0]:
        st.subheader("General Configuration")
        st.text_input("App Name", value="SENTINEL")
        st.selectbox("Environment", ["development", "staging", "production"], index=0)
        st.checkbox("Debug Mode", value=True)
        st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)

    with tabs[1]:
        st.subheader("YOLO Detection")
        st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)
        st.number_input("Max Detections/Frame", value=1000)
        st.selectbox("Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt"])
        st.selectbox("Device", ["cpu", "cuda", "mps"])

    with tabs[2]:
        st.subheader("Tracking Configuration")
        st.selectbox("Tracker", ["bytetrack", "botsort"])
        st.number_input("Track Buffer", value=30)
        st.slider("Match Threshold", 0.0, 1.0, 0.8, 0.05)
        st.checkbox("Enable Re-ID", value=False)

    with tabs[3]:
        st.subheader("Analytics Thresholds")
        st.slider("Occupancy Warning", 0.0, 1.0, 0.7, 0.05)
        st.slider("Occupancy Critical", 0.0, 1.0, 0.9, 0.05)
        st.number_input("Flow Window (seconds)", value=60)
        st.number_input("Bottleneck Persistence (frames)", value=30)

    with tabs[4]:
        st.subheader("Risk Engine Weights")
        st.slider("Density Weight", 0.0, 1.0, 0.3, 0.05)
        st.slider("Occupancy Weight", 0.0, 1.0, 0.2, 0.05)
        st.slider("Movement Weight", 0.0, 1.0, 0.2, 0.05)
        st.slider("Flow Weight", 0.0, 1.0, 0.15, 0.05)
        st.slider("Bottleneck Weight", 0.0, 1.0, 0.15, 0.05)
        st.divider()
        st.slider("Safe Threshold", 0, 100, 30)
        st.slider("Warning Threshold", 0, 100, 60)
        st.slider("Critical Threshold", 0, 100, 80)
        st.slider("Temporal Smoothing", 0.0, 1.0, 0.3, 0.05)

    with tabs[5]:
        st.subheader("Alert Configuration")
        st.number_input("Cooldown (seconds)", value=30)
        st.number_input("Max Active Alerts", value=100)
        st.checkbox("Sound Notifications", value=True)
        st.selectbox("Default Severity", ["info", "warning", "critical"])


if __name__ == "__main__":
    main()