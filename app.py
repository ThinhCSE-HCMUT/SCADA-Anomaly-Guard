"""
Main entry point for the SCADA Anomaly Detection Dashboard.
This file only handles page config and main header.
Sidebar is now managed centrally in src/sidebar.py
"""

import streamlit as st
from src.config import APP_TITLE, APP_DESCRIPTION, DEFAULT_MODEL, AVAILABLE_MODELS, TARGET_TURBINES
from src.sidebar import render_sidebar

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': f"{APP_DESCRIPTION}\nDeveloped for Wind Turbine SCADA Anomaly Detection"
    }
)

# ========================= RENDER SIDEBAR =========================
selected_model = render_sidebar()
state = st.session_state
is_running = state.get("is_monitoring", False)
alert_count = len(state.get("anomaly_records", []))
active_model = selected_model or state.get("selected_model", DEFAULT_MODEL)
fleet_size = len(TARGET_TURBINES)

# ========================= HERO SECTION =========================
st.title(APP_TITLE)
st.markdown(f"**{APP_DESCRIPTION}**")
st.caption(
    f"Monitoring: {'Running' if is_running else 'Idle'}  |  "
    f"Active model: {active_model}  |  "
    f"Session alerts: {alert_count}"
)

st.divider()

# ========================= GLOBAL KPI METRICS =========================
st.subheader("System Snapshot")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Active Turbines", value=f"{fleet_size} / {fleet_size}", delta="Fleet ready")
with kpi2:
    st.metric(
        label="Monitoring",
        value="Running" if is_running else "Idle",
        delta="Live stream" if is_running else "Standby",
        delta_color="normal" if is_running else "off",
    )
with kpi3:
    st.metric(label="Active Model", value=active_model, delta=f"{len(AVAILABLE_MODELS)} available")
with kpi4:
    st.metric(label="Session Alerts", value=f"{alert_count}", delta="Logged anomalies")

# ========================= QUICK ACCESS CARDS =========================
st.subheader("Quick Access")
st.caption("Open the operational views that matter most.")


def render_card(title: str, body: str, page_path: str, action_label: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(body)
        st.page_link(page_path, label=action_label, width="stretch")


row1_left, row1_right = st.columns(2)
with row1_left:
    render_card(
        "Fleet Overview",
        "View fleet-wide health, sensor aggregates, and anomaly rates in one place.",
        "pages/01_Overview.py",
        "Open overview",
    )
with row1_right:
    render_card(
        "Real-time Monitor",
        "Watch live batches, compare model output, and inspect the current stream.",
        "pages/02_Real-time_Monitor.py",
        "Open monitor",
    )

row2_left, row2_right = st.columns(2)
with row2_left:
    render_card(
        "Model Testing & Comparison",
        "Upload CSV data and compare anomaly detection speed, accuracy, and results.",
        "pages/05_Model_Testing_and_Comparison.py",
        "Open model testing",
    )
with row2_right:
    render_card(
        "Alerts & Logs",
        "Review active alerts, acknowledge findings, and export the event history.",
        "pages/06_Alerts_Logs.py",
        "Open alerts",
    )

st.divider()

# ========================= FOOTER =========================
st.caption("© 2026 SCADA Anomaly Guard. Data is simulated for demonstration purposes.")
