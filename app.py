"""
Main entry point for the SCADA Anomaly Detection Dashboard.
This file only handles page config and main header.
Sidebar is now managed centrally in src/sidebar.py
"""

import streamlit as st
from src.config import APP_TITLE, APP_DESCRIPTION, DEFAULT_MODEL, AVAILABLE_MODELS, TARGET_TURBINES
from src.i18n import get_language, t
from src.sidebar import render_sidebar

# ========================= PAGE CONFIG =========================
get_language()
st.set_page_config(
    page_title=t("app.title"),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': f"{t('app.description')}\nDeveloped for Wind Turbine SCADA Anomaly Detection"
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
st.markdown(f"**{t('app.description')}**")
st.caption(
    f"{t('home.monitoring')}: {t('status.running') if is_running else t('status.idle')}  |  "
    f"{t('home.active_model')}: {active_model}  |  "
    f"{t('home.session_alerts')}: {alert_count}"
)

st.divider()

# ========================= GLOBAL KPI METRICS =========================
st.subheader(t("home.snapshot"))
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label=t("home.active_turbines"), value=f"{fleet_size} / {fleet_size}", delta=t("status.fleet_ready"))
with kpi2:
    st.metric(
        label=t("home.monitoring"),
        value=t("status.running") if is_running else t("status.idle"),
        delta=t("status.live_stream") if is_running else t("status.standby_delta"),
        delta_color="normal" if is_running else "off",
    )
with kpi3:
    st.metric(label=t("home.active_model"), value=active_model, delta=f"{len(AVAILABLE_MODELS)} {t('status.available')}")
with kpi4:
    st.metric(label=t("home.session_alerts"), value=f"{alert_count}", delta=t("status.logged_anomalies"))

# ========================= QUICK ACCESS CARDS =========================
st.subheader(t("home.quick_access"))
st.caption(t("home.quick_access_caption"))


def render_card(title: str, body: str, page_path: str, action_label: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.write(body)
        st.page_link(page_path, label=action_label, width="stretch")


row1_left, row1_right = st.columns(2)
with row1_left:
    render_card(
        t("home.fleet_overview_title"),
        t("home.fleet_overview_body"),
        "pages/01_Overview.py",
        t("home.fleet_overview_action"),
    )
with row1_right:
    render_card(
        t("home.monitor_title"),
        t("home.monitor_body"),
        "pages/02_Real-time_Monitor.py",
        t("home.monitor_action"),
    )

row2_left, row2_right = st.columns(2)
with row2_left:
    render_card(
        t("home.testing_title"),
        t("home.testing_body"),
        "pages/05_Model_Testing_and_Comparison.py",
        t("home.testing_action"),
    )
with row2_right:
    render_card(
        t("home.alerts_title"),
        t("home.alerts_body"),
        "pages/06_Alerts_Logs.py",
        t("home.alerts_action"),
    )

st.divider()

# ========================= FOOTER =========================
st.caption(t("home.footer"))
