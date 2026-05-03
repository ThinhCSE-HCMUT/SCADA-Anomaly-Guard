"""
Main entry point for the SCADA Anomaly Detection Dashboard.
This file only handles page config and main header.
Sidebar is now managed centrally in src/sidebar.py
"""

import streamlit as st
from src.config import APP_TITLE, APP_DESCRIPTION
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

# ========================= HERO SECTION =========================
col1, col2 = st.columns([3, 1])
with col1:
    st.title(f"{APP_TITLE}")
    st.markdown(f"**{APP_DESCRIPTION}**")

st.divider()

# ========================= GLOBAL KPI METRICS =========================
st.subheader("Live System Metrics")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Active Turbines", value="5 / 5", delta="All Online")
with kpi2:
    st.metric(label="Data Stream", value="Connected", delta="Real-time", delta_color="normal")
with kpi3:
    st.metric(label="Active Models", value=selected_model, delta="Running", delta_color="normal")
with kpi4:
    # Sau này có thể thay bằng count của anomaly_records
    st.metric(label="Unresolved Anomalies", value="0", delta="-2 from yesterday", delta_color="inverse")

st.markdown("<br>", unsafe_allow_html=True) # Khoảng trống cho thoáng

# ========================= QUICK ACCESS CARDS =========================
st.subheader("Quick Access")
st.markdown("Navigate to key modules to explore the system capabilities:")

# Đặt chiều cao cố định cho VÙNG CHỨA CHỮ (giúp các thẻ tự động cao bằng nhau)
# Nút st.page_link nằm bên ngoài vùng này sẽ luôn bị đẩy xuống đáy một cách đều đặn!
TEXT_HEIGHT = 220 

card1, card2, card3, card4 = st.columns(4)

with card1:
    with st.container(border=True):
        with st.container(height=TEXT_HEIGHT, border=False):
            st.markdown("### System Overview \n")
            st.write("\n")
            st.write("\n")
            st.write("\n\n\nView aggregated sensor readings, current health status, and anomaly rates across all wind turbines.")
        st.page_link("pages/01_Overview.py", label="Go to Overview")

with card2:
    with st.container(border=True):
        with st.container(height=TEXT_HEIGHT, border=False):
            st.markdown("### Real-time Monitor")
            st.write("Watch the live data stream, run the machine learning simulation, and spot anomalies as they happen.")
        st.page_link("pages/02_Real-time_Monitor.py", label="Go to Monitor")

with card3:
    with st.container(border=True):
        with st.container(height=TEXT_HEIGHT, border=False):
            st.markdown("### Model Testing & Comparison")
            st.write("Evaluate and compare the performance of various machine learning and deep learning models.")
        st.page_link("pages/05_Model_Testing_and_Comparison.py", label="Go to Testing & Comparison")

with card4:
    with st.container(border=True):
        with st.container(height=TEXT_HEIGHT, border=False):
            st.markdown("### Alerts & Logs")
            st.write("\n")
            st.write("\n")
            st.write("Review historical system alerts, acknowledge active warnings, and export event logs.")
        st.page_link("pages/06_Alerts_Logs.py", label="Go to Alerts")

st.divider()

# ========================= FOOTER =========================
st.caption("© 2026 SCADA Anomaly Guard. Data is simulated for demonstration purposes.")