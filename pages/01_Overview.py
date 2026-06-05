"""
Page 01: Overview Dashboard
Hiển thị KPIs cấp cao và trạng thái toàn hệ thống dựa trên luồng dữ liệu Live (Simulation).
"""

import time
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np


from src.sidebar import render_sidebar
from src.config import (
    STATUS_COLORS, 
    get_sensor_label,
    SENSOR_GROUPS,
    TARGET_TURBINES,
    TURBINE_LABELS
)
from src.simulation import run_simulation_step

st.set_page_config(
    page_title="System Overview", 
    layout="wide",                
    initial_sidebar_state="expanded" 
)

selected_model = render_sidebar()

st.title("System Overview")

# ====================== DATA LOADING ======================
history = st.session_state.get("history_data", pd.DataFrame())
current_det = st.session_state.get("current_detection_history_data", pd.DataFrame())
is_running = st.session_state.get("is_monitoring", False)

active_model = st.session_state.get("selected_model", "XGBoost").upper()
is_ml_flow = any(kw in active_model for kw in ["XGBOOST", "RANDOM FOREST"])

if not history.empty or not current_det.empty:
    chart_data_list = []
    
    if is_ml_flow:
        if not current_det.empty and "pred_label" in current_det.columns:
            for tid in TARGET_TURBINES:
                sub = current_det[current_det["asset_id"].astype(str).str.contains(str(tid))].copy()
                if not sub.empty:
                    anomalies = int(sub["pred_label"].sum())
                    total_points = len(sub)
                    # Tính toán tỷ lệ % lỗi (tránh chia cho 0)
                    anomaly_rate = (anomalies / total_points * 100) if total_points > 0 else 0.0
                    
                    chart_data_list.append({
                        "Turbine": TURBINE_LABELS[tid],
                        "Anomaly Rate (%)": round(anomaly_rate, 2),
                        "Anomaly Count": anomalies,
                        "Flow Type": "ML Current Detection"
                    })
    else:
        if not history.empty and "future_risk_label" in history.columns:
            for tid in TARGET_TURBINES:
                sub = history[history["asset_id"].eq(tid)].copy()
                if not sub.empty:
                    anomalies = int(sub["future_risk_label"].sum())
                    total_points = len(sub)
                    # Tính toán tỷ lệ % rủi ro (tránh chia cho 0)
                    anomaly_rate = (anomalies / total_points * 100) if total_points > 0 else 0.0
                    
                    chart_data_list.append({
                        "Turbine": TURBINE_LABELS[tid],
                        "Anomaly Rate (%)": round(anomaly_rate, 2),
                        "Anomaly Count": anomalies,
                        "Flow Type": "DL Future Risk"
                    })
                    
    if chart_data_list:
        anomaly_summary_df = pd.DataFrame(chart_data_list)
    else:
        anomaly_summary_df = pd.DataFrame(columns=["Turbine", "Anomaly Rate (%)", "Anomaly Count", "Flow Type"])

    # ------------------ HIỂN THỊ KPIs ------------------
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_text = "RUNNING" if is_running else "PAUSED"
        st.metric("Simulation Status", status_text, delta="Live Stream" if is_running else "Static")
        
    with col2:
        total_anomalies = int(anomaly_summary_df["Anomaly Count"].sum()) if not anomaly_summary_df.empty else 0
        flow_label = "ML Faults" if is_ml_flow else "DL Risks"
        st.metric(f"Total Detected ({flow_label})", f"{total_anomalies} pts")
        
    with col3:
        if not anomaly_summary_df.empty and total_anomalies > 0:
            highest_row = anomaly_summary_df.loc[anomaly_summary_df["Anomaly Rate (%)"].idxmax()]
            if highest_row["Anomaly Rate (%)"] > 0:
                st.metric("Highest Risk Asset", highest_row["Turbine"], delta=f"{highest_row['Anomaly Rate (%)']}% rate")
            else:
                st.metric("Highest Risk Asset", "None", delta="0%")
        else:
            st.metric("Highest Risk Asset", "None")
            
    with col4:
        st.metric("Active Pipeline Model", st.session_state.get("selected_model", "XGBoost"))

    st.divider()

    # ------------------ KHU VỰC ĐỒ THỊ ------------------
    graph_col, kpi_col = st.columns([2, 1])

    with graph_col:
        st.subheader("Anomaly Rate by Turbine")
        
        if not anomaly_summary_df.empty and total_anomalies > 0:
            fig = px.bar(
                anomaly_summary_df,
                x="Turbine",
                y="Anomaly Rate (%)",
                color="Turbine",
                title=f"Distribution of Anomalies Across Assets",
                template="plotly_dark",
                labels={"Anomaly Rate (%)": "Anomaly Rate (%)"},
                text=anomaly_summary_df["Anomaly Rate (%)"].apply(lambda x: f"{x}%"), 
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            # --- PHẦN THAY ĐỔI: ÉP KHUNG TỪ 0% ĐẾN 100% CỐ ĐỊNH ---
            fig.update_yaxes(
                ticksuffix="%",
                range=[0, 100]  # Giữ trục Y cố định không bị nhảy co giãn linh hoạt theo dữ liệu
            )
            fig.update_traces(textposition='outside') 
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Hệ thống đang vận hành ổn định, chưa ghi nhận điểm bất thường nào trên cả 5 Tuabin.")

    with kpi_col:
        st.subheader("Live Component Status")
        
        for tid in TARGET_TURBINES:
            t_rate = 0.0
            t_count = 0
            if not anomaly_summary_df.empty:
                t_match = anomaly_summary_df[anomaly_summary_df["Turbine"] == TURBINE_LABELS[tid]]
                if not t_match.empty:
                    t_rate = float(t_match["Anomaly Rate (%)"].iloc[0])
                    t_count = int(t_match["Anomaly Count"].iloc[0])
            
            if t_rate > 15.0: 
                status_color = "🔴 Critical"
                help_text = "High frequency of alerts. Immediate technical inspection recommended."
            elif t_rate > 2.0: 
                status_color = "🟡 Warning"
                help_text = "Minor anomalies or early sign of risk detected. Monitor parameters closely."
            else:
                status_color = "🟢 Normal"
                help_text = "Parameters are within safe threshold boundaries."
                
            with st.container(border=True):
                st.markdown(f"**{TURBINE_LABELS[tid]}** : {status_color}")
                st.caption(f"Rate: `{t_rate}%` ({t_count} pts) · {help_text}")

    st.divider()

    # ------------------ CHI TIẾT NHÓM CẢM BIẾN (SENSORS GROUP) ------------------
    st.subheader("Detailed Sensor Aggregations (Latest Stream)")
    
    if not history.empty:
        latest_data = history.sort_values("time_stamp").groupby("asset_id").last().reset_index()
        group_names = list(SENSOR_GROUPS.keys())
        tabs = st.tabs(group_names)
        
        for idx, g_name in enumerate(group_names):
            with tabs[idx]:
                sensors_in_g = SENSOR_GROUPS[g_name]
                available_sensors = [s for s in sensors_in_g if s in latest_data.columns]
                
                if available_sensors:
                    t_cols = st.columns(len(TARGET_TURBINES))
                    for t_idx, tid in enumerate(TARGET_TURBINES):
                        with t_cols[t_idx]:
                            st.markdown(f"##### {TURBINE_LABELS[tid]}")
                            
                            t_latest = latest_data[latest_data["asset_id"] == tid]
                            if not t_latest.empty:
                                for sensor in available_sensors:
                                    label = get_sensor_label(sensor)
                                    curr_val = t_latest[sensor].values[0]
                                    
                                    prev_vals = None
                                    sub_hist = history[history["asset_id"] == tid].sort_values("time_stamp")
                                    if len(sub_hist) > 1:
                                        prev_vals = sub_hist.iloc[-2]
                                    
                                    if prev_vals is not None and sensor in prev_vals:
                                        delta_val = curr_val - prev_vals[sensor]
                                        delta_str = f"{delta_val:.2f}"
                                    else:
                                        delta_str = None
                                    
                                    st.metric(
                                        label=label, 
                                        value=f"{curr_val:.2f}", 
                                        delta=delta_str,
                                        delta_color="normal"
                                    )
                            else:
                                st.caption("No data for this asset.")
                else:
                    st.info("Không có dữ liệu cảm biến cho nhóm này.")
else:
    st.info("💡 Waiting for sensor data... Please go to 'Real-time Monitor' page and click 'Start' to begin streaming.")

st.divider()
st.caption("Tip: Click on 'Real-time Monitor' or 'Turbines List' in the sidebar to explore more details.")

# ====================== TRIGGER AUTO RERUN ======================
if is_running:
    status = run_simulation_step()
    if status == "DONE":
        st.success("✅ Đã stream hết toàn bộ dữ liệu 5 turbines!")
    else:
        time.sleep(1) 
        st.rerun()