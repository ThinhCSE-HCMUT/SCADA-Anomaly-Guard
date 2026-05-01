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
# Lấy dữ liệu live từ session_state (được sinh ra bởi engine)
df_live = st.session_state.get("history_data", pd.DataFrame())
is_running = st.session_state.get("is_monitoring", False)

if df_live.empty:
    # Thêm chút khoảng trống phía trên cho thoáng
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Cột ngoài cùng: Giữ cho khung thông báo rộng (tỷ lệ 0.5 : 10 : 0.5)
    col1, col2, col3 = st.columns([1, 7, 1])
    
    with col2:
        st.warning(
            "**No Active Data Stream**\n\n"
            "The system is waiting for data connection. Please start the simulation to view the overview dashboard.", 
        )
        
        # CỘT LỒNG CỘT: Tạo 3 cột nhỏ gọn bên trong col2 chỉ để bọc cái nút
        # Bạn có thể chỉnh tỷ lệ [3, 4, 3] thành [4, 4, 4] hoặc [1, 1, 1] để nút to/nhỏ tùy ý
        left_space, btn_col, right_space = st.columns([3, 2, 3])
        
        with btn_col:
            # Nút bấm giờ sẽ bị ép lại gọn gàng trong btn_col và nằm giữa màn hình
            if st.button("Go to Monitor", use_container_width=True):
                st.switch_page("pages/02_Real-time_Monitor.py") 
                
    st.stop()

# ====================== CALCULATE KPIs ======================
total_turbines = len(TARGET_TURBINES)
total_records = len(df_live)

# Lấy tổng số lỗi dựa trên dự đoán của Model
anomaly_records = int(df_live['pred_label'].sum()) if 'pred_label' in df_live.columns else 0
anomaly_rate = (anomaly_records / total_records) * 100 if total_records > 0 else 0

# Tính độ tự tin trung bình của Model
avg_score = df_live['anomaly_score'].mean() * 100 if 'anomaly_score' in df_live.columns else 0

# Tổng số alert hiện có trong session
total_alerts = len(st.session_state.get("anomaly_records", []))

# KPI Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Monitored Turbines",
        value=f"{total_turbines}",
    )

with col2:
    st.metric(
        label="Current Anomaly Rate",
        value=f"{anomaly_rate:.1f}%",
        delta="Warning" if anomaly_rate > 10 else "Stable",
        delta_color="inverse" if anomaly_rate > 10 else "normal"
    )

with col3:
    st.metric(
        label="Avg Anomaly Score",
        value=f"{avg_score:.1f}%",
        delta="High Risk" if avg_score > 60 else "Low Risk",
        delta_color="inverse" if avg_score > 60 else "normal"
    )

with col4:
    st.metric(
        label="Total Alerts (Session)",
        value=f"{total_alerts}",
        delta="Needs attention" if total_alerts > 0 else "All clear",
        delta_color="inverse" if total_alerts > 0 else "normal"
    )

st.divider()

# ====================== CHARTS ======================
st.subheader("Anomaly Rate by Turbine")

if not df_live.empty:
    # 1. Tính tổng số dòng và số lỗi của từng Turbine
    df_rate = df_live.groupby('asset_id').agg(
        total_records=('pred_label', 'count'),
        total_anomalies=('pred_label', 'sum')
    ).reset_index()
    
    # 2. Tính Tỷ lệ phần trăm (%)
    df_rate['anomaly_rate'] = (df_rate['total_anomalies'] / df_rate['total_records']) * 100
    
    # Map ID sang tên Turbine cho đẹp (Ví dụ: 0 -> "WT-00")
    df_rate['turbine_name'] = df_rate['asset_id'].map(TURBINE_LABELS)
    
    # 3. Vẽ Bar Chart
    fig = px.bar(
        df_rate, 
        x="turbine_name", 
        y="anomaly_rate", 
        title="Current Anomaly Rate per Turbine",
        template="plotly_dark",
        labels={"turbine_name": "Turbine", "anomaly_rate": "Anomaly Rate (%)"},
        text=df_rate['anomaly_rate'].apply(lambda x: f"{x:.1f}%") # Hiện con số % ngay trên đầu cột
    )
    
    # 4. Trang trí cột
    fig.update_traces(
        marker_color="rgba(239, 68, 68, 0.85)", # Màu đỏ báo lỗi
        textposition='outside', # Đẩy text lên trên đầu cột
        textfont=dict(color='white')
    )
    
    # 5. CỐ ĐỊNH TRỤC Y: Khóa trục Y để biểu đồ không bị "giật" khi data update
    fig.update_layout(
        height=450, # Có thể tăng height lên một chút (450) cho cân đối khi nó full width
        yaxis=dict(
            title="Anomaly Rate (%)",
            range=[0, 80], # Khóa cứng trục Y từ 0 đến 100% (có thể giảm xuống [0, 20] nếu ít lỗi)
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)"
        ),
        xaxis=dict(title=""), # Giấu chữ "Turbine" đi cho gọn
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    # Placeholder khi chưa có data
    st.info("💡 Waiting for simulation data...")

# ====================== SENSORS QUICK VIEW ======================
st.subheader("Average Sensor Readings (Latest Snapshot)")

if not df_live.empty:
    # 1. Tạo danh sách tất cả các sensor có trong SENSOR_GROUPS
    all_sensors = []
    for sensors in SENSOR_GROUPS.values():
        all_sensors.extend(sensors)
        
    # Chỉ giữ lại các sensor thực sự tồn tại trong dataset
    available_sensors = [s for s in all_sensors if s in df_live.columns]

    if SENSOR_GROUPS and available_sensors:
        # 2. Tính giá trị trung bình theo TỪNG MỐC THỜI GIAN để tìm xu hướng
        df_trend = df_live.groupby('time_stamp')[available_sensors].mean().sort_index()
        
        # Lấy giá trị của mốc thời gian MỚI NHẤT (hiện tại)
        latest_vals = df_trend.iloc[-1]
        
        # Lấy giá trị của mốc thời gian TRƯỚC ĐÓ (nếu đã có ít nhất 2 mốc dữ liệu)
        prev_vals = df_trend.iloc[-2] if len(df_trend) > 1 else None

        # 3. Render Tabs
        tab_names = ["All"] + list(SENSOR_GROUPS.keys())
        tabs = st.tabs(tab_names)
        
        for i, tab_name in enumerate(tab_names):
            with tabs[i]:
                if tab_name == "All":
                    valid_sensors = available_sensors
                else:
                    valid_sensors = [s for s in SENSOR_GROUPS[tab_name] if s in available_sensors]
                
                if valid_sensors:
                    cols = st.columns(3)
                    for j, sensor in enumerate(valid_sensors):
                        with cols[j % 3]:
                            label = get_sensor_label(sensor)
                            curr_val = latest_vals[sensor]
                            
                            # 4. Tính toán mức độ chênh lệch (Delta)
                            if prev_vals is not None:
                                delta_val = curr_val - prev_vals[sensor]
                                delta_str = f"{delta_val:.2f}"
                            else:
                                delta_str = None # Không có mốc cũ thì không hiện mũi tên
                            
                            # 5. Hiển thị Metric có chứa mũi tên
                            st.metric(
                                label=label, 
                                value=f"{curr_val:.2f}", 
                                delta=delta_str,
                                delta_color="normal" # Mặc định: Tăng = Xanh, Giảm = Đỏ
                            )
                else:
                    st.info("Không có dữ liệu cảm biến cho nhóm này.")
else:
    st.info("💡 Waiting for sensor data...")

st.divider()
st.caption("Tip: Click on 'Real-time Monitor' or 'Turbines List' in the sidebar to explore more details.")

# ====================== TRIGGER AUTO RERUN ======================
# Nếu công tắc đang bật, tiếp tục gọi engine cắt data và tự động refresh trang
if is_running:
    status = run_simulation_step()
    if status == "DONE":
        st.success("✅ Đã stream hết toàn bộ dữ liệu 5 turbines!")
    else:
        # Delay 1 chút để chống giật lag UI
        time.sleep(1) 
        st.rerun()