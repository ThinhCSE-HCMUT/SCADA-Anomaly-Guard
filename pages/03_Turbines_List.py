"""
Page 03: Machine List
Hiển thị danh sách tất cả các turbine gió với trạng thái hiện tại.
"""

import streamlit as st
import pandas as pd
from src.i18n import label_for_status, t
from src.sidebar import render_sidebar
from src.config import (
    STATUS_COLORS,
    get_sensor_label,
)

selected_model = render_sidebar()


st.title(t("turbines.title"))
st.markdown(f"### {t('turbines.description')}")

# ====================== FILTERS ======================
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    status_options = ["All", "Normal", "Warning", "Anomaly"]
    status_filter = st.selectbox(
        t("turbines.filter_status"),
        options=status_options,
        format_func=lambda status: t("turbines.all") if status == "All" else label_for_status(status),
        index=0
    )

with col2:
    search = st.text_input(t("turbines.search"), placeholder=t("turbines.search_placeholder"))

with col3:
    st.write("")  # spacer
    refresh_btn = st.button(t("turbines.refresh"), use_container_width=True)

# ====================== SAMPLE DATA (UI Demo) ======================
# Dữ liệu giả để demo giao diện - sau này sẽ thay bằng data thật từ backend
machines_data = {
    "Turbine_ID": [f"WT-{i:03d}" for i in range(1, 13)],
    "Location": ["Zone A", "Zone A", "Zone B", "Zone B", "Zone C", "Zone C",
                 "Zone A", "Zone B", "Zone C", "Zone A", "Zone B", "Zone C"],
    "Status": ["Normal", "Normal", "Warning", "Normal", "Anomaly", "Normal",
               "Normal", "Warning", "Normal", "Normal", "Normal", "Warning"],
    "Anomaly_Score": [0.12, 0.18, 0.68, 0.25, 0.92, 0.31, 0.09, 0.55, 0.22, 0.14, 0.27, 0.61],
    "Last_Updated": [t("turbines.sample_last_updated")] * 12,
    "Power_Output": [2450, 2310, 980, 2670, 420, 1890, 2540, 1120, 2380, 2490, 2210, 1350],  # kW
}

df_machines = pd.DataFrame(machines_data)

# Apply filter
if status_filter != "All":
    df_machines = df_machines[df_machines["Status"] == status_filter]

if search:
    df_machines = df_machines[
        df_machines["Turbine_ID"].str.contains(search, case=False) |
        df_machines["Location"].str.contains(search, case=False)
    ]

# ====================== DISPLAY MACHINES AS CARDS ======================
st.subheader(t("turbines.showing", count=len(df_machines)))

cols = st.columns(3)

for idx, row in df_machines.iterrows():
    with cols[idx % 3]:
        status_color = STATUS_COLORS.get(row["Status"], "#6b7280")
        
        with st.container(border=True):
            st.markdown(f"**{row['Turbine_ID']}**")
            st.caption(row["Location"])
            
            # Status badge
            st.markdown(f"""
                <div style="background-color: {status_color}20; 
                           color: {status_color}; 
                           padding: 4px 12px; 
                           border-radius: 20px; 
                           font-weight: bold; 
                           display: inline-block;">
                    ● {label_for_status(row['Status'])}
                </div>
            """, unsafe_allow_html=True)
            
            st.metric(
                label=t("turbines.anomaly_score"),
                value=f"{row['Anomaly_Score']:.2f}",
                delta=None
            )
            
            st.metric(
                label=t("turbines.power_output"),
                value=f"{row['Power_Output']} kW"
            )
            
            st.caption(t("turbines.last_updated", value=row['Last_Updated']))
            
            if st.button(t("turbines.view_detail"), key=f"detail_{idx}"):
                st.session_state.selected_turbine = row["Turbine_ID"]
                st.switch_page("pages/02_Real-time_Monitor.py")  # Có thể chuyển sang trang monitor

st.divider()
st.info(f"💡 {t('turbines.tip')}")
