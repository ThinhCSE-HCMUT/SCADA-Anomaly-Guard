"""
Page 06: Alerts & Logs
Trang hiển thị lịch sử cảnh báo và log hệ thống
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from src.sidebar import render_sidebar
from src.config import STATUS_COLORS, TURBINE_LABELS

st.set_page_config(
    page_title="Alert Logs",  # Tên hiển thị trên tab trình duyệt
    layout="wide",                 # 🔥 CHÌA KHÓA: Ép trang luôn ở chế độ Full Screen
    initial_sidebar_state="expanded" # (Tùy chọn) Ép sidebar luôn mở ra
)

# ====================== INIT STATE ======================
# Thêm dòng này vào khu vực INIT STATE ở đầu file
if "alert_page" not in st.session_state:
    st.session_state.alert_page = 1
    
if "acknowledged_alerts" not in st.session_state:
    st.session_state.acknowledged_alerts = set()

if "anomaly_records" not in st.session_state:
    st.session_state.anomaly_records = []

if "system_logs" not in st.session_state:
    st.session_state.system_logs = [
        {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Event": "System initialized and dashboard loaded"}
    ]
def add_system_log(event_message):
    """Hàm tiện ích để ghi lại sự kiện mới vào hệ thống"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Chèn log mới vào đầu danh sách (index 0) để lúc nào cũng thấy log mới nhất trước
    st.session_state.system_logs.insert(0, {"Timestamp": timestamp, "Event": event_message})

selected_model = render_sidebar()

st.title("Alerts & Logs")
st.markdown("### System Alerts and Event History")

# ====================== DATA PROCESSING ======================
# Lấy dữ liệu log từ hệ thống mô phỏng
df_alerts = pd.DataFrame(st.session_state.anomaly_records)

# Nếu có dữ liệu, xử lý thêm một số cột để map với UI
if not df_alerts.empty:
    df_alerts['time'] = pd.to_datetime(df_alerts['time'])
    # Tạo ID duy nhất cho mỗi alert để xử lý nút Acknowledge
    df_alerts['alert_id'] = df_alerts['turbine'] + "_" + df_alerts['time'].astype(str)
    
    # Loại bỏ những alert đã được acknowledge
    df_alerts = df_alerts[~df_alerts['alert_id'].isin(st.session_state.acknowledged_alerts)]
    
    # Sắp xếp mới nhất lên đầu
    df_alerts = df_alerts.sort_values('time', ascending=False)


# ====================== FILTERS ======================
col1, col2, col3 = st.columns([2, 2, 2])

with col1:
    alert_type = st.selectbox(
        "Alert Type",
        options=["All Alerts", "Anomaly"] # Hiện tại engine chỉ log Anomaly
    )

with col2:
    time_range = st.selectbox(
        "Show Events",
        options=["Last 50 events", "Last 200 events", "All Time"],
        index=0
    )

with col3:
    # Lấy danh sách turbine linh động từ file config
    turbine_options = ["All Turbines"] + list(TURBINE_LABELS.values())
    turbine_filter = st.selectbox("Turbine", options=turbine_options)

# Apply filters
if not df_alerts.empty:
    if turbine_filter != "All Turbines":
        df_alerts = df_alerts[df_alerts["turbine"] == turbine_filter]
        
    if time_range == "Last 50 events":
        df_alerts = df_alerts.head(50)
    elif time_range == "Last 200 events":
        df_alerts = df_alerts.head(200)

# ====================== DISPLAY ALERTS ======================
st.subheader(f"Recent Alerts ({len(df_alerts)} active events)")

# ĐỊNH NGHĨA FRAGMENT: Độc lập hóa khu vực này để không bị F5 cả trang
def render_alert_card(alert):
    alert_id = alert['alert_id']
    
    # 1. Trạng thái ĐÃ XÁC NHẬN -> Hiện thẻ màu xanh mỏng, gọn gàng
    if alert_id in st.session_state.acknowledged_alerts:
        st.markdown(
            f"""
            <div style="color: #22c55e; background-color: rgba(34, 197, 94, 0.05); 
                        border-left: 4px solid #22c55e; border-radius: 4px; 
                        padding: 10px 15px; margin-bottom: 1rem;">
                <span style="font-weight: bold; font-size: 1.1em; margin-right: 8px;">✓</span> 
                Acknowledged: <b>{alert['turbine']}</b> issue resolved at {alert['time'].strftime('%H:%M:%S')}.
            </div>
            """, 
            unsafe_allow_html=True
        )
        return # Dừng vẽ thẻ đỏ ở dưới

    # 2. Trạng thái CHƯA XÁC NHẬN -> Hiện thẻ cảnh báo (Alert Card)
    level = "Anomaly" 
    color = STATUS_COLORS.get(1, "#ef4444")
    
    with st.container(border=True):
        cols = st.columns([1.5, 2.5, 2, 2, 1.5])
        
        with cols[0]:
            st.markdown(f"**{alert['time'].strftime('%Y-%m-%d %H:%M:%S')}**")
        with cols[1]:
            st.write(f"**{alert['turbine']}**")
        with cols[2]:
            st.markdown(f"""
                <span style="color:{color}; font-weight:bold;">● {level}</span>
            """, unsafe_allow_html=True)
        with cols[3]:
            st.write(f"Score: **{alert['score']:.2f}**")
        with cols[4]:
            # Tạo một không gian trống (placeholder) để nhét nút vào
            action_ph = st.empty()
            
            # Vẽ nút vào placeholder
            if action_ph.button("Acknowledge", key=f"btn_{alert_id}", use_container_width=True):
                # KHI BẤM NÚT: Thay thế cái nút bằng dòng chữ báo thành công
                action_ph.markdown(
                    """
                    <div style='text-align: center; background-color: #dcfce7; color: #16a34a; 
                                border-radius: 5px; padding: 5px 0; border: 1px solid #22c55e;'>
                        <b style='font-size: 1.2em;'>✓</b> Đã acknowledge
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Cập nhật Data & Log
                st.session_state.acknowledged_alerts.add(alert_id)
                add_system_log(f"Alert Acknowledged: Anomaly cleared for {alert['turbine']} (Score: {alert['score']:.2f})")
                
                # Dừng hình 2 giây cho user kịp nhìn UI/UX
                time.sleep(2)
                
                # Ép Streamlit reload trang để cập nhật lại danh sách phân trang
                st.rerun() 
                
        st.caption(f"System detected abnormal behavior exceeding threshold. Confidence score is {alert['score']:.2f}.")
# XUẤT RA GIAO DIỆN
if df_alerts.empty:
    if len(st.session_state.anomaly_records) == 0:
        st.info("Waiting for simulation data... No anomalies detected yet.")
    else:
        st.markdown("""
            <div style="color: #22c55e; padding: 15px; border: 1px solid #22c55e; border-radius: 5px;">
                All clear! All anomalies have been acknowledged or filtered out.
            </div>
        """, unsafe_allow_html=True)
else:
    # ====================== PAGINATION CALCULATION ======================
    ITEMS_PER_PAGE = 10
    total_items = len(df_alerts)
    total_pages = (total_items - 1) // ITEMS_PER_PAGE + 1 
    
    # Đảm bảo page hiện tại không bị lố nếu dữ liệu bị xóa bớt (Acknowledge)
    if st.session_state.alert_page > total_pages and total_pages > 0:
        st.session_state.alert_page = total_pages
        
    current_page = st.session_state.alert_page
    
    # Cắt (Slice) DataFrame theo số trang hiện tại
    start_idx = (current_page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    paginated_df = df_alerts.iloc[start_idx:end_idx]

    st.caption(f"Showing **{start_idx + 1} - {min(end_idx, total_items)}** of **{total_items}** active events.")

    # 1. VẼ CÁC ALERT CARD TRƯỚC (NẰM Ở TRÊN)
    for _, alert in paginated_df.iterrows():
        render_alert_card(alert)

    # 2. VẼ THANH PAGINATION Ở DƯỚI CÙNG (Chỉ hiện khi > 10 items tương đương > 1 page)
    if total_pages > 1:
        st.markdown("<br>", unsafe_allow_html=True) # Thêm chút khoảng trống cho thoáng
        
        # Tạo lưới cột để căn giữa thanh pagination giống mẫu ảnh của bạn
        _, col_prev, col_p1, col_p2, col_p3, col_next, _ = st.columns([2, 1.5, 1, 1, 1, 1.5, 2])
        
        # --- Nút Previous ---
        with col_prev:
            if st.button("◀ Prev", disabled=(current_page == 1), use_container_width=True):
                st.session_state.alert_page -= 1
                st.rerun()
                
        # --- Tính toán 3 số trang hiển thị ở giữa (Windowing) ---
        page_window = []
        if total_pages <= 3:
            page_window = list(range(1, total_pages + 1))
        else:
            if current_page == 1:
                page_window = [1, 2, 3]
            elif current_page == total_pages:
                page_window = [total_pages - 2, total_pages - 1, total_pages]
            else:
                page_window = [current_page - 1, current_page, current_page + 1]

        # --- Các nút số trang [1] [2] [3] ---
        # Map các cột vừa tạo vào 1 list để nhét nút vào cho gọn
        num_cols = [col_p1, col_p2, col_p3] 
        for i, page_num in enumerate(page_window):
            with num_cols[i]:
                # Nút đang ở trang hiện tại thì cho màu nổi lên (type="primary")
                btn_type = "primary" if page_num == current_page else "secondary"
                if st.button(str(page_num), key=f"page_{page_num}", type=btn_type, use_container_width=True):
                    st.session_state.alert_page = page_num
                    st.rerun()
                    
        # --- Nút Next ---
        with col_next:
             if st.button("Next ▶", disabled=(current_page == total_pages), use_container_width=True):
                st.session_state.alert_page += 1
                st.rerun()

st.divider()

# ====================== LOG HISTORY ======================
st.subheader("System Log")

# Chuyển đổi list dictionary thành DataFrame để hiển thị đẹp hơn
if st.session_state.system_logs:
    df_log = pd.DataFrame(st.session_state.system_logs)
else:
    # Fallback dự phòng nếu chưa có log nào
    df_log = pd.DataFrame(columns=["Timestamp", "Event"])

st.dataframe(df_log, use_container_width=True, hide_index=True)

# Thêm nút Export cho có cảm giác chuyên nghiệp
if not df_log.empty:
    csv = df_log.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Log Report",
        data=csv,
        file_name=f"scada_system_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

st.caption(f"Total events logged: {len(df_log)}. All alerts are logged and can be exported for reporting.")