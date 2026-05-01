"""
Common Sidebar Component - Fixed header + correct navigation
"""

import streamlit as st
from src.config import APP_TITLE, APP_DESCRIPTION

def render_sidebar():
    """Render custom sidebar với header đẹp + điều hướng SPA (không F5)"""

    st.markdown("""
        <style>
        /* 1. Xóa giao diện mặc định của nút bấm Streamlit ngay dưới marker */
        div.element-container:has(#home-btn-marker) + div.element-container [data-testid="stButton"] button {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 12px 10px !important;
            border-radius: 10px !important;
            margin-bottom: 20px !important;
            transition: all 0.25s ease !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important; /* ĐÃ SỬA: Căn giữa toàn bộ nội dung (Title + Subtitle) */
            justify-content: center !important;
            height: auto !important;
            min-height: 0 !important;
            gap: 4px !important; /* Tăng khoảng cách giữa 2 dòng một chút cho thoáng */
            width: 100% !important;
        }
        
        /* 2. Hiệu ứng hover trượt ngang y hệt code cũ của bạn */
        div.element-container:has(#home-btn-marker) + div.element-container [data-testid="stButton"] button:hover {
            background-color: rgba(0, 0, 0, 0.15) !important;
            transform: translateX(6px) !important;
            color: inherit !important;
        }
        
        /* 3. Style cho dòng chữ chính "SCADA Anomaly Guard" */
        div.element-container:has(#home-btn-marker) + div.element-container [data-testid="stButton"] button p {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #000000 !important; 
            margin: 0 !important;
            line-height: 1.2 !important;
            text-align: center !important; /* ĐÃ SỬA: Đảm bảo khi chữ rớt dòng vẫn căn giữa */
        }
        
        /* 4. Tự động chèn dòng chữ phụ "Wind Turbine" xuống bên dưới */
        div.element-container:has(#home-btn-marker) + div.element-container [data-testid="stButton"] button::after {
            content: "Wind Turbine";
            font-size: 0.92rem;
            color: #a0a0a0;
            margin-top: 2px;
            font-weight: normal;
            text-align: center !important; /* ĐÃ SỬA: Căn giữa chữ phụ */
        }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # ====================== HEADER: LOGO + TÊN ======================
        # Căn giữa logo bằng các cột (Columns)
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2:
            st.image("assets/logo.png", use_container_width=True)

        # --- CSS HACK MARKER ---
        st.markdown('<div id="home-btn-marker"></div>', unsafe_allow_html=True)
        
        # Nút bấm tàng hình đã được CSS hóa phép thành Header
        if st.button("SCADA Anomaly Guard", use_container_width=True):
            st.switch_page("app.py") 

        st.divider()

        # ====================== NAVIGATION ======================
        st.subheader("Navigation")
        
        st.page_link("pages/01_Overview.py", label="Overview", use_container_width=True)
        st.page_link("pages/02_Real-time_Monitor.py", label="Real-time Monitor", use_container_width=True)
        st.page_link("pages/05_Model_Testing_and_Comparison.py", label="Model Testing & Comparison", use_container_width=True)
        st.page_link("pages/06_Alerts_Logs.py", label="Alerts & Logs", use_container_width=True)