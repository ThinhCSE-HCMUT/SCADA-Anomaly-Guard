import os
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

from src.config import (
    TARGET_TURBINES, TURBINE_LABELS, MODELS_DIR,
    SIMULATION_BATCH_SIZE, SIMULATION_MAX_HISTORY
)
from src.data_loader import load_data, split_by_turbine
from src.model_manager import load_model, load_ml_scaler
from src.inference import predict_batch

# --- HÀM BỔ TRỢ: CACHE DEEP LEARNING SCALER THEO TỪNG ASSET TRÁNH ĐỌC FILE LIÊN TỤC ---
@st.cache_resource
def load_dl_scaler_by_asset(turbine_id: int):
    """
    Tự động nạp scaler tương ứng với từng asset_id từ thư mục models/DeepLearning/DL_scaler
    """
    scaler_file_name = f"asset_{turbine_id}.pkl"
    # Định nghĩa đường dẫn chuẩn dựa vào MODELS_DIR từ config
    scaler_path = MODELS_DIR / "DeepLearning" / "DL_scaler" / scaler_file_name
    
    if scaler_path.exists():
        try:
            return joblib.load(scaler_path)
        except Exception as e:
            print(f"Error loading DL scaler for asset {turbine_id}: {e}")
            return None
    else:
        print(f"Warning: DL scaler not found at {scaler_path}")
        return None

# ====================== CORE SIMULATION ENGINE ======================
def run_simulation_step():
    """Hàm chạy 1 nhịp mô phỏng, hỗ trợ đồng thời ML Flow và DL Flow."""
    raw_df = load_data()
    if raw_df.empty:
        return "NO_DATA"

    turbine_dfs = split_by_turbine(raw_df)
    current_model_name = st.session_state.selected_model
    running_model = load_model(current_model_name)
    
    # Xác định loại luồng mô hình toàn cục
    is_ml_flow = current_model_name in ["XGBoost", "Random Forest"]
    
    # Nếu là ML Flow, dùng chung 1 scaler cho toàn bộ tuabin
    global_ml_scaler = None
    if is_ml_flow:
        global_ml_scaler = load_ml_scaler()
    
    new_rows = []
    any_active = False

    for tid in TARGET_TURBINES:
        if st.session_state.stream_done[tid]:
            continue

        step = st.session_state.turbine_steps[tid]
        batch = turbine_dfs[tid].iloc[
            step * SIMULATION_BATCH_SIZE : (step + 1) * SIMULATION_BATCH_SIZE
        ].copy()

        if batch.empty:
            st.session_state.stream_done[tid] = True
            continue

        # ✨ ĐÃ THÊM: Quyết định chọn bộ Scaler phù hợp cho lượt dự đoán này
        if is_ml_flow:
            current_scaler = global_ml_scaler
        else:
            # Nếu không phải ML (tức là thuộc nhóm LSTM, GRU, Hybrid...), nạp scaler động theo Asset ID
            current_scaler = load_dl_scaler_by_asset(tid)

        # Predict từ file inference (Luồng xử lý hình dáng shape 2D/3D đã được tự động hóa tại đây)
        pred_labels, pred_probas = predict_batch(running_model, batch, current_scaler)
        
        batch['pred_label'] = pred_labels
        batch['anomaly_score'] = pred_probas
        batch['model_used'] = current_model_name

        new_rows.append(batch)
        any_active = True
        st.session_state.turbine_steps[tid] += 1

        # Lưu log anomaly
        for _, r in batch[batch['pred_label'] == 1].iterrows():
            st.session_state.anomaly_records.append({
                "time"   : r['time_stamp'],
                "turbine": TURBINE_LABELS[tid],
                "score"  : round(float(r['anomaly_score']), 3),
            })

    # Cập nhật lịch sử
    if new_rows:
        st.session_state.history_data = pd.concat(
            [st.session_state.history_data, *new_rows],
            ignore_index=True,
        )

    # Trim history per turbine để không bị tràn RAM
    hist = st.session_state.history_data
    if len(hist) > SIMULATION_MAX_HISTORY * len(TARGET_TURBINES):
        st.session_state.history_data = (
            hist.groupby('asset_id', group_keys=False)
                .apply(lambda g: g.tail(SIMULATION_MAX_HISTORY))
                .reset_index(drop=True)
        )

    # Kiểm tra kết thúc
    if not any_active or all(st.session_state.stream_done.values()):
        st.session_state.is_monitoring = False
        return "DONE"
        
    return "RUNNING"