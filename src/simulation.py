import pandas as pd
import streamlit as st

from src.config import (
    TARGET_TURBINES, TURBINE_LABELS, 
    SIMULATION_BATCH_SIZE, SIMULATION_MAX_HISTORY
)
# Đã tách các hàm ra, giờ chỉ cần import vào để dùng
from src.data_loader import load_data, split_by_turbine
from src.model_manager import load_model
from src.inference import predict_batch

# ====================== CORE ENGINE ======================
def run_simulation_step():
    """Hàm chạy 1 nhịp mô phỏng, dùng chung cho mọi trang."""
    raw_df = load_data()
    if raw_df.empty:
        return "NO_DATA"

    turbine_dfs = split_by_turbine(raw_df)
    running_model = load_model(st.session_state.selected_model)
    
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

        # Predict từ file inference
        pred_labels, pred_probas = predict_batch(running_model, batch)
        batch['pred_label'] = pred_labels
        batch['anomaly_score'] = pred_probas
        batch['model_used'] = st.session_state.selected_model

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