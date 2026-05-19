"""
Page 05: Model Testing and Comparison
Tích hợp Backend xử lý dữ liệu thật, Scale, Dự đoán và So sánh các model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import plotly.express as px
from xgboost import XGBClassifier
from src.model_manager import load_model
import os
import json
from pathlib import Path

from src.sidebar import render_sidebar
from src.config import AVAILABLE_MODELS, CHART_SENSOR_COLS, DEFAULT_TABLE_ROWS, MODEL_THRESHOLDS, XGBOOST_FORECAST_OPTIONS, XGBOOST_FORECAST_MODEL_PATHS, RF_FORECAST_OPTIONS, RF_FORECAST_MODEL_PATHS, DL_FORECAST_MODEL_PATHS, FEATURE_COLS
from tensorflow.keras.models import load_model as keras_load_model
st.set_page_config(
    page_title="Model Testing and Comparison", 
    layout="wide",                 
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_buffer():
    return pd.read_csv("train_buffer.csv")

def load_backend_artifacts():
    try:
        # 2. Load TẤT CẢ Models tự động thông qua model_manager
        loaded_models = {}
        
        # Lặp qua tất cả các tên model (XGBoost, Random Forest, LSTM, GRU...)
        for model_name in AVAILABLE_MODELS.keys():
            # Truyền ĐÚNG tên model (model_name) vào hàm load_model
            model = load_model(model_name)
            
            # Nếu load thành công thì đưa vào dictionary
            if model is not None:
                loaded_models[model_name] = model
            else:
                st.warning(f"Không thể nạp được model: {model_name}. Vui lòng kiểm tra lại file.")
                
        # Load các phiên bản XGBoost theo từng horizon dự báo sớm
        xgb_forecast_models = {}
        for horizon, path in XGBOOST_FORECAST_MODEL_PATHS.items():
            if os.path.exists(path):
                try:
                    xgb_forecast_models[horizon] = joblib.load(path)
                except Exception as e:
                    st.warning(f"Không thể nạp XGBoost {horizon} từ '{path}': {e}")
            else:
                st.warning(f"Không tìm thấy file XGBoost {horizon}: {path}")

        if xgb_forecast_models:
            if "XGBoost" in loaded_models and not isinstance(loaded_models["XGBoost"], dict):
                xgb_forecast_models.setdefault("Current", loaded_models["XGBoost"])
            loaded_models["XGBoost"] = xgb_forecast_models

        # Load các phiên bản Random Forest theo từng horizon dự báo sớm
        rf_forecast_models = {}
        for horizon, path in RF_FORECAST_MODEL_PATHS.items():
            if os.path.exists(path):
                try:
                    rf_forecast_models[horizon] = joblib.load(path)
                except Exception as e:
                    st.warning(f"Không thể nạp Random Forest {horizon} từ '{path}': {e}")
            else:
                st.warning(f"Không tìm thấy file Random Forest {horizon}: {path}")

        if rf_forecast_models:
            if "Random Forest" in loaded_models and not isinstance(loaded_models["Random Forest"], dict):
                rf_forecast_models.setdefault("Current", loaded_models["Random Forest"])
            loaded_models["Random Forest"] = rf_forecast_models

        # Load các phiên bản Deep Learning theo từng horizon dự báo sớm
        for model_name, horizon_paths in DL_FORECAST_MODEL_PATHS.items():
            dl_models = {}
            for horizon, path in horizon_paths.items():
                if os.path.exists(path):
                    try:
                        dl_models[horizon] = keras_load_model(path)
                    except Exception as e:
                        st.warning(f"Không thể nạp {model_name} {horizon} từ '{path}': {e}")
                else:
                    st.warning(f"Không tìm thấy file {model_name} {horizon}: {path}")
            if dl_models:
                if model_name in loaded_models and not isinstance(loaded_models[model_name], dict):
                    dl_models.setdefault("Current", loaded_models[model_name])
                loaded_models[model_name] = dl_models
        
        # (Riêng XGBoost nếu bạn bắt buộc dùng file .json thay vì .pkl thì nạp thủ công ở đây)
        xgb_json_path = "models/Baseline/XGBoost/xgb_model.json"
        if "XGBoost" not in loaded_models and os.path.exists(xgb_json_path):
            xgb_model = XGBClassifier()
            xgb_model.load_model(xgb_json_path)
            loaded_models["XGBoost"] = xgb_model
        
        return None, loaded_models
        
    except Exception as e:
        st.error(f"Lỗi khi load models: {e}")
        return None, None

def apply_rolling_window(df_raw, windows=[3, 6]):
    """Rolling window cho ML - Tính trên dữ liệu đã ghép buffer + upload"""
    df_out = df_raw.copy()
    
    # Sort lại để đảm bảo thứ tự thời gian (an toàn)
    if 'asset_id' in df_out.columns and 'time_stamp' in df_out.columns:
        df_out = df_out.sort_values(by=['asset_id', 'time_stamp']).reset_index(drop=True)

    cols_to_drop = ['time_stamp', 'asset_id', 'label', 'train_test', 'status_type_id', 'sequence_id', 'is_buffer']
    sensor_cols = [col for col in df_out.columns 
                   if col not in cols_to_drop 
                   and '_mean_' not in col 
                   and '_std_' not in col]

    # Ép kiểu float32
    for col in sensor_cols:
        df_out[col] = df_out[col].astype(np.float32)

    rolling_features = []

    for w in windows:
        if 'asset_id' in df_out.columns:
            grouped = df_out.groupby('asset_id')[sensor_cols]
        else:
            grouped = df_out[sensor_cols]

        # Tính rolling mean và std
        roll_mean = grouped.rolling(window=w, min_periods=1).mean()
        roll_std  = grouped.rolling(window=w, min_periods=1).std()

        # Reset index để align lại với df_out
        roll_mean = roll_mean.reset_index(level=0, drop=True)
        roll_std  = roll_std.reset_index(level=0, drop=True)

        roll_mean.columns = [f'{col}_mean_{w}' for col in sensor_cols]
        roll_std.columns  = [f'{col}_std_{w}' for col in sensor_cols]

        rolling_features.extend([roll_mean, roll_std])

    # Ghép rolling features vào
    df_rolling = pd.concat(rolling_features, axis=1)
    df_out = pd.concat([df_out, df_rolling], axis=1)

    # Chỉ bfill cho những vị trí thật sự thiếu (thường chỉ vài dòng đầu của buffer)
    df_out = df_out.bfill()

    return df_out

def shift_labels_for_forecast(df, forecast_horizon_hours: str):
    """
    Shift nhãn (label) theo horizon dự báo sớm.
    Ví dụ: horizon='12h' → shift nhãn 12h về trước (72 steps with rows_per_hour=6)
    """
    if forecast_horizon_hours == "Current" or 'label' not in df.columns:
        return df
    
    df_shifted = df.copy()
    
    try:
        hours = int(forecast_horizon_hours.replace('h', ''))
    except ValueError:
        st.warning(f"Không thể parse horizon '{forecast_horizon_hours}'. Sử dụng nhãn hiện tại.")
        return df_shifted
    
    if hours == 0:
        return df_shifted
    
    rows_per_hour = 6
    shift_steps = hours * rows_per_hour
    
    if 'asset_id' in df_shifted.columns:
        df_shifted['label'] = df_shifted.groupby('asset_id')['label'].shift(-shift_steps)
    else:
        df_shifted['label'] = df_shifted['label'].shift(-shift_steps)
    
    initial_len = len(df_shifted)
    df_shifted = df_shifted.dropna(subset=['label']).reset_index(drop=True)
    
    return df_shifted


def validate_uploaded_csv(df):
    required_cols = ["time_stamp", "asset_id"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        return False, (
            "Uploaded CSV must contain the following columns: `time_stamp`, `asset_id`. "
            f"Missing columns: {', '.join(missing_required)}."
        )

    missing_features = [col for col in CHART_SENSOR_COLS if col not in df.columns]
    if missing_features:
        return False, (
            "Uploaded CSV does not match the expected model input format. "
            "Please upload a CSV with all required feature columns used during model training. "
            f"Missing columns: {', '.join(missing_features[:10])}{'...' if len(missing_features) > 10 else ''}"
        )

    return True, None

df_buffer = load_buffer()

# ====================== GIAO DIỆN CHÍNH ======================
selected_model_sidebar = render_sidebar()

st.title("Model Testing & Comparison")
st.markdown("### Run Anomaly Detection and Compare Performance")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload SCADA Data (CSV)",
        type=["csv"],
    )
    st.info(
        "Please upload a CSV file with the same feature names used by the trained models. "
        "Required columns include `time_stamp`, `asset_id`, and the model feature columns."
    )
    with st.expander("CSV format guidance", expanded=False):
        st.markdown(
            "- Required metadata columns: `time_stamp`, `asset_id`.\n"
            "- Required feature columns: same names used during training, for example `sensor_0_avg`, `sensor_5_avg_sin`, `sensor_5_avg_cos`, ...\n"
            "- Use `data/sample_data.csv` as a reference template if available."
        )

with col2:
    model_options = list(AVAILABLE_MODELS.keys())
        
    selected_models = st.multiselect(
        "Select Models to Test",
        options=model_options,
    )

    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        options=list(XGBOOST_FORECAST_OPTIONS.keys()),
        index=0,
        help="Chọn Current hoặc dự báo sớm. Áp dụng cho XGBoost và Random Forest."
    )

# ====================== PROCESS DATA ======================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'time_stamp' in df.columns:
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        
    valid_format, validation_message = validate_uploaded_csv(df)
    if not valid_format:
        st.error(validation_message)
        st.stop()

    if 'asset_id' in df.columns and 'time_stamp' in df.columns:
        df = df.sort_values(by=['asset_id', 'time_stamp']).reset_index(drop=True)
    
    st.success(f"Loaded {len(df)} records from uploaded file")

    st.subheader("Data Preview")
    st.dataframe(df.head(DEFAULT_TABLE_ROWS), use_container_width=True)

    col_left, col_mid, col_right = st.columns([1, 0.5, 1])
    with col_mid:
        run_button = st.button("Run Prediction", type="primary", use_container_width=True)
        
    if run_button:
        if not selected_models:
            st.warning("Please select at least one model!")
            st.stop()

        with st.spinner("Loading stored model performance..."):
            perf_path = Path("models") / "model_performance.json"
            if perf_path.exists():
                try:
                    model_perf = json.loads(perf_path.read_text())
                except Exception as e:
                    st.error(f"Cannot read performance file: {e}")
                    model_perf = {}
            else:
                st.warning("No saved performance file found at models/model_performance.json")
                model_perf = {}

            results = []
            raw_metrics_for_plot = []

            for model_name in selected_models:
                display_model_name = model_name
                if model_name in ["XGBoost", "Random Forest", "LSTM", "GRU", "CNN - LSTM", "CNN - GRU"]:
                    display_model_name = f"{model_name} ({forecast_horizon})"

                perf_entry = model_perf.get(model_name, {}).get(forecast_horizon, {}) if model_perf else {}

                def fmt(x):
                    return f"{x:.2f}" if isinstance(x, (int, float)) else (str(x) if x is not None and x != "" else "N/A")

                model_result = {
                    "Model": display_model_name,
                    "Accuracy": fmt(perf_entry.get("accuracy")),
                    "F1-Score": fmt(perf_entry.get("f1")),
                    "Precision": fmt(perf_entry.get("precision")),
                    "Recall": fmt(perf_entry.get("recall")),
                    "PR-AUC": fmt(perf_entry.get("pr_auc")),
                    "ROC-AUC": fmt(perf_entry.get("roc_auc")),
                }
                results.append(model_result)
                
                # ✨ ĐOẠN SỬA ĐỔI: Đẩy đầy đủ cả 6 metrics thực tế vào danh sách để vẽ chart tổng hợp
                raw_metrics_for_plot.append({
                    "Model": display_model_name,
                    "Accuracy": perf_entry.get("accuracy", 0.0),
                    "F1-Score": perf_entry.get("f1", 0.0),
                    "Precision": perf_entry.get("precision", 0.0),
                    "Recall": perf_entry.get("recall", 0.0),
                    "PR-AUC": perf_entry.get("pr_auc", 0.0),
                    "ROC-AUC": perf_entry.get("roc_auc", 0.0)
                })

            st.success("Prediction completed!")
            st.toast("Model prediction completed successfully!", icon="✅")
            
            # --- TỔNG HỢP KẾT QUẢ ---
            st.subheader("Results Summary")
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # --- BIỂU ĐỒ CHI TIẾT TỪNG MODEL ---
            st.subheader("Detailed Prediction Dashboard")
            
            # Định nghĩa bảng màu đồng bộ toàn cục (Precision dùng màu vàng chanh #FFFF08)
            metrics_list = ["Accuracy", "F1-Score", "Precision", "Recall", "PR-AUC", "ROC-AUC"]
            metric_colors = {
                "Accuracy": "#10b981",    # Xanh Emerald rực rỡ
                "F1-Score": "#0ea5e9",    # Xanh Ocean thanh lịch
                "Precision": "#CFCC4E",   # ✨ Cập nhật màu Vàng Chanh chuẩn rực rỡ
                "Recall": "#636efa",      # Tím Indigo hoàng gia
                "PR-AUC": "#a855f7",      # Tím Purple huyền bí
                "ROC-AUC": "#ec4899"      # Hồng Neon đậm chất dữ liệu
            }

            for model_name in selected_models:
                display_model_name = model_name
                label_shifted = False
                if model_name in ["XGBoost", "Random Forest", "LSTM", "GRU", "CNN - LSTM", "CNN - GRU"]:
                    display_model_name = f"{model_name} ({forecast_horizon})"
                    label_shifted = (forecast_horizon != "Current")

                with st.expander(f"Detailed Results: {display_model_name}", expanded=True):
                    model_row = next((r for r in results if r["Model"] == display_model_name), None)
                    c1, c2 = st.columns([1.2, 2.8])
                    
                    def to_float(v):
                        try:
                            if v is None or str(v).strip().upper() in ["N/A", "NONE", "NULL"]:
                                return None
                            return float(v)
                        except Exception:
                            return None

                    with c1:
                        total_samples = len(df) if 'df' in locals() else "N/A"
                        if isinstance(total_samples, int):
                            total_samples_str = f"{total_samples:,}"
                        else:
                            total_samples_str = str(total_samples)

                        st.markdown(f"""
                            <div style="background-color: #1e293b; border-radius: 6px; padding: 6px 12px; margin-bottom: 8px; border-left: 4px solid #94a3b8;">
                                <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Total Samples</div>
                                <div style="font-size: 18px; color: #f8fafc; font-weight: 700; line-height: 1.2;">{total_samples_str}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if model_row is not None:
                            for metric in metrics_list:
                                raw_val = model_row.get(metric, "N/A")
                                val_float = to_float(raw_val)
                                color = metric_colors.get(metric, "#636efa")
                                
                                if val_float is not None:
                                    pct = max(0.0, min(100.0, val_float * 100))
                                    val_str = f"{val_float:.3f}"
                                else:
                                    pct = 0
                                    val_str = "N/A"
                                
                                st.markdown(f"""
                                    <div style="background-color: #1e293b; border-radius: 6px; padding: 6px 12px; margin-bottom: 7px; border-left: 4px solid {color}; position: relative;">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <span style="font-size: 12px; color: #cbd5e1; font-weight: 500;">{metric}</span>
                                            <span style="font-size: 14px; color: #ffffff; font-weight: 700;">{val_str}</span>
                                        </div>
                                        <div style="background-color: #334155; border-radius: 2px; height: 4px; width: 100%; margin-top: 5px; overflow: hidden;">
                                            <div style="background-color: {color}; height: 100%; width: {pct}%; border-radius: 2px; transition: width 0.6s ease-in-out;"></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No saved performance for this model/horizon.")

                    with c2:
                        if model_row is not None:
                            chart_data = []
                            for metric in metrics_list:
                                val = to_float(model_row.get(metric, None))
                                chart_data.append({
                                    "Metric": metric,
                                    "Value": val if val is not None else 0.0,
                                    "Label": f"{val:.3f}" if val is not None else "N/A"
                                })
                            
                            chart_df = pd.DataFrame(chart_data)

                            fig = px.bar(
                                chart_df,
                                x="Value",
                                y="Metric",
                                orientation='h',
                                text="Label",
                                color="Metric",
                                color_discrete_map=metric_colors
                            )
                            
                            fig.update_layout(
                                template="plotly_dark",
                                height=360, 
                                margin=dict(t=5, b=5, l=10, r=10),
                                xaxis=dict(range=[0, 1.05], title="Score Value (0.0 - 1.0)", gridcolor="#334155"),
                                yaxis=dict(
                                    title="", 
                                    autorange="reversed",
                                    categoryorder="array",
                                    categoryarray=metrics_list
                                ),
                                showlegend=False
                            )
                            
                            fig.update_traces(
                                textposition='inside', 
                                insidetextanchor='middle',
                                textfont=dict(size=12, color="white", family="Arial Black")
                            )

                            st.plotly_chart(fig, use_container_width=True)
                            
                            if chart_df["Label"].str.contains("N/A").any():
                                st.caption("⚠️ Một số metric đang hiển thị dạng N/A. Bạn có thể cập nhật chúng trong file JSON kết quả ở backend.")
                        else:
                            st.write("No additional metrics available.")

            # --- BIỂU ĐỒ SO SÁNH ĐA ĐỒNG BỘ (HEATMAP MATRIX - PHÓNG TO TỐI ĐA) ---
            # --- BIỂU ĐỒ SO SÁNH ĐA ĐỒNG BỘ (HEATMAP MATRIX - PHÓNG TO CHUẨN) ---
            if len(selected_models) >= 2:
                st.markdown("---")
                st.subheader("Models Comparison Analysis (Heatmap View)")
                
                df_compare = pd.DataFrame(raw_metrics_for_plot)
                
                st.markdown("#### Performance Metrics Heatmap Matrix")
                
                # Sao chép dataframe để xử lý dữ liệu số
                df_heatmap_input = df_compare.copy()
                
                # Lọc động: Chỉ lấy những chỉ số thực sự tồn tại trong DataFrame dựa trên metrics_list gốc
                available_compare_metrics = [m for m in metrics_list if m in df_heatmap_input.columns]
                
                if available_compare_metrics:
                    # Ép kiểu tất cả các cột metric về dạng số (float)
                    for col in available_compare_metrics:
                        df_heatmap_input[col] = pd.to_numeric(df_heatmap_input[col], errors='coerce').fillna(0.0)
                    
                    # Biến đổi dữ liệu: Đặt 'Model' làm Index (Trục Y) và chọn toàn bộ cột chỉ số (Trục X)
                    df_heatmap_matrix = df_heatmap_input.set_index("Model")[available_compare_metrics]
                    
                    # Vẽ biểu đồ ma trận nhiệt Heatmap với tone màu Đơn sắc xanh dương
                    fig_heatmap = px.imshow(
                        df_heatmap_matrix,
                        text_auto=".3f",                 # Tự động in số lên ô, làm tròn 3 chữ số thập phân
                        color_continuous_scale="Blues",     # Tone đơn sắc xanh dương (đậm dần = tốt dần)
                        title="Model Performance Heatmap (Darker Blue is better)",
                        labels=dict(x="Performance Metrics", y="Models", color="Score")
                    )
                    
                    fig_heatmap.update_layout(
                        template="plotly_dark", 
                        height=650,                      # Chiều cao phóng to toàn diện lên 650px
                        margin=dict(t=80, b=80, l=80, r=80), # Nới rộng lề tối đa cho thoáng chữ trục X và Y
                        
                        title=dict(
                            text="Model Performance Heatmap (Darker Blue is better)",
                            font=dict(size=18)           # Phóng to chữ tiêu đề biểu đồ
                        ),
                        
                        # ĐÃ SỬA: Loại bỏ thuộc tính 'font' không hợp lệ, phóng to trực tiếp qua 'tickfont'
                        xaxis=dict(tickfont=dict(size=14)), # Cỡ chữ các nhãn chỉ số ngang
                        yaxis=dict(tickfont=dict(size=14)), # Cỡ chữ các nhãn tên Mô hình dọc
                        
                        # Cấu hình thanh đo màu sắc (Colorbar) tương xứng với biểu đồ lớn
                        coloraxis_colorbar=dict(
                            title="Score",
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            thicknessmode="pixels", thickness=22, # Tăng độ dày thanh màu
                            lenmode="pixels", len=400,            # Kéo dài thanh màu lên 400px cho cân đối dọc
                            yanchor="middle", y=0.5
                        )
                    )
                    
                    # PHÓNG TO CHỮ SỐ BÊN TRONG CÁC Ô HEATMAP
                    fig_heatmap.update_traces(
                        textfont=dict(size=15, weight='bold') # Đẩy cỡ chữ số lên 15 và in đậm cho cực kỳ dễ đọc
                    )
                    
                    # Hiển thị biểu đồ bao phủ toàn bộ chiều ngang vùng chứa (Container)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("⚠️ Không tìm thấy các cột dữ liệu chỉ số (Metrics) tương ứng trong kết quả so sánh.")
else:
    st.info("Please upload a CSV file to start testing and comparison.")