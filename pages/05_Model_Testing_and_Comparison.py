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

from src.i18n import get_language, t
from src.sidebar import render_sidebar
from src.config import AVAILABLE_MODELS, DEFAULT_TABLE_ROWS

get_language()
st.set_page_config(
    page_title=t("testing.title"),  # Tên hiển thị trên tab trình duyệt
    layout="wide",                 # 🔥 CHÌA KHÓA: Ép trang luôn ở chế độ Full Screen
    initial_sidebar_state="expanded" # (Tùy chọn) Ép sidebar luôn mở ra
)

@st.cache_resource
def load_buffer():
    return pd.read_csv("train_buffer.csv")

def load_backend_artifacts():
    try:
        # 1. Load Scaler
        scaler = joblib.load("models/scada_scaler_full.pkl")
        
        # 2. Load XGBoost Model
        xgb_model = XGBClassifier()
        xgb_model.load_model("models/xgb_model.json")
        rf_model = joblib.load("models/rf_model.pkl")
        # LSTM_model = joblib.load("models/lstm.onnx")
        
        # Gói vào dictionary để xài
        loaded_models = {
            "XGBoost": xgb_model,
            "Random Forest": rf_model,
            # "LSTM": LSTM_model
        }
        
        return scaler, loaded_models
    except Exception as e:
        st.error(t("common.model_load_failed", model=t("testing.model_unknown"), error=e))
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

scaler, loaded_models = load_backend_artifacts()
df_buffer = load_buffer()

# ====================== GIAO DIỆN CHÍNH ======================
selected_model_sidebar = render_sidebar()

st.title(t("testing.title"))
st.markdown(f"### {t('testing.description')}")

model_col = t("table.model_column")
processing_time_col = t("table.processing_time")
detected_anomalies_col = t("table.detected_anomalies")
accuracy_col = t("table.accuracy")
f1_col = t("table.f1_score")
precision_col = t("table.precision")
recall_col = t("table.recall")
anomalies_found_col = t("testing.anomalies_found")
status_col = t("common.status")
count_col = t("table.count")
score_col = t("table.score_column")

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        t("testing.upload"),
        type=["csv"],
    )

with col2:
    # Lấy danh sách model từ config, nếu chưa có XGBoost thì tự động thêm vào để test
    model_options = list(AVAILABLE_MODELS.keys())
    if "XGBoost" not in model_options:
        model_options.append("XGBoost")
    if "Random Forest" not in model_options:
        model_options.append("Random Forest")
        
    selected_models = st.multiselect(
        t("testing.select_models"),
        options=model_options,
        default=["XGBoost", "Random Forest"], # Mặc định chọn 2 cái để user thấy tính năng compare
    )

# ====================== PROCESS DATA ======================
if uploaded_file is not None:
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    
    # --- 🌟 CRITICAL FIX: ĐỒNG BỘ DATA VÀ THỜI GIAN TẠI ĐÂY 🌟 ---
    # Ép kiểu datetime để sort không bị sai
    if 'time_stamp' in df.columns:
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        
    # Sắp xếp toàn bộ df ngay từ đầu
    if 'asset_id' in df.columns and 'time_stamp' in df.columns:
        df = df.sort_values(by=['asset_id', 'time_stamp']).reset_index(drop=True)
    # -------------------------------------------------------------
        
    st.success(t("testing.loaded_records", count=len(df)))

    st.subheader(t("testing.data_preview"))
    st.dataframe(df.head(DEFAULT_TABLE_ROWS), use_container_width=True)

    # Nút chạy dự đoán canh giữa
    col_left, col_mid, col_right = st.columns([1, 0.5, 1])
    with col_mid:
        run_button = st.button(t("testing.run_prediction"), type="primary", use_container_width=True)
        
    if run_button:
        if not selected_models:
            st.warning(t("testing.select_one_model"))
            st.stop()

        with st.spinner(t("testing.loading_inference")):
            
            has_label = 'label' in df.columns
            y_true = df['label'] if has_label else None
            
            cols_to_drop = ['time_stamp', 'asset_id', 'label', 'train_test', 'status_type_id', 'sequence_id']
            
            scaler_ml = joblib.load("models/scada_scaler_full.pkl") 

            results = []
            predictions_dict = {}
            raw_metrics_for_plot = [] # Danh sách lưu dữ liệu thô để vẽ biểu đồ so sánh

            for model_name in selected_models:
                start_time = time.time()
                
                if model_name in ["XGBoost", "Random Forest"]:
                    try:
                        if 'asset_id' in df.columns and 'asset_id' not in df_buffer.columns:
                            df_buffer['asset_id'] = df['asset_id'].iloc[0]
                        if 'time_stamp' in df.columns and 'time_stamp' not in df_buffer.columns:
                            start_time_upload = df['time_stamp'].min()
                            df_buffer['time_stamp'] = [start_time_upload - pd.Timedelta(minutes=i) for i in range(6, 0, -1)]

                        common_cols = [c for c in df.columns if c in df_buffer.columns]
                        df_buffer_clean = df_buffer[common_cols].copy()
                        
                        df_buffer_clean['is_buffer'] = True
                        df_upload_temp = df.copy()
                        df_upload_temp['is_buffer'] = False

                        df_combined = pd.concat([df_buffer_clean, df_upload_temp], ignore_index=True)
                        df_combined = df_combined.sort_values(by=['asset_id', 'time_stamp']).reset_index(drop=True)

                        df_rolled_full = apply_rolling_window(df_combined, windows=[3, 6])

                        df_rolled = df_rolled_full[df_rolled_full['is_buffer'] == False].copy()
                        df_rolled = df_rolled.drop(columns=['is_buffer']).reset_index(drop=True)
                        
                    except FileNotFoundError:
                        st.warning(t("testing.no_buffer"))
                        df_rolled = apply_rolling_window(df, windows=[3, 6])
                    
                    has_label = 'label' in df_rolled.columns
                    y_true = df_rolled['label'] if has_label else None
                    
                    X_ml_raw = df_rolled.drop(columns=[c for c in cols_to_drop if c in df_rolled.columns])
                    
                    if hasattr(scaler_ml, 'feature_names_in_'):
                        X_ml_raw = X_ml_raw[scaler_ml.feature_names_in_]
                    
                    # Chuẩn hóa
                    X_scaled_array = scaler_ml.transform(X_ml_raw)
                    X_scaled = pd.DataFrame(X_scaled_array, columns=X_ml_raw.columns)
                    
                    # Dự đoán
                    model = loaded_models.get(model_name)
                    if model is not None:
                        y_pred_class = model.predict(X_scaled)
                    else:
                        y_pred_class = np.zeros(len(df_rolled))
                        
                else:
                    # Logic cho Deep Learning...
                    y_pred_class = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
                
                process_time = (time.time() - start_time) * 1000
                predictions_dict[model_name] = y_pred_class

                anomaly_count = sum(y_pred_class)
                anomaly_rate = anomaly_count / len(y_pred_class)

                # Lưu vào bảng hiển thị
                model_result = {
                    model_col: model_name,
                    processing_time_col: f"{process_time:.1f}",
                    detected_anomalies_col: f"{anomaly_count} ({anomaly_rate*100:.1f}%)"
                }
                
                # Lưu vào list raw để vẽ biểu đồ
                plot_data = {
                    model_col: model_name,
                    processing_time_col: process_time,
                    anomalies_found_col: anomaly_count
                }

                if has_label and y_true is not None:
                    acc = accuracy_score(y_true, y_pred_class)
                    f1 = f1_score(y_true, y_pred_class, zero_division=0)
                    model_result[accuracy_col] = f"{acc:.3f}"
                    model_result[f1_col] = f"{f1:.3f}"
                    model_result[precision_col] = f"{precision_score(y_true, y_pred_class, zero_division=0):.3f}"
                    model_result[recall_col] = f"{recall_score(y_true, y_pred_class, zero_division=0):.3f}"
                    
                    plot_data[accuracy_col] = acc
                    plot_data[f1_col] = f1
                else:
                    model_result[f1_col] = "N/A"

                results.append(model_result)
                raw_metrics_for_plot.append(plot_data)

            st.success(t("testing.prediction_completed"))
            st.toast(t("testing.prediction_toast"), icon="✅")
            
            # --- TỔNG HỢP KẾT QUẢ ---
            st.subheader(t("testing.results_summary"))
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # --- BIỂU ĐỒ CHI TIẾT TỪNG MODEL ---
            st.subheader(t("testing.detailed_dashboard"))
            for model_name in selected_models:
                with st.expander(t("testing.detailed_results", model=model_name), expanded=True): # Đổi thành False cho gọn gàng khi có nhiều model
                    y_pred_m = predictions_dict[model_name]
                    
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.metric(t("testing.total_samples"), len(df))
                        st.metric(t("testing.anomalies_found"), sum(y_pred_m))
                        if has_label:
                            st.metric(t("testing.real_anomalies"), sum(y_true))
                            
                    with c2:
                        anomaly_count = sum(y_pred_m)
                        normal_count = len(y_pred_m) - anomaly_count
                        
                        pie_data = pd.DataFrame({
                            status_col: [t("testing.normal_predicted"), t("testing.anomaly_predicted")],
                            count_col: [normal_count, anomaly_count]
                        })
                        
                        fig = px.pie(
                            pie_data, 
                            names=status_col, 
                            values=count_col, 
                            title=t("testing.prediction_distribution", model=model_name),
                            color=status_col,
                            color_discrete_map={
                                t("testing.normal_predicted"): "#22c55e",  
                                t("testing.anomaly_predicted"): "#ef4444"  
                            },
                            hole=0.45 
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(template="plotly_dark", margin=dict(t=50, b=20, l=20, r=20))
                        st.plotly_chart(fig, use_container_width=True)

            if len(selected_models) >= 2:
                st.markdown("---")
                st.subheader(t("testing.models_comparison"))
                
                df_compare = pd.DataFrame(raw_metrics_for_plot)
                
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    # So sánh số lượng bất thường tìm được
                    fig_anom = px.bar(
                        df_compare, x=model_col, y=anomalies_found_col, color=model_col,
                        title=t("testing.total_anomalies_detected"),
                        text=anomalies_found_col,
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_anom.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_anom, use_container_width=True)
                    
                with col_comp2:
                    # So sánh thời gian xử lý
                    fig_time = px.bar(
                        df_compare, x=model_col, y=processing_time_col, color=model_col,
                        title=t("testing.processing_speed"),
                        text=df_compare[processing_time_col].round(1).astype(str) + " ms",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_time.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_time, use_container_width=True)

                # Nếu có nhãn thực tế, vẽ thêm biểu đồ so sánh Accuracy và F1-Score
                if has_label:
                    st.markdown(f"#### {t('testing.performance_metrics')}")
                    df_metrics_melted = df_compare.melt(
                        id_vars=[model_col], 
                        value_vars=[accuracy_col, f1_col], 
                        var_name=t("common.metric"), 
                        value_name=score_col
                    )
                    
                    fig_metrics = px.bar(
                        df_metrics_melted, x=model_col, y=score_col, color=t("common.metric"), barmode="group",
                        title=t("testing.accuracy_f1"),
                        text=df_metrics_melted[score_col].round(3)
                    )
                    fig_metrics.update_layout(template="plotly_dark", yaxis_range=[0, 1.1])
                    st.plotly_chart(fig_metrics, use_container_width=True)

else:
    st.info(t("testing.upload_prompt"))
