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

os.environ["TF_USE_LEGACY_KERAS"] = "1"

from src.sidebar import render_sidebar
from src.config import AVAILABLE_MODELS, DEFAULT_TABLE_ROWS, MODEL_THRESHOLDS

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
        # 1. Load Scaler
        scaler = joblib.load("models/scada_scaler_full.pkl")
        
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
                
        # (Riêng XGBoost nếu bạn bắt buộc dùng file .json thay vì .pkl thì nạp thủ công ở đây)
        if "XGBoost" not in loaded_models and os.path.exists("models/xgb_model.json"):
            xgb_model = XGBClassifier()
            xgb_model.load_model("models/xgb_model.json")
            loaded_models["XGBoost"] = xgb_model
        
        return scaler, loaded_models
        
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

scaler, loaded_models = load_backend_artifacts()
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

with col2:
    # Lấy danh sách model từ config, nếu chưa có XGBoost thì tự động thêm vào để test
    model_options = list(AVAILABLE_MODELS.keys())
        
    selected_models = st.multiselect(
        "Select Models to Test",
        options=model_options,
    )

# ====================== PROCESS DATA ======================
if uploaded_file is not None:
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    
    # Ép kiểu datetime để sort không bị sai
    if 'time_stamp' in df.columns:
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        
    # Sắp xếp toàn bộ df ngay từ đầu
    if 'asset_id' in df.columns and 'time_stamp' in df.columns:
        df = df.sort_values(by=['asset_id', 'time_stamp']).reset_index(drop=True)
    # -------------------------------------------------------------
        
    st.success(f"Loaded {len(df)} records from uploaded file")

    st.subheader("Data Preview")
    st.dataframe(df.head(DEFAULT_TABLE_ROWS), use_container_width=True)

    # Nút chạy dự đoán canh giữa
    col_left, col_mid, col_right = st.columns([1, 0.5, 1])
    with col_mid:
        run_button = st.button("Run Prediction", type="primary", use_container_width=True)
        
    if run_button:
        if not selected_models:
            st.warning("Please select at least one model!")
            st.stop()

        with st.spinner("Running inference on selected models..."):
            
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
                        
                    has_label = 'label' in df_rolled.columns
                    y_true = df_rolled['label'] if has_label else None
                    
                    X_ml_raw = df_rolled.drop(columns=[c for c in cols_to_drop if c in df_rolled.columns])
                    
                    if hasattr(scaler_ml, 'feature_names_in_'):
                        X_ml_raw = X_ml_raw[scaler_ml.feature_names_in_]
                    
                    # Scale
                    X_scaled_array = scaler_ml.transform(X_ml_raw)
                    X_scaled = pd.DataFrame(X_scaled_array, columns=X_ml_raw.columns)
                    
                    # Predict
                    model = loaded_models.get(model_name)
                    if model is not None:
                        y_pred_class = model.predict(X_scaled)
                    else:
                        y_pred_class = np.zeros(len(df_rolled))
                        
                else:
                    # DL Logic
                    SEQ_LENGTH = 144
                    
                    cols_to_exclude = ['asset_id', 'time_stamp', 'label']
                    base_features = [c for c in df.columns if c not in cols_to_exclude]
                    
                    df_dl_raw = df.copy()
                    
                    X_dl_raw = df_dl_raw[base_features]

                    current_asset = df['asset_id'].iloc[0] if 'asset_id' in df.columns else None
                    
                    if current_asset is not None:
                        if isinstance(current_asset, str) and 'asset_' in current_asset:
                            asset_num = current_asset.replace('asset_', '')
                        else:
                            asset_num = int(current_asset)
                            
                        scaler_path = f"models/asset_{asset_num}.pkl"
                        
                        try:
                            scaler_dl = joblib.load(scaler_path)
                            if hasattr(scaler_dl, 'feature_names_in_'):
                                X_dl_raw = X_dl_raw[scaler_dl.feature_names_in_]
                                
                            X_scaled_array = scaler_dl.transform(X_dl_raw)
                        except FileNotFoundError:
                            st.error(f"Không tìm thấy Scaler cho turbine {asset_num} tại đường dẫn: '{scaler_path}'. Vui lòng kiểm tra lại!")
                            st.stop()
                    else:
                        st.error("Dữ liệu không có cột 'asset_id' để xác định turbine. Vui lòng kiểm tra lại CSV.")
                        st.stop()
                    
                    
                    if len(X_scaled_array) < SEQ_LENGTH:
                        st.warning(f"Dữ liệu tải lên quá ngắn. Cần ít nhất {SEQ_LENGTH} dòng để chạy Deep Learning.")
                        y_pred_class = np.zeros(len(df_dl_raw))
                    else:
                        # (samples, time_steps, features)
                        X_seq = []
                        for i in range(len(X_scaled_array) - SEQ_LENGTH + 1):
                            X_seq.append(X_scaled_array[i : i + SEQ_LENGTH])
                        X_seq = np.array(X_seq)
                        
                        # Predict
                        model = loaded_models.get(model_name)

                        if model is not None:
                            y_pred_prob = model.predict(X_seq, verbose=0)
                            current_threshold = MODEL_THRESHOLDS.get(model_name, 0.5)
                            y_pred_class_seq = (y_pred_prob >= current_threshold).astype(int).flatten()

                            pad_length = SEQ_LENGTH - 1
                            y_pred_class = np.pad(y_pred_class_seq, (pad_length, 0), 'constant', constant_values=0)

                        else:
                            print(f"Model {model_name} not found in loaded_models. Skipping prediction.")
                            y_pred_class = np.zeros(len(df_dl_raw))
                            
                    # Update y_true
                    has_label = 'label' in df_dl_raw.columns
                    y_true = df_dl_raw['label'] if has_label else None
                
                process_time = (time.time() - start_time)
                predictions_dict[model_name] = y_pred_class

                anomaly_count = sum(y_pred_class)
                anomaly_rate = anomaly_count / len(y_pred_class)

                # Lưu vào bảng hiển thị
                model_result = {
                    "Model": model_name,
                    "Processing Time (s)": f"{process_time:.1f}",
                    "Detected Anomalies": f"{anomaly_count} ({anomaly_rate*100:.1f}%)"
                }
                
                # Lưu vào list raw để vẽ biểu đồ
                plot_data = {
                    "Model": model_name,
                    "Processing Time (s)": process_time,
                    "Anomalies Found": anomaly_count
                }

                if has_label and y_true is not None:
                    acc = accuracy_score(y_true, y_pred_class)
                    f1 = f1_score(y_true, y_pred_class, zero_division=0)
                    model_result["Accuracy"] = f"{acc:.3f}"
                    model_result["F1-Score"] = f"{f1:.3f}"
                    model_result["Precision"] = f"{precision_score(y_true, y_pred_class, zero_division=0):.3f}"
                    model_result["Recall"] = f"{recall_score(y_true, y_pred_class, zero_division=0):.3f}"
                    
                    plot_data["Accuracy"] = acc
                    plot_data["F1-Score"] = f1
                else:
                    model_result["F1-Score"] = "N/A"

                results.append(model_result)
                raw_metrics_for_plot.append(plot_data)

            st.success("Prediction completed!")
            st.toast("Model prediction completed successfully!", icon="✅")
            
            # --- TỔNG HỢP KẾT QUẢ ---
            st.subheader("Results Summary")
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            # --- BIỂU ĐỒ CHI TIẾT TỪNG MODEL ---
            st.subheader("Detailed Prediction Dashboard")
            for model_name in selected_models:
                with st.expander(f"Detailed Results: {model_name}", expanded=True): # Đổi thành False cho gọn gàng khi có nhiều model
                    y_pred_m = predictions_dict[model_name]
                    
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.metric("Total Samples", len(df))
                        st.metric("Anomalies Found", sum(y_pred_m))
                        if has_label:
                            st.metric("Real Anomalies", sum(y_true))
                            
                    with c2:
                        anomaly_count = sum(y_pred_m)
                        normal_count = len(y_pred_m) - anomaly_count
                        
                        pie_data = pd.DataFrame({
                            "Status": ["Normal Predicted (0)", "Anomaly Predicted (1)"],
                            "Count": [normal_count, anomaly_count]
                        })
                        
                        fig = px.pie(
                            pie_data, 
                            names="Status", 
                            values="Count", 
                            title=f"Prediction Distribution - {model_name}",
                            color="Status",
                            color_discrete_map={
                                "Normal Predicted (0)": "#22c55e",  
                                "Anomaly Predicted (1)": "#ef4444"  
                            },
                            hole=0.45 
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(template="plotly_dark", margin=dict(t=50, b=20, l=20, r=20))
                        st.plotly_chart(fig, use_container_width=True)

            if len(selected_models) >= 2:
                st.markdown("---")
                st.subheader("Models Comparison Analysis")
                
                df_compare = pd.DataFrame(raw_metrics_for_plot)
                
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    # So sánh số lượng bất thường tìm được
                    fig_anom = px.bar(
                        df_compare, x="Model", y="Anomalies Found", color="Model",
                        title="Total Anomalies Detected by Model",
                        text="Anomalies Found",
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_anom.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_anom, use_container_width=True)
                    
                with col_comp2:
                    # So sánh thời gian xử lý
                    fig_time = px.bar(
                        df_compare, x="Model", y="Processing Time (s)", color="Model",
                        title="Processing Speed Comparison (Lower is better)",
                        text=df_compare["Processing Time (s)"].round(1).astype(str) + " s",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_time.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_time, use_container_width=True)

                # Nếu có nhãn thực tế, vẽ thêm biểu đồ so sánh Accuracy và F1-Score
                if has_label:
                    st.markdown("#### Performance Metrics")
                    df_metrics_melted = df_compare.melt(
                        id_vars=["Model"], 
                        value_vars=["Accuracy", "F1-Score"], 
                        var_name="Metric", 
                        value_name="Score"
                    )
                    
                    fig_metrics = px.bar(
                        df_metrics_melted, x="Model", y="Score", color="Metric", barmode="group",
                        title="Accuracy & F1-Score Comparison",
                        text=df_metrics_melted["Score"].round(3)
                    )
                    fig_metrics.update_layout(template="plotly_dark", yaxis_range=[0, 1.1])
                    st.plotly_chart(fig_metrics, use_container_width=True)

else:
    st.info("Please upload a CSV file to start testing and comparison.")