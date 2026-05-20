"""
Page 05: Model Testing and Comparison
Tích hợp Backend xử lý dữ liệu thật, Scale, Dự đoán và So sánh các model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import plotly.express as px
from xgboost import XGBClassifier
from src.model_manager import load_model
import os
import json
import hashlib
import zipfile
from pathlib import Path

from src.sidebar import render_sidebar
from src.config import (
    AVAILABLE_MODELS,
    CHART_SENSOR_COLS,
    DEFAULT_TABLE_ROWS,
    DL_FORECAST_MODEL_PATHS,
    FEATURE_COLS,
    LOCAL_PREDICTION_SCALER_DIR,
    MODEL_THRESHOLDS,
    PREDICTION_CLASSIFIER_EXPORT_DIR,
    PREDICTION_WINDOW_STEPS,
    RF_FORECAST_MODEL_PATHS,
    RF_FORECAST_OPTIONS,
    ROOT_DIR,
    XGBOOST_FORECAST_MODEL_PATHS,
    XGBOOST_FORECAST_OPTIONS,
)
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


SEQUENCE_MODEL_NAMES = {"LSTM", "GRU", "CNN - LSTM", "CNN - GRU"}
LIVE_INFERENCE_BATCH_SIZE = 1024
DEFAULT_SEQUENCE_STRIDE_STEPS = 6


@st.cache_resource(show_spinner=False)
def load_keras_model_for_inference(model_path: str):
    original_path = Path(model_path)
    errors = []

    try:
        return _load_keras_model_path(original_path)
    except Exception as exc:
        errors.append(f"original: {exc}")

    try:
        compat_path = build_keras_compat_archive(original_path)
        return _load_keras_model_path(compat_path)
    except Exception as exc:
        errors.append(f"compat: {exc}")

    raise RuntimeError(" | ".join(errors))


def _load_keras_model_path(model_path: Path):
    try:
        return keras_load_model(str(model_path), compile=False, safe_mode=False)
    except TypeError:
        return keras_load_model(str(model_path), compile=False)


def strip_keras_incompatible_config(value):
    if isinstance(value, dict):
        return {
            key: strip_keras_incompatible_config(item)
            for key, item in value.items()
            if key != "quantization_config"
        }
    if isinstance(value, list):
        return [strip_keras_incompatible_config(item) for item in value]
    return value


def build_keras_compat_archive(model_path: Path) -> Path:
    if not zipfile.is_zipfile(model_path):
        return model_path

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
    compat_dir = ROOT_DIR / ".streamlit" / "keras_compat"
    compat_dir.mkdir(parents=True, exist_ok=True)
    compat_path = compat_dir / f"{model_path.stem}-{digest}.keras"
    if compat_path.exists():
        return compat_path

    with zipfile.ZipFile(model_path, "r") as src:
        config = json.loads(src.read("config.json").decode("utf-8"))
        config = strip_keras_incompatible_config(config)

        with zipfile.ZipFile(compat_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                if info.filename == "config.json":
                    dst.writestr(
                        info,
                        json.dumps(config, ensure_ascii=False, separators=(",", ":")),
                    )
                else:
                    dst.writestr(info, src.read(info.filename))

    return compat_path


@st.cache_resource(show_spinner=False)
def load_scaler_for_inference(scaler_path: str):
    return joblib.load(scaler_path)


@st.cache_data(show_spinner=False)
def load_sequence_metadata_settings():
    metadata_path = PREDICTION_CLASSIFIER_EXPORT_DIR / "metadata.json"
    if not metadata_path.exists():
        return {
            "feature_cols": CHART_SENSOR_COLS,
            "window_steps": PREDICTION_WINDOW_STEPS,
            "stride_steps": DEFAULT_SEQUENCE_STRIDE_STEPS,
            "metadata_path": "",
        }

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        metadata = {}

    return {
        "feature_cols": metadata.get("feature_cols") or CHART_SENSOR_COLS,
        "window_steps": int(metadata.get("window_steps") or PREDICTION_WINDOW_STEPS),
        "stride_steps": int(metadata.get("stride_steps") or DEFAULT_SEQUENCE_STRIDE_STEPS),
        "metadata_path": str(metadata_path),
    }


def get_model_input_shape(model):
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]
    if not isinstance(input_shape, tuple) or len(input_shape) != 3:
        return None, None
    return input_shape[1], input_shape[2]


def get_target_column(df, forecast_horizon: str):
    if forecast_horizon == "Current":
        if "label_in_0h" in df.columns:
            return "label_in_0h"
        if "label" in df.columns:
            return "label"
        return None

    horizon_col = f"label_in_{forecast_horizon}"
    if horizon_col in df.columns:
        return horizon_col
    return "label" if "label" in df.columns else None


def iter_sequence_groups(df):
    group_df = df.copy()
    group_df["asset_id"] = pd.to_numeric(group_df["asset_id"], errors="coerce")
    group_df = group_df.dropna(subset=["asset_id"]).copy()
    group_df["asset_id"] = group_df["asset_id"].astype(int)

    sort_cols = ["asset_id"]
    if "sequence_id" in group_df.columns:
        sort_cols.append("sequence_id")
    if "time_stamp" in group_df.columns:
        group_df["time_stamp"] = pd.to_datetime(group_df["time_stamp"], errors="coerce")
        sort_cols.append("time_stamp")

    group_df = group_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if "sequence_id" in group_df.columns:
        group_cols = ["asset_id", "sequence_id"]
    elif "time_stamp" in group_df.columns:
        group_df["__sequence_segment"] = group_df.groupby("asset_id")["time_stamp"].transform(
            lambda s: s.diff().gt(pd.Timedelta(minutes=20)).fillna(False).cumsum()
        )
        group_cols = ["asset_id", "__sequence_segment"]
    else:
        group_cols = ["asset_id"]

    for _, group in group_df.groupby(group_cols, sort=False):
        yield int(group["asset_id"].iloc[0]), group.reset_index(drop=True)


def safe_auc(metric_fn, y_true, scores):
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(metric_fn(y_true, scores))
    except Exception:
        return None


def compute_live_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred_labels = (scores >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, pred_labels)),
        "precision": float(precision_score(y_true, pred_labels, zero_division=0)),
        "recall": float(recall_score(y_true, pred_labels, zero_division=0)),
        "f1": float(f1_score(y_true, pred_labels, zero_division=0)),
        "pr_auc": safe_auc(average_precision_score, y_true, scores),
        "roc_auc": safe_auc(roc_auc_score, y_true, scores),
        "threshold": float(threshold),
        "evaluated_samples": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "predicted_positive_count": int(pred_labels.sum()),
        "source": "live_inference",
    }


def run_sequence_live_inference(df, model_name: str, forecast_horizon: str):
    horizon_paths = DL_FORECAST_MODEL_PATHS.get(model_name, {})
    model_path = Path(horizon_paths.get(forecast_horizon, ""))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    metadata = load_sequence_metadata_settings()
    feature_cols = list(metadata["feature_cols"])
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(
            "Missing sequence features: "
            + ", ".join(missing_features[:8])
            + ("..." if len(missing_features) > 8 else "")
        )

    target_col = get_target_column(df, forecast_horizon)
    if target_col is None:
        raise ValueError("No label column found for live metric calculation")

    model = load_keras_model_for_inference(str(model_path))
    shape_steps, shape_features = get_model_input_shape(model)
    window_steps = int(shape_steps or metadata["window_steps"])
    if shape_features is not None and int(shape_features) != len(feature_cols):
        raise ValueError(
            f"Model expects {shape_features} features, but upload/metadata provides {len(feature_cols)}"
        )

    stride_steps = max(1, int(metadata["stride_steps"]))
    threshold = float(MODEL_THRESHOLDS.get(model_name, {}).get(forecast_horizon, 0.5))

    all_scores = []
    all_labels = []
    generated_windows = 0
    skipped_assets = []

    for asset_id, group in iter_sequence_groups(df):
        if len(group) < window_steps:
            continue

        scaler_path = LOCAL_PREDICTION_SCALER_DIR / f"asset_{asset_id}.pkl"
        if not scaler_path.exists():
            skipped_assets.append(asset_id)
            continue

        scaler = load_scaler_for_inference(str(scaler_path))
        feature_frame = group[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scaled = scaler.transform(feature_frame.to_numpy(dtype=np.float32))
        target_values = pd.to_numeric(group[target_col], errors="coerce").to_numpy()
        end_indices = np.arange(window_steps - 1, len(group), stride_steps, dtype=int)

        for batch_start in range(0, len(end_indices), LIVE_INFERENCE_BATCH_SIZE):
            batch_end_indices = end_indices[batch_start : batch_start + LIVE_INFERENCE_BATCH_SIZE]
            X = np.stack(
                [scaled[end_idx - window_steps + 1 : end_idx + 1] for end_idx in batch_end_indices],
                axis=0,
            ).astype(np.float32)
            scores = np.asarray(model.predict(X, verbose=0)).reshape(-1)
            labels = target_values[batch_end_indices]
            valid_mask = ~pd.isna(labels)
            if valid_mask.any():
                all_scores.extend(scores[valid_mask].astype(float).tolist())
                all_labels.extend(labels[valid_mask].astype(int).tolist())
            generated_windows += int(len(batch_end_indices))

    if not all_labels:
        raise ValueError("No labeled windows were generated for live inference")

    metrics = compute_live_metrics(all_labels, all_scores, threshold)
    details = {
        "model_path": str(model_path),
        "threshold": threshold,
        "threshold_source": "MODEL_THRESHOLDS",
        "scaler_dir": str(LOCAL_PREDICTION_SCALER_DIR),
        "feature_count": len(feature_cols),
        "window_steps": window_steps,
        "stride_steps": stride_steps,
        "target_col": target_col,
        "generated_windows": generated_windows,
        "skipped_assets": sorted(set(skipped_assets)),
        "metadata_path": metadata.get("metadata_path", ""),
    }
    return metrics, details

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

        with st.spinner("Running live inference and loading saved performance fallback..."):
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
            inference_details = {}

            for model_name in selected_models:
                display_model_name = model_name
                if model_name in ["XGBoost", "Random Forest", "LSTM", "GRU", "CNN - LSTM", "CNN - GRU"]:
                    display_model_name = f"{model_name} ({forecast_horizon})"

                saved_perf_entry = model_perf.get(model_name, {}).get(forecast_horizon, {}) if model_perf else {}
                perf_entry = dict(saved_perf_entry)
                source_label = "saved_json" if saved_perf_entry else "unavailable"

                if model_name in SEQUENCE_MODEL_NAMES:
                    try:
                        live_entry, live_details = run_sequence_live_inference(df, model_name, forecast_horizon)
                        perf_entry = live_entry
                        source_label = "live_inference"
                        inference_details[display_model_name] = {
                            "status": "live_inference",
                            **live_details,
                        }
                    except Exception as exc:
                        inference_details[display_model_name] = {
                            "status": "saved_json_fallback" if saved_perf_entry else "failed",
                            "error": str(exc),
                        }
                        if saved_perf_entry:
                            st.warning(
                                f"Live inference failed for {display_model_name}; "
                                f"showing saved JSON metrics instead. Error: {exc}"
                            )
                        else:
                            st.error(f"Live inference failed for {display_model_name}: {exc}")

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
                    "Threshold": fmt(perf_entry.get("threshold")),
                    "Samples": fmt(perf_entry.get("evaluated_samples")),
                    "Source": source_label,
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

                        inference_detail = inference_details.get(display_model_name)
                        if inference_detail:
                            if inference_detail.get("status") == "live_inference":
                                st.caption(
                                    "Live inference: "
                                    f"{model_row.get('Samples', 'N/A')} windows, "
                                    f"threshold {model_row.get('Threshold', 'N/A')}, "
                                    f"window {inference_detail.get('window_steps')} rows, "
                                    f"stride {inference_detail.get('stride_steps')}, "
                                    f"label `{inference_detail.get('target_col')}`."
                                )
                                st.caption(f"Model path: {inference_detail.get('model_path')}")
                            else:
                                st.caption(
                                    "Saved JSON fallback: "
                                    f"{inference_detail.get('error', 'live inference unavailable')}"
                                )
                        
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
