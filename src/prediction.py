"""Sequence prediction helpers for future-risk monitoring."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_PREDICTION_HORIZON,
    PREDICTION_CLASSIFIER_EXPORT_DIR,
    PREDICTION_HORIZON_RUNS,
    PREDICTION_RESULTS_ROOT,
    PREDICTION_SCALER_DIR,
    PREDICTION_WINDOW_STEPS,
)

from src.model_manager import load_model




MODEL_DISPLAY_NAMES = {
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN - LSTM",
    "cnn_gru": "CNN - GRU",
}
MODEL_INTERNAL_NAMES = {display: internal for internal, display in MODEL_DISPLAY_NAMES.items()}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_keras_input_shape(model_path: Path) -> tuple | None:
    try:
        with zipfile.ZipFile(model_path) as archive:
            config = json.loads(archive.read("config.json"))
        layers = config.get("config", {}).get("layers", [])
        if not layers:
            return None
        return tuple(layers[0].get("config", {}).get("batch_shape") or ())
    except Exception:
        return None


def _feature_count_from_model(model_path: Path) -> int | None:
    input_shape = _read_keras_input_shape(model_path)
    if input_shape and len(input_shape) == 3:
        return int(input_shape[-1])
    return None


def _window_steps_from_model(model_path: Path) -> int | None:
    input_shape = _read_keras_input_shape(model_path)
    if input_shape and len(input_shape) == 3:
        return int(input_shape[-2])
    return None


@st.cache_data(show_spinner=False)
def load_prediction_metadata() -> dict[str, Any]:
    metadata_path =  Path("data/metadata.json")
    if not metadata_path.exists():
        return {
            "available": False,
            "error": f"Prediction metadata not found: {metadata_path}",
            "feature_cols": [],
            "window_steps": PREDICTION_WINDOW_STEPS,
        }

    metadata = _read_json(metadata_path)
    metadata["available"] = True
    metadata.setdefault("window_steps", PREDICTION_WINDOW_STEPS)
    metadata.setdefault("feature_cols", [])
    return metadata


@st.cache_data(show_spinner=False)
def load_prediction_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    metadata = load_prediction_metadata()
    expected_feature_count = len(metadata.get("feature_cols") or [])

    for horizon, run_name in PREDICTION_HORIZON_RUNS.items():
        classifier_dir = PREDICTION_RESULTS_ROOT / run_name / "window_24h" / "classifier"
        comparison_path = classifier_dir / "model_comparison.csv"

        if not comparison_path.exists():
            registry[horizon] = {
                "available": False,
                "horizon": horizon,
                "run_name": run_name,
                "error": f"Prediction comparison not found: {comparison_path}",
            }
            continue

        try:
            comparison = pd.read_csv(comparison_path)
            if comparison.empty or "f1" not in comparison.columns:
                raise ValueError("model_comparison.csv has no usable f1 column")

            best = comparison.sort_values("f1", ascending=False).iloc[0]
            model_name = str(best["model_name"])
            model_dir = classifier_dir / model_name
            model_path = model_dir / "model.keras"
            threshold = float(best["threshold"])
            model_feature_count = _feature_count_from_model(model_path)
            model_window_steps = _window_steps_from_model(model_path)

            registry[horizon] = {
                "available": True,
                "horizon": horizon,
                "run_name": run_name,
                "model_name": model_name,
                "model_display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "model_path": str(model_path),
                "metrics_path": str(model_dir / "metrics.json"),
                "comparison_path": str(comparison_path),
                "threshold": threshold,
                "threshold_source": str(best.get("threshold_source", "")),
                "scaler_dir": str(PREDICTION_SCALER_DIR),
                "artifact_source": "local_training_results",
                "model_feature_count": model_feature_count,
                "feature_contract_count": expected_feature_count,
                "model_window_steps": model_window_steps,
                "f1": float(best.get("f1", np.nan)),
                "precision": float(best.get("precision", np.nan)),
                "recall": float(best.get("recall", np.nan)),
                "accuracy": float(best.get("accuracy", np.nan)),
                "event_f1": float(best.get("event_f1", np.nan)),
            }
        except Exception as exc:
            registry[horizon] = {
                "available": False,
                "horizon": horizon,
                "run_name": run_name,
                "error": f"Cannot read prediction registry for {horizon}: {exc}",
            }

    return registry


def get_prediction_artifact(horizon: str | None = None) -> dict[str, Any]:
    selected_horizon = horizon or DEFAULT_PREDICTION_HORIZON
    registry = load_prediction_registry()
    artifact = registry.get(selected_horizon)
    if artifact:
        return artifact

    # Fallback: if DL artifact not found, check for ML forecast models (XGBoost / Random Forest)
    from src.config import XGBOOST_FORECAST_MODEL_PATHS, RF_FORECAST_MODEL_PATHS, MODEL_THRESHOLDS
    from src.config import FEATURE_COLS as GLOBAL_FEATURE_COLS

    # Prefer XGBoost if available
    xgb_path = Path(XGBOOST_FORECAST_MODEL_PATHS.get(selected_horizon, ""))
    if xgb_path.exists():
        return {
            "available": True,
            "horizon": selected_horizon,
            "model_name": "xgboost",
            "model_display_name": "XGBoost",
            "model_path": str(xgb_path),
            "metrics_path": "",
            "comparison_path": "",
            "threshold": float(MODEL_THRESHOLDS.get("XGBoost", {}).get(selected_horizon, 0.5)),
            "scaler_dir": "",
            "artifact_source": "ml_forecast",
            "model_feature_count": len(GLOBAL_FEATURE_COLS),
            "feature_contract_count": len(GLOBAL_FEATURE_COLS),
            "model_window_steps": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
        }

    rf_path = Path(RF_FORECAST_MODEL_PATHS.get(selected_horizon, ""))
    if rf_path.exists():
        return {
            "available": True,
            "horizon": selected_horizon,
            "model_name": "rf",
            "model_display_name": "Random Forest",
            "model_path": str(rf_path),
            "metrics_path": "",
            "comparison_path": "",
            "threshold": float(MODEL_THRESHOLDS.get("Random Forest", {}).get(selected_horizon, 0.5)),
            "scaler_dir": "",
            "artifact_source": "ml_forecast",
            "model_feature_count": len(GLOBAL_FEATURE_COLS),
            "feature_contract_count": len(GLOBAL_FEATURE_COLS),
            "model_window_steps": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
        }

    return {
        "available": False,
        "horizon": selected_horizon,
        "error": f"Unsupported prediction horizon: {selected_horizon}",
    }


@st.cache_resource(show_spinner=False)
def _load_keras_model(model_path: str):
    errors = []
    
    # 🌟 BƯỚC VÁ TƯƠNG THÍCH: Chuyển chuỗi đường dẫn thành Path object để dọn dẹp config
    native_path = Path(model_path)
    if native_path.suffix.lower() == ".keras":
        compat_path = build_keras_compat_archive(native_path)
        # Ép ngược về dạng chuỗi tuyệt đối chuẩn để Keras load mượt mà không lỗi dấu gạch chéo
        model_path = str(compat_path.resolve())

    # Tiến hành nạp mô hình bản vá
    for import_path in ("keras", "tensorflow.keras"):
        try:
            if import_path == "keras":
                import keras
                return keras.models.load_model(model_path, compile=False), None
            else:
                from tensorflow.keras.models import load_model as keras_load_model
                return keras_load_model(model_path, compile=False), None
                
        except Exception as exc:
            import traceback
            errors.append(f"[{import_path}] {str(exc)}")
            print(f"DEBUG LOAD ERROR ({import_path}): {traceback.format_exc()}")

    return None, " | ".join(errors)


@st.cache_resource(show_spinner=False)
def _load_asset_scaler(scaler_path: str):
    try:
        return joblib.load(scaler_path), None
    except Exception as exc:
        return None, str(exc)


def _empty_result(length: int, horizon: str, status: str) -> dict[str, Any]:
    return {
        "scores": np.full(length, np.nan, dtype=float),
        "labels": np.zeros(length, dtype=int),
        "statuses": [status] * length,
        "horizon": horizon,
        "model_name": "",
        "threshold": np.nan,
    }


def _short_status(message: str, limit: int = 90) -> str:
    clean = " ".join(str(message).split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def _sort_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [col for col in ("sequence_id", "time_stamp", "id") if col in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _candidate_rows_for_current(
    combined: pd.DataFrame,
    current_row: pd.Series,
) -> pd.DataFrame:
    candidates = combined

    if "sequence_id" in candidates.columns and "sequence_id" in current_row.index:
        candidates = candidates[candidates["sequence_id"].astype(str).eq(str(current_row["sequence_id"]))]

    if "id" in candidates.columns and "id" in current_row.index and pd.notna(current_row["id"]):
        row_id = pd.to_numeric(pd.Series([current_row["id"]]), errors="coerce").iloc[0]
        if pd.notna(row_id):
            ids = pd.to_numeric(candidates["id"], errors="coerce")
            return candidates[ids.le(row_id)]

    if "time_stamp" in candidates.columns and "time_stamp" in current_row.index:
        row_time = pd.to_datetime(current_row["time_stamp"], errors="coerce")
        if pd.notna(row_time):
            times = pd.to_datetime(candidates["time_stamp"], errors="coerce")
            return candidates[times.le(row_time)]

    return candidates


def predict_future_risk_batch(
    asset_id: int,
    history: pd.DataFrame,
    batch: pd.DataFrame,
    horizon: str | None = None,
    artifact: dict | None = None,
) -> dict[str, Any]:
    """Predict future risk for each row in a just-streamed turbine batch."""
    selected_horizon = horizon or DEFAULT_PREDICTION_HORIZON
    batch_len = len(batch)

    if batch.empty:
        return _empty_result(0, selected_horizon, "No batch data")

    # 🌟 SỬA LOGIC: Nếu UI truyền artifact qua thì ưu tiên dùng, không dùng mặc định
    if artifact is not None:
        current_artifact = artifact
    else:
        current_artifact = get_prediction_artifact(selected_horizon)

    if not current_artifact.get("available"):
        return _empty_result(batch_len, selected_horizon, current_artifact.get("error", "Prediction unavailable"))

    # Lấy metadata, nếu lỗi hoặc không có thì vẫn tiếp tục chạy luồng Custom UI thay vì return _empty_result ngay
    metadata = load_prediction_metadata()
    
    # 🌟 SỬA ĐOẠN NÀY: Dự phòng danh sách các cột sensor (features) nếu file metadata không có
    feature_cols = list(metadata.get("feature_cols") or [])
    from src.config import FEATURE_COLS as GLOBAL_FEATURE_COLS
    if not feature_cols:
        feature_cols = list(GLOBAL_FEATURE_COLS)
        
    window_steps = int(metadata.get("window_steps") or PREDICTION_WINDOW_STEPS)
    
    # 🌟 SỬA ĐOẠN NÀY: Nới lỏng kiểm tra contract. 
    # Nếu đang dùng luồng Tùy chỉnh từ UI (custom_ui_selection), ta bỏ qua việc so khớp nghiêm ngặt này
    if current_artifact.get("artifact_source") != "custom_ui_selection":
        model_feature_count = current_artifact.get("model_feature_count")
        if model_feature_count and len(feature_cols) != int(model_feature_count):
            return _empty_result(
                batch_len,
                selected_horizon,
                f"Feature contract mismatch: model expects {int(model_feature_count)} features, online stream provides {len(feature_cols)}."
            )

        model_window_steps = current_artifact.get("model_window_steps")
        if model_window_steps and window_steps != int(model_window_steps):
            return _empty_result(
                batch_len,
                selected_horizon,
                f"Window mismatch: model expects {int(model_window_steps)} steps, online stream provides {window_steps}."
            )

    missing_features = [col for col in feature_cols if col not in batch.columns]
    if missing_features:
        msg = "Missing prediction features: " + ", ".join(missing_features[:5])
        if len(missing_features) > 5:
            msg += "..."
        return _empty_result(batch_len, selected_horizon, msg)

    # 🌟 LẤY ĐƯỜNG DẪN MÔ HÌNH CHUẨN ĐÃ ĐƯỢC PHÂN LUỒNG TỪ UI
    model_path = Path("models") / str(current_artifact["model_path"])
    if not model_path.exists():
        # Sửa câu thông báo lỗi chi tiết đường dẫn tuyệt đối giúp bạn dễ debug file nằm đâu
        print(f"DEBUG: Model file for prediction not found at path: {model_path.absolute()}")
        return _empty_result(
            batch_len, 
            selected_horizon, 
            f"Prediction model file NOT FOUND at: {model_path.absolute()}"
        )
        
    # --- PHÂN LUỒNG TẢI MÔ HÌNH ML (.pkl) ---
    if model_path.suffix.lower() == ".pkl":
        try:
            import joblib
        except Exception:
            return _empty_result(batch_len, selected_horizon, "joblib not available to load ML model")

        try:
            model = joblib.load(str(model_path))
        except Exception as exc:
            return _empty_result(batch_len, selected_horizon, _short_status(f"ML model load failed: {exc}"))

        from src.config import ROOT_DIR
        ml_scaler_path = Path(ROOT_DIR) / "models" / "Baseline" / "ML_scaler" / "scada_scaler_full.pkl"
        if not ml_scaler_path.exists():
            return _empty_result(batch_len, selected_horizon, f"ML scaler not found: {ml_scaler_path}")

        scaler, scaler_error = _load_asset_scaler(str(ml_scaler_path.resolve()))
        if scaler is None:
            print(f"DEBUG: {scaler_error} when loading scaler from {ml_scaler_path.resolve()}")
            return _empty_result(batch_len, selected_horizon, _short_status(f"ML scaler load failed: {scaler_error}"))

        try:
            X = batch[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            X_scaled = scaler.transform(X)
        except Exception as exc:
            return _empty_result(batch_len, selected_horizon, _short_status(f"Scaler transform failed: {exc}"))

        try:
            if hasattr(model, "predict_proba"):
                raw_scores = model.predict_proba(X_scaled)[:, 1]
            else:
                raw_scores = model.predict(X_scaled).astype(float)
        except Exception as exc:
            return _empty_result(batch_len, selected_horizon, _short_status(f"ML prediction failed: {exc}"))

        # Sử dụng ngưỡng threshold động lấy từ current_artifact
        threshold = float(current_artifact.get("threshold") or 0.5)
        scores = np.asarray(raw_scores, dtype=float)
        labels = (scores >= threshold).astype(int)
        statuses = ["Ready"] * batch_len

        return {
            "scores": scores,
            "labels": labels,
            "statuses": statuses,
            "horizon": selected_horizon,
            "model_name": current_artifact.get("model_display_name", model_path.stem),
            "threshold": threshold,
        }

    # --- PHÂN LUỒNG TẢI MÔ HÌNH DL (.keras) ---
    scaler_dir = Path("models/DeepLearning/detection_scalers")
    scaler_path = scaler_dir / f"asset_{asset_id}.pkl"
    if not scaler_path.exists():
        return _empty_result(batch_len, selected_horizon, f"Prediction scaler not found: {scaler_path}")

    resolved_path_str = str(model_path.resolve())
    model = load_model(resolved_path_str, use_for_real_time_monitor=True)
    
    if model is None:
        model_error = f"Khong the nap model tu file: {resolved_path_str}. Vui long kiem tra Terminal de xem chi tiet loi can thiep Keras!"
        return _empty_result(
            batch_len, 
            selected_horizon, 
            f"Prediction model load failed: {model_error}"
        )
    scaler, scaler_error = _load_asset_scaler(str(scaler_path))
    if scaler is None:
        return _empty_result(batch_len, selected_horizon, _short_status(f"Prediction scaler load failed: {scaler_error}"))

    history_for_asset = pd.DataFrame()
    if history is not None and not history.empty and "asset_id" in history.columns:
        history_for_asset = history[history["asset_id"] == asset_id].copy()

    combined = pd.concat([history_for_asset, batch.copy()], ignore_index=True)
    combined = _sort_prediction_rows(combined)
    scores = np.full(batch_len, np.nan, dtype=float)
    labels = np.zeros(batch_len, dtype=int)
    statuses = [f"Pending (0/{window_steps})" for _ in range(batch_len)]

    sequences: list[np.ndarray] = []
    row_offsets: list[int] = []
    for offset in range(batch_len):
        current_row = batch.iloc[offset]
        candidates = _sort_prediction_rows(_candidate_rows_for_current(combined, current_row))
        available_steps = len(candidates)
        statuses[offset] = f"Pending ({min(available_steps, window_steps)}/{window_steps})"
        if available_steps < window_steps:
            continue

        window = candidates.tail(window_steps)
        feature_frame = window[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        try:
            scaled = scaler.transform(feature_frame.to_numpy(dtype=np.float32))
        except Exception as exc:
            print(f"DEBUG: {exc} when scaling features for asset {asset_id} at offset {offset}")
            statuses[offset] = _short_status(f"Scaler transform failed: {exc}")
            continue

        sequences.append(np.asarray(scaled, dtype=np.float32))
        row_offsets.append(offset)

    threshold = float(current_artifact["threshold"])
    
    if not sequences:
        return {
            "scores": scores,
            "labels": labels,
            "statuses": statuses,
            "horizon": selected_horizon,
            "model_name": current_artifact.get("model_display_name", ""),
            "threshold": threshold,
        }

    try:
        X = np.stack(sequences, axis=0)
        raw_scores = np.asarray(model.predict(X, verbose=0)).reshape(-1)
    except Exception as exc:
        for offset in row_offsets:
            statuses[offset] = _short_status(f"Prediction failed: {exc}")
        return {
            "scores": scores,
            "labels": labels,
            "statuses": statuses,
            "horizon": selected_horizon,
            "model_name": current_artifact.get("model_display_name", ""),
            "threshold": threshold,
        }

    for offset, score in zip(row_offsets, raw_scores):
        score_value = float(score)
        scores[offset] = score_value
        labels[offset] = int(score_value >= threshold)
        statuses[offset] = "Ready"

    return {
        "scores": scores,
        "labels": labels,
        "statuses": statuses,
        "horizon": selected_horizon,
        "model_name": current_artifact.get("model_display_name", ""),
        "threshold": threshold,
    }
