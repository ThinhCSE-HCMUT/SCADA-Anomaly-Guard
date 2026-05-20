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
    DEFAULT_PREDICTION_MODEL_BY_HORIZON,
    DL_FORECAST_MODEL_PATHS,
    LOCAL_PREDICTION_SCALER_DIR,
    MODEL_THRESHOLDS,
    PREDICTION_CLASSIFIER_EXPORT_DIR,
    PREDICTION_HORIZON_RUNS,
    PREDICTION_RESULTS_ROOT,
    PREDICTION_WINDOW_STEPS,
)


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


def _local_prediction_artifact(horizon: str, reason: str = "") -> dict[str, Any]:
    display_name = DEFAULT_PREDICTION_MODEL_BY_HORIZON.get(horizon, "CNN - LSTM")
    horizon_paths = DL_FORECAST_MODEL_PATHS.get(display_name, {})
    model_path = Path(horizon_paths.get(horizon, ""))
    threshold = MODEL_THRESHOLDS.get(display_name, {}).get(horizon, 0.5)
    model_name = MODEL_INTERNAL_NAMES.get(display_name, display_name.lower().replace(" - ", "_").replace(" ", "_"))

    if not model_path.exists():
        return {
            "available": False,
            "horizon": horizon,
            "model_name": model_name,
            "model_display_name": display_name,
            "error": f"Compatible local prediction model not found: {model_path}",
        }

    return {
        "available": True,
        "horizon": horizon,
        "run_name": "local_21_feature_forecast",
        "model_name": model_name,
        "model_display_name": display_name,
        "model_path": str(model_path),
        "threshold": float(threshold),
        "threshold_source": "local_config",
        "scaler_dir": str(LOCAL_PREDICTION_SCALER_DIR),
        "artifact_source": "local_21_feature_model",
        "fallback_reason": reason,
    }


@st.cache_data(show_spinner=False)
def load_prediction_metadata() -> dict[str, Any]:
    metadata_path = PREDICTION_CLASSIFIER_EXPORT_DIR / "metadata.json"
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
            registry[horizon] = _local_prediction_artifact(
                horizon,
                f"External comparison not found: {comparison_path}",
            )
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

            if (
                expected_feature_count
                and model_feature_count
                and model_feature_count != expected_feature_count
            ):
                registry[horizon] = _local_prediction_artifact(
                    horizon,
                    (
                        f"External {model_name} expects {model_feature_count} features; "
                        f"online stream provides {expected_feature_count}."
                    ),
                )
                continue

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
                "scaler_dir": str(PREDICTION_CLASSIFIER_EXPORT_DIR / "scalers"),
                "artifact_source": "external_results",
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
    return registry.get(
        selected_horizon,
        {
            "available": False,
            "horizon": selected_horizon,
            "error": f"Unsupported prediction horizon: {selected_horizon}",
        },
    )


@st.cache_resource(show_spinner=False)
def _load_keras_model(model_path: str):
    errors = []
    for import_path in ("keras", "tensorflow.keras"):
        try:
            if import_path == "keras":
                import keras

                return keras.models.load_model(model_path, compile=False, safe_mode=False), None

            from tensorflow.keras.models import load_model as keras_load_model

            return keras_load_model(model_path, compile=False, safe_mode=False), None
        except TypeError:
            try:
                if import_path == "keras":
                    import keras

                    return keras.models.load_model(model_path, compile=False), None

                from tensorflow.keras.models import load_model as keras_load_model

                return keras_load_model(model_path, compile=False), None
            except Exception as exc:
                errors.append(f"{import_path}: {exc}")
        except Exception as exc:
            errors.append(f"{import_path}: {exc}")

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
) -> dict[str, Any]:
    """Predict future risk for each row in a just-streamed turbine batch."""
    selected_horizon = horizon or DEFAULT_PREDICTION_HORIZON
    batch_len = len(batch)

    if batch.empty:
        return _empty_result(0, selected_horizon, "No batch data")

    artifact = get_prediction_artifact(selected_horizon)
    if not artifact.get("available"):
        return _empty_result(batch_len, selected_horizon, artifact.get("error", "Prediction unavailable"))

    metadata = load_prediction_metadata()
    if not metadata.get("available"):
        return _empty_result(batch_len, selected_horizon, metadata.get("error", "Metadata unavailable"))

    feature_cols = list(metadata.get("feature_cols") or [])
    window_steps = int(metadata.get("window_steps") or PREDICTION_WINDOW_STEPS)
    missing_features = [col for col in feature_cols if col not in batch.columns]
    if missing_features:
        msg = "Missing prediction features: " + ", ".join(missing_features[:5])
        if len(missing_features) > 5:
            msg += "..."
        return _empty_result(batch_len, selected_horizon, msg)

    model_path = Path(str(artifact["model_path"]))
    if not model_path.exists():
        return _empty_result(batch_len, selected_horizon, f"Prediction model not found: {model_path}")

    scaler_dir = Path(str(artifact.get("scaler_dir") or (PREDICTION_CLASSIFIER_EXPORT_DIR / "scalers")))
    scaler_path = scaler_dir / f"asset_{asset_id}.pkl"
    if not scaler_path.exists():
        return _empty_result(batch_len, selected_horizon, f"Prediction scaler not found: {scaler_path}")

    model, model_error = _load_keras_model(str(model_path))
    if model is None:
        return _empty_result(batch_len, selected_horizon, _short_status(f"Prediction model load failed: {model_error}"))

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
            statuses[offset] = _short_status(f"Scaler transform failed: {exc}")
            continue

        sequences.append(np.asarray(scaled, dtype=np.float32))
        row_offsets.append(offset)

    if not sequences:
        return {
            "scores": scores,
            "labels": labels,
            "statuses": statuses,
            "horizon": selected_horizon,
            "model_name": artifact.get("model_display_name", ""),
            "threshold": float(artifact["threshold"]),
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
            "model_name": artifact.get("model_display_name", ""),
            "threshold": float(artifact["threshold"]),
        }

    threshold = float(artifact["threshold"])
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
        "model_name": artifact.get("model_display_name", ""),
        "threshold": threshold,
    }
