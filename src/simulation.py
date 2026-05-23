import pandas as pd
import streamlit as st

from src.config import (
    DEFAULT_PREDICTION_HORIZON,
    PREDICTION_WINDOW_STEPS,
    SIMULATION_BATCH_SIZE,
    SIMULATION_MAX_HISTORY,
    TARGET_TURBINES,
    TURBINE_LABELS,
)
from src.data_loader import load_current_detection_data, load_online_prediction_data, split_by_turbine
from src.inference import predict_batch
from src.model_manager import load_model
from src.prediction import predict_future_risk_batch


def _trim_history(df: pd.DataFrame, rows_per_asset: int) -> pd.DataFrame:
    if df.empty or "asset_id" not in df.columns:
        return df
    if len(df) <= rows_per_asset * len(TARGET_TURBINES):
        return df

    sort_cols = [col for col in ("asset_id", "sequence_id", "time_stamp", "id") if col in df.columns]
    return (
        df.sort_values(sort_cols, kind="mergesort")
        .groupby("asset_id", group_keys=False)
        .apply(lambda g: g.tail(rows_per_asset))
        .reset_index(drop=True)
    )


def _run_current_detection_step() -> bool:
    """Run the current-detection path on df_simulation.csv and 105 features."""
    current_detection_df = load_current_detection_data()
    if current_detection_df.empty:
        return False

    current_detection_dfs = split_by_turbine(current_detection_df)
    running_model = load_model(st.session_state.selected_model)
    st.session_state.setdefault("current_detection_turbine_steps", {tid: 0 for tid in TARGET_TURBINES})
    st.session_state.setdefault("current_detection_stream_done", {tid: False for tid in TARGET_TURBINES})
    st.session_state.setdefault("current_detection_history_data", pd.DataFrame())
    st.session_state.setdefault("anomaly_records", [])

    new_rows = []
    any_active = False

    for tid in TARGET_TURBINES:
        if st.session_state.current_detection_stream_done.get(tid, False):
            continue

        step = st.session_state.current_detection_turbine_steps.get(tid, 0)
        batch = current_detection_dfs[tid].iloc[
            step * SIMULATION_BATCH_SIZE : (step + 1) * SIMULATION_BATCH_SIZE
        ].copy()

        if batch.empty:
            st.session_state.current_detection_stream_done[tid] = True
            continue

        pred_labels, pred_probas = predict_batch(running_model, batch)
        batch["pred_label"] = pred_labels
        batch["anomaly_score"] = pred_probas
        batch["model_used"] = st.session_state.selected_model

        new_rows.append(batch)
        any_active = True
        st.session_state.current_detection_turbine_steps[tid] = step + 1

        for _, row in batch[batch["pred_label"] == 1].iterrows():
            st.session_state.anomaly_records.append(
                {
                    "time": row["time_stamp"],
                    "turbine": TURBINE_LABELS[tid],
                    "score": round(float(row["anomaly_score"]), 3),
                }
            )

    if new_rows:
        st.session_state.current_detection_history_data = pd.concat(
            [st.session_state.current_detection_history_data, *new_rows],
            ignore_index=True,
        )
        st.session_state.current_detection_history_data = _trim_history(
            st.session_state.current_detection_history_data,
            SIMULATION_MAX_HISTORY,
        )

    return any_active


def run_simulation_step():
    """Run one dashboard tick for all-turbine future prediction plus current detection."""
    prediction_df = load_online_prediction_data()
    if prediction_df.empty:
        st.session_state.is_monitoring = False
        return "NO_DATA"

    prediction_dfs = split_by_turbine(prediction_df)
    prediction_horizon = st.session_state.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON)
    st.session_state.setdefault("turbine_steps", {tid: 0 for tid in TARGET_TURBINES})
    st.session_state.setdefault("stream_done", {tid: False for tid in TARGET_TURBINES})
    st.session_state.setdefault("history_data", pd.DataFrame())
    st.session_state.setdefault("future_risk_records", [])

    new_rows = []
    prediction_active = False

    for tid in TARGET_TURBINES:
        if st.session_state.stream_done.get(tid, False):
            continue

        step = st.session_state.turbine_steps.get(tid, 0)
        batch = prediction_dfs[tid].iloc[
            step * SIMULATION_BATCH_SIZE : (step + 1) * SIMULATION_BATCH_SIZE
        ].copy()

        if batch.empty:
            st.session_state.stream_done[tid] = True
            continue

        future_risk = predict_future_risk_batch(
            tid,
            st.session_state.history_data,
            batch,
            prediction_horizon,
        )
        batch["future_risk_score"] = future_risk["scores"]
        batch["future_risk_label"] = future_risk["labels"]
        batch["future_prediction_status"] = future_risk["statuses"]
        batch["future_horizon"] = future_risk["horizon"]
        batch["future_model"] = future_risk["model_name"]
        batch["future_threshold"] = future_risk["threshold"]

        new_rows.append(batch)
        prediction_active = True
        st.session_state.turbine_steps[tid] = step + 1

        for _, row in batch[batch["future_risk_label"] == 1].iterrows():
            if pd.notna(row.get("future_risk_score")):
                st.session_state.future_risk_records.append(
                    {
                        "time": row["time_stamp"],
                        "turbine": TURBINE_LABELS[tid],
                        "event_id": int(row.get("sequence_id", -1)),
                        "event_label": row.get("event_label", ""),
                        "event_description": row.get("event_description", ""),
                        "score": round(float(row["future_risk_score"]), 3),
                        "horizon": row.get("future_horizon", prediction_horizon),
                        "model": row.get("future_model", ""),
                        "threshold": round(float(row.get("future_threshold", 0.0)), 3),
                    }
                )

    if new_rows:
        st.session_state.history_data = pd.concat(
            [st.session_state.history_data, *new_rows],
            ignore_index=True,
        )
        st.session_state.history_data = _trim_history(
            st.session_state.history_data,
            max(SIMULATION_MAX_HISTORY, PREDICTION_WINDOW_STEPS + SIMULATION_BATCH_SIZE),
        )

    _run_current_detection_step()

    if not prediction_active and all(st.session_state.stream_done.values()):
        st.session_state.is_monitoring = False
        return "DONE"

    return "RUNNING"
