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

    # FIX: Chạy current detection TRƯỚC để current_detection_history_data đã có kết quả
    # của tick hiện tại khi vòng lặp prediction bên dưới cố merge pred_label vào batch.
    # Trước đây _run_current_detection_step() được gọi SAU vòng lặp → pred_label luôn
    # lệch 1 tick (merge chỉ thấy kết quả batch N-1 chứ không phải batch N).
    _run_current_detection_step()

    # 🌟 BƯỚC 1: Lấy artifact cấu hình chuẩn (đường dẫn chuẩn + threshold động) từ UI lưu trong Session State
    active_artifact = st.session_state.get("active_prediction_artifact", None)

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

        # 🌟 BƯỚC 2: Truyền thêm tham số 'artifact=active_artifact' vào hàm dưới đây
        future_risk = predict_future_risk_batch(
            asset_id=tid,
            history=st.session_state.history_data,
            batch=batch,
            horizon=prediction_horizon,
            artifact=active_artifact,  # <-- Thêm dòng này để ép backend dùng cấu hình từ UI
        )
        
        batch["future_risk_score"] = future_risk["scores"]
        batch["future_risk_label"] = future_risk["labels"]
        batch["future_prediction_status"] = future_risk["statuses"]
        batch["future_horizon"] = future_risk["horizon"]
        batch["future_model"] = future_risk["model_name"]
        batch["future_threshold"] = future_risk["threshold"]
        
        current_det = st.session_state.get("current_detection_history_data", pd.DataFrame())
        if not current_det.empty:
            tid_current = current_det[
                pd.to_numeric(current_det["asset_id"], errors="coerce").fillna(-1).astype(int) == tid
            ].copy()
            
            if not tid_current.empty:
                # First try to map by `sequence_id` (stable across datasets)
                if "sequence_id" in batch.columns and "sequence_id" in tid_current.columns:
                    merge_cols = [c for c in ("sequence_id", "pred_label", "anomaly_score") if c in tid_current.columns]
                    if "pred_label" in tid_current.columns:
                        left = batch
                        right = tid_current[[c for c in ("sequence_id", "pred_label", "anomaly_score") if c in tid_current.columns]]
                        # perform left merge on sequence_id
                        batch = pd.merge(left, right, on="sequence_id", how="left", suffixes=(None, "_det"))
                        # normalize columns: prefer detection values from `tid_current`
                        # If merge produced *_det columns, use them; otherwise if `pred_label`/`anomaly_score`
                        # exist (coming from right-side selection), keep them. Only default when missing.
                        if "pred_label_det" in batch.columns:
                            batch["pred_label"] = batch["pred_label_det"].fillna(0).astype(int)
                            batch.drop(columns=["pred_label_det"], inplace=True)
                        elif "pred_label" in batch.columns:
                            # keep existing pred_label (likely from right-side join)
                            batch["pred_label"] = pd.to_numeric(batch["pred_label"], errors="coerce").fillna(0).astype(int)
                        else:
                            batch["pred_label"] = 0

                        if "anomaly_score_det" in batch.columns:
                            batch["anomaly_score"] = batch["anomaly_score_det"].fillna(0.0).astype(float)
                            batch.drop(columns=["anomaly_score_det"], inplace=True)
                        elif "anomaly_score" in batch.columns:
                            batch["anomaly_score"] = pd.to_numeric(batch["anomaly_score"], errors="coerce").fillna(0.0).astype(float)
                        else:
                            batch["anomaly_score"] = 0.0

                        # If mapping by sequence_id produced no anomalies, fall back to timestamp nearest join for leftovers
                        unmapped_mask = (~batch["pred_label"].astype(bool))
                    else:
                        # No pred_label in detection data; initialize defaults
                        batch["pred_label"] = 0
                        batch["anomaly_score"] = 0.0
                        unmapped_mask = pd.Series([True] * len(batch), index=batch.index)
                else:
                    # No sequence_id available — mark all as unmapped and fallback to time mapping
                    batch["pred_label"] = 0
                    batch["anomaly_score"] = 0.0
                    unmapped_mask = pd.Series([True] * len(batch), index=batch.index)

                # For any rows still unmapped, try nearest-time matching within 60s
                try:
                    batch_ts = pd.to_datetime(batch.loc[unmapped_mask, "time_stamp"]).reset_index()
                    curr_ts = pd.to_datetime(tid_current["time_stamp"])
                    # build time-indexed mapping
                    pred_map = {pd.to_datetime(r["time_stamp"]): (int(r.get("pred_label", 0)), float(r.get("anomaly_score", 0.0))) for _, r in tid_current.iterrows()}
                    for idx_row in batch_ts.itertuples(index=False):
                        i = idx_row["index"]
                        ts = pd.to_datetime(idx_row["time_stamp"]) if "time_stamp" in idx_row._fields else pd.to_datetime(idx_row[1])
                        if ts in pred_map:
                            batch.loc[i, "pred_label"] = pred_map[ts][0]
                            batch.loc[i, "anomaly_score"] = pred_map[ts][1]
                        else:
                            # find nearest timestamp within tolerance
                            nearest_ts = min(pred_map.keys(), key=lambda x: abs((x - ts).total_seconds()))
                            if abs((nearest_ts - ts).total_seconds()) <= 60:
                                batch.loc[i, "pred_label"] = pred_map[nearest_ts][0]
                                batch.loc[i, "anomaly_score"] = pred_map[nearest_ts][1]
                except Exception:
                    # if anything fails, keep defaults
                    pass
        
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

    if not prediction_active and all(st.session_state.stream_done.values()):
        st.session_state.is_monitoring = False
        return "DONE"

    return "RUNNING"