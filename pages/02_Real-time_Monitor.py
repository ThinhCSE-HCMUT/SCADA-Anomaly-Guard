"""Page 02: all-turbine real-time monitor."""

from datetime import datetime
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
import zipfile
import json
import hashlib
from src.config import ROOT_DIR

st.set_page_config(
    page_title="Real-time Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import (
    AVAILABLE_MODELS,
    CHART_SENSOR_COLS,
    DEFAULT_PREDICTION_HORIZON,
    PREDICTION_HORIZONS,
    PREDICTION_WINDOW_STEPS,
    SIMULATION_DELAY,
    STATUS_COLORS,
    TARGET_TURBINES,
    TRACE_COLORS,
    TURBINE_LABELS,
    get_sensor_label,
    get_sensor_unit,
    MODEL_THRESHOLDS,
    DL_FORECAST_OPTIONS,
)
from src.data_loader import load_online_prediction_data
from src.model_manager import load_model
from src.prediction import get_prediction_artifact
from src.sidebar import render_sidebar
from src.simulation import run_simulation_step


DEFAULT_SELECTED_SENSORS = [
    "sensor_0_avg",
    "sensor_10_avg",
    "wind_speed_3_min",
    "sensor_41_avg",
    "sensor_18_avg",
]


def add_system_log(event_message: str) -> None:
    if "system_logs" not in st.session_state:
        st.session_state.system_logs = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.system_logs.insert(0, {"Timestamp": timestamp, "Event": event_message})


def _init_state() -> None:
    defaults = {
        "is_monitoring": False,
        "turbine_steps": {tid: 0 for tid in TARGET_TURBINES},
        "stream_done": {tid: False for tid in TARGET_TURBINES},
        "history_data": pd.DataFrame(),
        "current_detection_history_data": pd.DataFrame(),
        "anomaly_records": [],
        "future_risk_records": [],
        "selected_model": next(iter(AVAILABLE_MODELS)),
        "prediction_horizon": DEFAULT_PREDICTION_HORIZON,
        "current_detection_turbine_steps": {tid: 0 for tid in TARGET_TURBINES},
        "current_detection_stream_done": {tid: False for tid in TARGET_TURBINES},
        # Fix lỗi 1: Khởi tạo giá trị lựa chọn tuabin mặc định trong session_state
        "selected_detail_turbine_label": TURBINE_LABELS[TARGET_TURBINES[0]],
        "system_logs": [
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Event": "Real-time Monitor Page Loaded",
            }
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _fault_ranges(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if "status_type_id" not in df.columns:
        return []

    ranges = []
    in_fault = False
    start_time = None

    for idx, row in df.iterrows():
        is_fault = int(row.get("status_type_id", 0)) == 1
        if is_fault and not in_fault:
            in_fault = True
            start_time = row["time_stamp"]
        elif not is_fault and in_fault:
            in_fault = False
            ranges.append((start_time, df.iloc[idx - 1]["time_stamp"]))

    if in_fault:
        ranges.append((start_time, df.iloc[-1]["time_stamp"]))
    return ranges


def _format_percent_score(score: float) -> str:
    return f"{float(score) * 100:.1f}%"


def _format_threshold_delta(threshold: float) -> str:
    if pd.notna(threshold):
        return f"Threshold {_format_percent_score(threshold)}"
    return "Ready"


def _chart_revision(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    latest_time = df["time_stamp"].iloc[-1] if "time_stamp" in df.columns else ""
    return f"{len(df)}-{latest_time}"


def _render_live_chart(fig: go.Figure, chart_key: str) -> None:
    chart_config = {
        "displayModeBar": False,
        "responsive": True,
        "scrollZoom": False,
    }
    try:
        st.plotly_chart(fig, use_container_width=True, key=chart_key, config=chart_config)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=chart_config)

_init_state()
raw_df = load_online_prediction_data()

render_sidebar()

st.title("Real-time Monitor")
st.markdown("All-turbine monitoring with current fault detection and DL future-risk prediction.")
st.divider()

col2, col3, col4 = st.columns([2.5, 1.2, 1.1])

with col2:
    default_sensors = [col for col in DEFAULT_SELECTED_SENSORS if col in CHART_SENSOR_COLS]
    chosen_sensors = st.multiselect(
        "Choose Sensors to display",
        options=CHART_SENSOR_COLS,
        default=default_sensors,
        format_func=get_sensor_label,
        max_selections=10,
    )
    if not chosen_sensors:
        chosen_sensors = default_sensors

with col3:
    # 1. Lấy danh sách các model khả dụng
    model_options = list(AVAILABLE_MODELS.keys())
    
    # 2. Tìm xem model đã lưu trong session_state đang nằm ở vị trí (index) thứ mấy
    saved_model = st.session_state.get("selected_model", model_options[0])
    default_model_index = model_options.index(saved_model) if saved_model in model_options else 0

    # 3. Truyền index này vào selectbox để giữ vững trạng thái khi chuyển trang
    model_choice = st.selectbox(
        "Current Detection Model (ML/DL)",
        options=model_options,
        index=default_model_index,  # <-- QUAN TRỌNG: Ép selectbox hiển thị đúng model đã chọn trước đó
        disabled=st.session_state.is_monitoring,
    )
    st.session_state["selected_model"] = model_choice
    
    active_model = load_model(model_choice)
    if active_model is not None:
        st.caption(f"Real-time Model loaded: **{model_choice}**")
    else:
        st.caption("Model unavailable; current detection uses fallback labels.")

with col4:
    current_horizon = st.session_state.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON)
    horizon_index = PREDICTION_HORIZONS.index(current_horizon) if current_horizon in PREDICTION_HORIZONS else 1
    
    prediction_horizon = st.selectbox(
        "Future Risk Horizon",
        options=PREDICTION_HORIZONS,
        index=horizon_index,
        disabled=st.session_state.is_monitoring,
    )
    st.session_state["prediction_horizon"] = prediction_horizon
    
    chosen_threshold = 0.5
    if model_choice in MODEL_THRESHOLDS:
        chosen_threshold = MODEL_THRESHOLDS[model_choice].get(prediction_horizon, 0.5)
    elif "XGB" in model_choice.upper():
        chosen_threshold = MODEL_THRESHOLDS["XGBoost"].get(prediction_horizon, 0.5)
    elif "FOREST" in model_choice.upper() or "RF" in model_choice.upper():
        chosen_threshold = MODEL_THRESHOLDS["Random Forest"].get(prediction_horizon, 0.5)

    prediction_artifact = {
        "available": True,
        "model_display_name": f"{model_choice} ({prediction_horizon})",
        "threshold": chosen_threshold,
        "artifact_source": "custom_ui_selection"
    }

    if model_choice in DL_FORECAST_OPTIONS:
        horizon_mapping = DL_FORECAST_OPTIONS[model_choice]
        if prediction_horizon in horizon_mapping:
            target_keras_path = horizon_mapping[prediction_horizon]
            prediction_artifact["model_path"] = target_keras_path
            prediction_artifact["artifact_source"] = "local_21_feature_model"
            # st.caption(f"Future Model Path: `{target_keras_path}`")
        else:
            target_keras_path = f"models/DeepLearning/{model_choice}_{prediction_horizon}.keras"
            prediction_artifact["model_path"] = target_keras_path
            prediction_artifact["artifact_source"] = "local_21_feature_model"
            # st.caption(f"Future Model Path (Fallback): `{target_keras_path}`")
    else:
        from pathlib import Path
        ml_folder_name = "XGBoost" if "XGB" in model_choice.upper() else "RandomForest"
        file_prefix = "xgb_model" if ml_folder_name == "XGBoost" else "rf_model"
        target_pkl_path = Path(ROOT_DIR) / "models" / "Baseline" / "ML_scaler" / "scada_scaler_full.pkl"
        
        prediction_artifact["model_path"] = str(target_pkl_path)
        prediction_artifact["artifact_source"] = "ml_forecast"
        # st.caption(f"Future Model Path: `{target_pkl_path}`")

    # st.session_state["active_prediction_artifact"] = prediction_artifact

    if prediction_artifact.get("available"):
        artifact_source = prediction_artifact.get("artifact_source", "")
        source_labels = {
            "local_training_results": "local training result",
            "external_results": "external training result",
            "local_21_feature_model": "local 21-feature DL",
            "ml_forecast": "ML forecast",
            "custom_ui_selection": "UI custom path"
        }
        source_label = source_labels.get(artifact_source, artifact_source or "unknown")
        # st.caption(
        #     f"Active Forecast Target: **{prediction_artifact.get('model_display_name')}** "
        #     f"@ {prediction_artifact.get('threshold'):.3f} ({source_label})"
        # )
    else:
        st.caption("Prediction model unavailable")

_, col_start, col_stop, _ = st.columns([2, 1, 1, 2])

with col_start:
    start_clicked = st.button(
        "Start",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_monitoring or raw_df.empty,
    )

with col_stop:
    stop_clicked = st.button(
        "Stop",
        use_container_width=True,
        disabled=not st.session_state.is_monitoring,
    )

if start_clicked:
    add_system_log(
        f"Simulation STARTED for all target turbines; "
        f"current detection model: {model_choice}; prediction horizon: {prediction_horizon}"
    )
    st.session_state.update(
        {
            "is_monitoring": True,
            "turbine_steps": {tid: 0 for tid in TARGET_TURBINES},
            "stream_done": {tid: False for tid in TARGET_TURBINES},
            "history_data": pd.DataFrame(),
            "current_detection_history_data": pd.DataFrame(),
            "anomaly_records": [],
            "future_risk_records": [],
            "selected_model": model_choice,
            "prediction_horizon": prediction_horizon,
            "current_detection_turbine_steps": {tid: 0 for tid in TARGET_TURBINES},
            "current_detection_stream_done": {tid: False for tid in TARGET_TURBINES},
        }
    )
    st.rerun()

if stop_clicked:
    add_system_log("Simulation STOPPED by user")
    st.session_state.is_monitoring = False
    st.rerun()

st.divider()

if st.session_state.is_monitoring:
    prev_anomaly_count = len(st.session_state.anomaly_records)
    prev_future_risk_count = len(st.session_state.get("future_risk_records", []))

    status = run_simulation_step()

    current_anomaly_count = len(st.session_state.anomaly_records)
    if current_anomaly_count > prev_anomaly_count:
        new_anoms = st.session_state.anomaly_records[prev_anomaly_count:]
        for anom in new_anoms:
            add_system_log(f"NEW ANOMALY detected: {anom['turbine']} (Score: {anom['score']:.2f})")

    current_future_risk_count = len(st.session_state.get("future_risk_records", []))
    if current_future_risk_count > prev_future_risk_count:
        new_risks = st.session_state.future_risk_records[prev_future_risk_count:]
        for risk in new_risks:
            add_system_log(
                f"FUTURE RISK detected: {risk['turbine']} "
                f"{risk.get('horizon', '')} (Score: {risk['score']:.2f})"
            )

    if status == "DONE":
        add_system_log("Simulation FINISHED for all target turbines")
    elif status == "NO_DATA":
        add_system_log("No online prediction data found")

# --- ĐỌC DỮ LIỆU ĐỂ RENDER PHẦN THÂN ---
history = st.session_state.history_data
current_detection_history = st.session_state.get("current_detection_history_data", pd.DataFrame())

prediction_artifact_for_ui = st.session_state.get("active_prediction_artifact", prediction_artifact)

if prediction_artifact_for_ui.get("available"):
    artifact = prediction_artifact_for_ui
    artifact_source = artifact.get("artifact_source", "")
    source_labels = {
        "local_training_results": "local training result",
        "external_results": "external training result",
        "local_21_feature_model": "local 21-feature",
        "ml_forecast": "ML forecast",
    }
    source_label = source_labels.get(artifact_source, artifact_source or "unknown")
    st.subheader("Future Risk Summary")
    
    active_model_upper = model_choice.upper() if 'model_choice' in locals() else st.session_state.get("selected_model", "").upper()
    
    _ML_KEYWORDS = ["XGBOOST", "XGB", "RANDOM FOREST", "RANDOMFOREST", "RF_MODEL", " RF "]
    is_ml_future = any(kw in active_model_upper for kw in _ML_KEYWORDS) or active_model_upper.startswith("RF")

    for index, tid in enumerate(TARGET_TURBINES):
        if is_ml_future:
            # match asset_id exactly (safe across int/str formats)
            # match asset_id numerically to handle '10.0' / string variants
            sub = (
                current_detection_history[
                    pd.to_numeric(current_detection_history["asset_id"], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                    .eq(tid)
                ].copy()
                if not current_detection_history.empty
                else pd.DataFrame()
            )
            if sub.empty:
                with st.container(border=True):
                    st.markdown(f"⚪ **{TURBINE_LABELS[tid]}** | Status: `Waiting` | Step {st.session_state.get('current_detection_turbine_steps', {}).get(tid, 0)}")
                continue
            
            last = sub.iloc[-1]
            risk_horizon = st.session_state.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON)
            is_anom_now = int(last.get("pred_label", 0)) == 1
            risk_score = float(last.get("anomaly_score", 0.0))
            
            n_risk = int(sub["pred_label"].sum()) if "pred_label" in sub.columns else 0
            pct = (n_risk / len(sub) * 100) if len(sub) > 0 else 0
            
            if is_anom_now:
                status_icon = "🔴"
                value_text = "CRITICAL RISK DETECTED"
                delta_text = f"Emergency Action required: **STOP THE TURBINE IMMEDIATELY!** Continuous anomalies found in history."
            elif pct > 30 and risk_score > 0.60:
                status_icon = "🟠"
                value_text = f"URGENT RISK ({risk_horizon} Forecast)"
                delta_text = f"High probability of failure within next {risk_horizon}. **Schedule preventative maintenance soon.**"
            elif pct > 10 or risk_score > 0.40:
                status_icon = "🟡"
                value_text = f"MONITOR TRENDS"
                delta_text = f"Slight degradation noticed (Score: {risk_score:.2f} · Cumulative Alerts: {pct:.0f}%). **Check sensor trends.**"
            else:
                status_icon = "🟢"
                value_text = "STABLE OPERATION"
                delta_text = f"All operational parameters normal. Component is verified **safe for the next {risk_horizon}**."
        else:
            sub = history[history["asset_id"].eq(tid)].copy() if not history.empty else pd.DataFrame()
            if sub.empty:
                with st.container(border=True):
                    st.markdown(f"⚪ **{TURBINE_LABELS[tid]}** | Status: `Waiting` | Step {st.session_state.get('turbine_steps', {}).get(tid, 0)}")
                continue
            
            last = sub.iloc[-1]
            risk_horizon = last.get("future_horizon", st.session_state.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON))
            
            if "future_risk_score" in last.index and pd.notna(last.get("future_risk_score")):
                risk_score = last.get("future_risk_score")
                risk_label = int(last.get("future_risk_label", 0)) == 1
            else:
                if "future_risk_score" in sub.columns:
                    non_null = sub["future_risk_score"].dropna()
                    risk_score = non_null.iloc[-1] if not non_null.empty else float("nan")
                    risk_label = int(sub.loc[non_null.index[-1], "future_risk_label"]) == 1 if not non_null.empty else False
                else:
                    risk_score = float("nan")
                    risk_label = False
                    
            n_risk = int(sub["future_risk_label"].sum()) if "future_risk_label" in sub.columns else 0
            pct = (n_risk / len(sub) * 100) if len(sub) > 0 else 0
            risk_status = str(last.get("future_prediction_status", "Pending"))
            status_short = risk_status if len(risk_status) <= 24 else risk_status[:21] + "..."
            
            if pd.notna(risk_score):
                if risk_label:
                    status_icon = "🔴"
                    value_text = f"HIGH RISK ENCOUNTERED ({risk_horizon})"
                    delta_text = f"Risk Score: **{risk_score:.3f}** · Recurrent window anomalies: **{n_risk} alerts** ({pct:.0f}% of streaming window)."
                else:
                    status_icon = "🟢"
                    value_text = f"NORMAL CONDITION ({risk_horizon})"
                    delta_text = f"Risk Score: **{risk_score:.3f}** · The time-series window shows optimal mechanical health."
            else:
                status_icon = "🔵"
                value_text = "INITIALIZING SLIDING WINDOW"
                delta_text = f"Acquiring time-series data steps. Current pipeline status: `{status_short}`"
        
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"### {status_icon} {TURBINE_LABELS[tid]}")
                st.markdown(f"**{value_text}**")
            with c2:
                st.caption("FORECAST INSIGHTS & RECOMMENDED ACTION")
                st.markdown(delta_text)
st.divider()

_active_model_upper_cd = model_choice.upper() if 'model_choice' in locals() else st.session_state.get("selected_model", "").upper()
_ML_KEYWORDS_CD = ["XGBOOST", "XGB", "RANDOM FOREST", "RANDOMFOREST", "RF_MODEL", " RF "]
_is_ml_model_cd = any(kw in _active_model_upper_cd for kw in _ML_KEYWORDS_CD) or _active_model_upper_cd.startswith("RF")

if _is_ml_model_cd:
    
    st.subheader("Current Fault Detection")

    current_detection_cols = st.columns(5)
    for index, tid in enumerate(TARGET_TURBINES):
        # match asset_id exactly to avoid partial matches (10 vs 1)
        sub = (
            current_detection_history[
                pd.to_numeric(current_detection_history["asset_id"], errors="coerce").fillna(-1).astype(int).eq(tid)
            ].copy()
            if not current_detection_history.empty
            else pd.DataFrame()
        )
        if sub.empty:
            continue

        last = sub.iloc[-1]
        is_current_anom = int(last.get("pred_label", 0)) == 1
        current_detection_score = float(last.get("anomaly_score", 0.0))
        current_detection_cols[index].metric(
            label=TURBINE_LABELS[tid],
            value="ANOMALY" if is_current_anom else "NORMAL",
            delta=f"Score {current_detection_score:.3f}",
            delta_color="inverse" if is_current_anom else "normal",
        )

    st.divider()

col_title, col_select = st.columns([2.5, 1])
with col_select:
    
    selected_label = st.selectbox(
        "Choose Turbine for detail analysis",
        options=[TURBINE_LABELS[tid] for tid in TARGET_TURBINES],
        key="selected_detail_turbine_label",
        label_visibility="collapsed"
    )
    selected_asset = int(selected_label.split("-")[1])

with col_title:
    live_flag = "LIVE" if st.session_state.is_monitoring else "Paused"
    st.subheader(f"Live Sensor Trends — {TURBINE_LABELS[selected_asset]} ({live_flag})")


current_data = (
    history[
        pd.to_numeric(history["asset_id"], errors="coerce").fillna(-1).astype(int).eq(selected_asset)
    ].copy()
    if not history.empty
    else pd.DataFrame()
)

if not current_data.empty and chosen_sensors:
    valid_sensors = [col for col in chosen_sensors if col in current_data.columns]
    cd = current_data.sort_values("time_stamp").reset_index(drop=True)
    
    active_model_upper = model_choice.upper() if 'model_choice' in locals() else st.session_state.get("selected_model", "").upper()
    
    _ML_KEYWORDS = ["XGBOOST", "XGB", "RANDOM FOREST", "RANDOMFOREST", "RF_MODEL", " RF "]
    is_ml_future = any(kw in active_model_upper for kw in _ML_KEYWORDS) or active_model_upper.startswith("RF")

    label_col = "pred_label" if is_ml_future else "future_risk_label"
    score_col = "anomaly_score" if is_ml_future else "future_risk_score"
    model_type_text = "ML Prediction" if is_ml_future else "Future Risk"

    if is_ml_future:
        label_col = "pred_label"
        score_col = "anomaly_score"
        model_type_text = "ML Prediction"

        _cd_has_labels = label_col in cd.columns and cd[label_col].sum() > 0
        if not _cd_has_labels:
            _current_det = st.session_state.get("current_detection_history_data", pd.DataFrame())
            if not _current_det.empty:
                _tid_det = _current_det[
                    pd.to_numeric(_current_det["asset_id"], errors="coerce").fillna(-1).astype(int) == selected_asset
                ].copy()
                if not _tid_det.empty and "pred_label" in _tid_det.columns:
                    # Merge by sequence_id first (stable), then fall back to nearest timestamp
                    if "sequence_id" in cd.columns and "sequence_id" in _tid_det.columns:
                        _right = _tid_det[["sequence_id", "pred_label", "anomaly_score"]].drop_duplicates("sequence_id")
                        # Drop stale columns from cd before re-merging to avoid suffix collisions
                        cd = cd.drop(columns=[c for c in ("pred_label", "anomaly_score") if c in cd.columns], errors="ignore")
                        cd = pd.merge(cd, _right, on="sequence_id", how="left")
                    else:
                        # Nearest-timestamp merge (tolerance: 60 s)
                        _tid_det = _tid_det.copy()
                        _tid_det["_ts"] = pd.to_datetime(_tid_det["time_stamp"], errors="coerce")
                        cd = cd.copy()
                        cd["_ts"] = pd.to_datetime(cd["time_stamp"], errors="coerce")
                        cd = pd.merge_asof(
                            cd.sort_values("_ts"),
                            _tid_det[["_ts", "pred_label", "anomaly_score"]].sort_values("_ts"),
                            on="_ts",
                            direction="nearest",
                            tolerance=pd.Timedelta("60s"),
                        ).sort_values("time_stamp").reset_index(drop=True)
                        cd = cd.drop(columns=["_ts"], errors="ignore")

        # Ensure columns always exist with safe defaults after merge attempts
        if label_col not in cd.columns:
            cd[label_col] = 0
        if score_col not in cd.columns:
            cd[score_col] = 0.0
        cd[label_col] = pd.to_numeric(cd[label_col], errors="coerce").fillna(0).astype(int)
        cd[score_col] = pd.to_numeric(cd[score_col], errors="coerce").fillna(0.0)
    else:
        label_col = "future_risk_label"
        score_col = "future_risk_score"
        model_type_text = "Future Risk"

    fault_pts = cd[cd["status_type_id"].eq(1)] if "status_type_id" in cd.columns else pd.DataFrame()
    
    risk_pts = (
        cd[(cd[label_col].eq(1)) & cd[score_col].notna()]
        if {label_col, score_col}.issubset(cd.columns)
        else pd.DataFrame()
    )

    if valid_sensors:
        fig = make_subplots(
            rows=len(valid_sensors),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            subplot_titles=[get_sensor_label(col) for col in valid_sensors],
        )

        for index, col in enumerate(valid_sensors):
            row_idx = index + 1
            color = TRACE_COLORS[index % len(TRACE_COLORS)]
            label_name = get_sensor_label(col)
            unit = get_sensor_unit(col)
            y_title = unit if unit else label_name[:12]

            fig.add_trace(
                go.Scatter(
                    x=cd["time_stamp"],
                    y=cd[col],
                    mode="lines",
                    name=label_name,
                    line=dict(color=color, width=1.8),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{label_name}</b><br>"
                        "%{x|%Y-%m-%d %H:%M}<br>"
                        f"Value: %{{y:.3f}} {unit}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

            if not fault_pts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=fault_pts["time_stamp"],
                        y=fault_pts[col],
                        mode="markers",
                        marker=dict(color=STATUS_COLORS["Anomaly"], size=5),
                        name="Actual Fault Label" if row_idx == 1 else None,
                        showlegend=(row_idx == 1),
                        hovertemplate=(
                            "<b>ACTUAL FAULT LABEL</b><br>%{x|%H:%M}<br>"
                            f"{label_name}: %{{y:.3f}}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=1,
                )

            if not risk_pts.empty:
                fig.add_trace(
                    go.Scatter(
                        x=risk_pts["time_stamp"],
                        y=risk_pts[col],
                        mode="markers",
                        marker=dict(color=STATUS_COLORS["Warning"], size=7, symbol="diamond"),
                        name=model_type_text if row_idx == 1 else None,
                        showlegend=(row_idx == 1),
                        customdata=risk_pts[score_col],
                        hovertemplate=(
                            f"<b>{model_type_text.upper()}</b><br>%{{x|%H:%M}}<br>"
                            f"{label_name}: %{{y:.3f}}<br>"
                            "Score: %{customdata:.3f}<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=1,
                )

            for start_time, end_time in _fault_ranges(cd):
                fig.add_vrect(
                    x0=start_time,
                    x1=end_time,
                    fillcolor="rgba(239,68,68,0.10)",
                    line_width=0,
                    row=row_idx,
                    col=1,
                )

            fig.update_yaxes(title_text=y_title, title_font_size=10, row=row_idx, col=1)

        fig.update_layout(
            height=max(350, 165 * len(valid_sensors)),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            margin=dict(l=60, r=20, t=35, b=40),
            legend=dict(orientation="h", y=1.01, yanchor="bottom", x=1, xanchor="right", font_size=11),
            transition=dict(duration=0),
        )
        chart_key = f"live_sensor_trends_{selected_asset}_{'_'.join(valid_sensors)}"
        fig.update_layout(
            uirevision=chart_key,
            datarevision=_chart_revision(cd),
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", tickformat="%m/%d %H:%M", tickangle=-30)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")

        _render_live_chart(fig, chart_key)

        total = len(cd)
        n_risk = int(cd[label_col].sum()) if label_col in cd.columns else 0
        st.caption(
            f"{TURBINE_LABELS[selected_asset]} - "
            f"{n_risk}/{total} predicted anomaly points ({n_risk / total * 100:.1f}%) - "
            f"Model Type: {model_type_text} · "
            f"Horizon: {st.session_state.get('prediction_horizon', DEFAULT_PREDICTION_HORIZON)}"
        )
    else:
        st.warning("No selected chart sensors are available in this stream.")
else:
    st.warning("No data is streaming yet. Press Start to begin.")

st.divider()

if st.session_state.is_monitoring:
    time.sleep(SIMULATION_DELAY)
    st.rerun()