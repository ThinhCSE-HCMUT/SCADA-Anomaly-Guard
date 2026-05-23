"""Page 02: all-turbine real-time monitor."""

from datetime import datetime
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

col1, col2, col3, col4 = st.columns([1.1, 2.3, 1.2, 1.1])

with col1:
    selected_label = st.selectbox(
        "Choose Turbine for detail",
        options=[TURBINE_LABELS[tid] for tid in TARGET_TURBINES],
    )
    selected_asset = int(selected_label.split("-")[1])

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
    model_choice = st.selectbox(
        "Current Detection Model (ML/DL)",
        options=list(AVAILABLE_MODELS.keys()),
        disabled=st.session_state.is_monitoring,
    )
    active_model = load_model(model_choice)
    if active_model is not None:
        st.caption(f"Model loaded: **{model_choice}**")
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
    prediction_artifact = get_prediction_artifact(prediction_horizon)
    if prediction_artifact.get("available"):
        artifact_source = prediction_artifact.get("artifact_source", "")
        source_labels = {
            "local_training_results": "local training result",
            "external_results": "external training result",
            "local_21_feature_model": "local 21-feature",
        }
        source_label = source_labels.get(artifact_source, artifact_source or "unknown")
        st.caption(
            f"Best trained model: **{prediction_artifact['model_display_name']}** "
            f"@ {prediction_artifact['threshold']:.3f} ({source_label})"
        )
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

history = st.session_state.history_data
current_detection_history = st.session_state.get("current_detection_history_data", pd.DataFrame())
current_data = (
    history[history["asset_id"].eq(selected_asset)].copy()
    if not history.empty
    else pd.DataFrame()
)

if not current_data.empty:
    latest = current_data.iloc[-1]
    ready_rows = (
        int(current_data["future_risk_score"].notna().sum())
        if "future_risk_score" in current_data.columns
        else 0
    )
    risk_score = latest.get("future_risk_score", float("nan"))
    risk_label = int(latest.get("future_risk_label", 0)) == 1
    risk_status = latest.get("future_prediction_status", "Pending")
    risk_model = latest.get("future_model", "")
    risk_horizon = latest.get("future_horizon", st.session_state.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON))
    risk_threshold = latest.get("future_threshold", float("nan"))
    future_alerts_selected = sum(
        record["turbine"] == TURBINE_LABELS[selected_asset]
        for record in st.session_state.get("future_risk_records", [])
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{TURBINE_LABELS[selected_asset]} raw rows", len(current_data), delta="Wind Farm A")
    m2.metric("Future Horizon", risk_horizon)
    m3.metric("Ready Windows", ready_rows, delta=f"Need {PREDICTION_WINDOW_STEPS} rows")
    m4.metric("Online Source", "All events", delta="21-feature DL path")

    p1, p2, p3, p4 = st.columns(4)
    if pd.notna(risk_score):
        p1.metric(
            f"{risk_horizon} future score",
            _format_percent_score(risk_score),
            delta=_format_threshold_delta(risk_threshold),
            delta_color="inverse" if risk_label else "normal",
        )
    else:
        p1.metric(f"{risk_horizon} future score", "PENDING", delta=str(risk_status), delta_color="off")
    p2.metric("Prediction Model", risk_model if risk_model else "Unavailable")
    p3.metric(f"Future Alerts {TURBINE_LABELS[selected_asset]}", future_alerts_selected)
    p4.metric("Future Alerts (all WT)", len(st.session_state.get("future_risk_records", [])))
else:
    st.info("Press Start to begin the all-turbine simulation.")

st.divider()

st.subheader("Online Prediction Stream - 5 Turbines")
ov_cols = st.columns(5)

for index, tid in enumerate(TARGET_TURBINES):
    sub = history[history["asset_id"].eq(tid)].copy() if not history.empty else pd.DataFrame()

    if sub.empty:
        ov_cols[index].metric(
            label=TURBINE_LABELS[tid],
            value="Waiting",
            delta=f"Step {st.session_state.get('turbine_steps', {}).get(tid, 0)}",
            delta_color="off",
        )
        continue

    last = sub.iloc[-1]
    is_high_risk = int(last.get("future_risk_label", 0)) == 1
    risk_score = last.get("future_risk_score", float("nan"))
    risk_status = last.get("future_prediction_status", "Pending")
    n_risk = int(sub["future_risk_label"].sum()) if "future_risk_label" in sub.columns else 0
    pct = (n_risk / len(sub) * 100) if len(sub) > 0 else 0
    if pd.notna(risk_score):
        ov_cols[index].metric(
            label=TURBINE_LABELS[tid],
            value=_format_percent_score(risk_score),
            delta=f"{n_risk} future alerts ({pct:.0f}%)",
            delta_color="inverse" if is_high_risk else "normal",
        )
    else:
        ov_cols[index].metric(
            label=TURBINE_LABELS[tid],
            value="PENDING",
            delta=str(risk_status),
            delta_color="off",
        )

st.divider()

st.subheader("Future Risk Overview")
risk_cols = st.columns(5)

for index, tid in enumerate(TARGET_TURBINES):
    sub = history[history["asset_id"].eq(tid)].copy() if not history.empty else pd.DataFrame()

    if sub.empty:
        risk_cols[index].metric(
            label=TURBINE_LABELS[tid],
            value="Pending",
            delta=f"Step {st.session_state.get('turbine_steps', {}).get(tid, 0)}",
            delta_color="off",
        )
        continue

    last = sub.iloc[-1]
    risk_score = last.get("future_risk_score", float("nan"))
    risk_label = int(last.get("future_risk_label", 0)) == 1
    risk_horizon = last.get("future_horizon", st.session_state.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON))
    risk_status = last.get("future_prediction_status", "Pending")
    risk_threshold = last.get("future_threshold", float("nan"))

    if pd.notna(risk_score):
        risk_cols[index].metric(
            label=f"{TURBINE_LABELS[tid]} {risk_horizon}",
            value=_format_percent_score(risk_score),
            delta=_format_threshold_delta(risk_threshold),
            delta_color="inverse" if risk_label else "normal",
        )
    else:
        risk_cols[index].metric(
            label=f"{TURBINE_LABELS[tid]} {risk_horizon}",
            value="PENDING",
            delta=str(risk_status),
            delta_color="off",
        )

st.divider()

st.subheader("Current Fault Detection")
st.caption("ML and DL current-detection models run on df_simulation.csv with 105 features.")

current_detection_cols = st.columns(5)
for index, tid in enumerate(TARGET_TURBINES):
    sub = (
        current_detection_history[current_detection_history["asset_id"].eq(tid)].copy()
        if not current_detection_history.empty
        else pd.DataFrame()
    )

    if sub.empty:
        current_detection_cols[index].metric(
            label=TURBINE_LABELS[tid],
            value="Waiting",
            delta=f"Step {st.session_state.get('current_detection_turbine_steps', {}).get(tid, 0)}",
            delta_color="off",
        )
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

live_flag = "LIVE" if st.session_state.is_monitoring else "Paused"
st.subheader(f"Live Sensor Trends - {TURBINE_LABELS[selected_asset]} ({live_flag})")

if not current_data.empty and chosen_sensors:
    valid_sensors = [col for col in chosen_sensors if col in current_data.columns]
    cd = current_data.sort_values("time_stamp").reset_index(drop=True)
    fault_pts = cd[cd["status_type_id"].eq(1)] if "status_type_id" in cd.columns else pd.DataFrame()
    risk_pts = (
        cd[(cd["future_risk_label"].eq(1)) & cd["future_risk_score"].notna()]
        if {"future_risk_label", "future_risk_score"}.issubset(cd.columns)
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
                        name="Future Risk" if row_idx == 1 else None,
                        showlegend=(row_idx == 1),
                        customdata=risk_pts["future_risk_score"],
                        hovertemplate=(
                            "<b>FUTURE RISK</b><br>%{x|%H:%M}<br>"
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
        n_risk = int(cd["future_risk_label"].sum()) if "future_risk_label" in cd.columns else 0
        st.caption(
            f"{TURBINE_LABELS[selected_asset]} - "
            f"{n_risk}/{total} future-risk points ({n_risk / total * 100:.1f}%) - "
            f"Horizon: {st.session_state.get('prediction_horizon', DEFAULT_PREDICTION_HORIZON)}"
        )
    else:
        st.warning("No selected chart sensors are available in this stream.")
else:
    st.warning("No data is streaming yet. Press Start to begin.")

st.divider()

if st.session_state.anomaly_records:
    st.subheader("Current Fault Detection Log")
    log_df = (
        pd.DataFrame(st.session_state.anomaly_records)
        .sort_values("time", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )
    log_df["time"] = log_df["time"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        log_df.rename(columns={"time": "Time stamp", "turbine": "Turbine", "score": "Anomaly Score"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Anomaly Score": st.column_config.ProgressColumn(
                "Anomaly Score",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            )
        },
    )

if st.session_state.get("future_risk_records"):
    st.subheader("Future Risk Log")
    risk_log_df = (
        pd.DataFrame(st.session_state.future_risk_records)
        .sort_values("time", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )
    risk_log_df["time"] = pd.to_datetime(risk_log_df["time"]).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        risk_log_df.rename(
            columns={
                "time": "Time stamp",
                "turbine": "Turbine",
                "event_id": "Event",
                "event_label": "Event Label",
                "event_description": "Fault Type",
                "score": "Future Risk Score",
                "horizon": "Horizon",
                "model": "Prediction Model",
                "threshold": "Threshold",
            }
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Future Risk Score": st.column_config.ProgressColumn(
                "Future Risk Score",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            )
        },
    )

if st.session_state.is_monitoring:
    time.sleep(SIMULATION_DELAY)
    st.rerun()
