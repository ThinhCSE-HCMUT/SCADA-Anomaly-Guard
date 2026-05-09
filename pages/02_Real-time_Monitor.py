"""
Page 02: Real-time Monitor
Giám sát song song 5 turbines (0, 10, 11, 13, 21)
"""
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from datetime import datetime

from src.i18n import get_language, t

# 🔥 ĐƯA LÊN TRÊN CÙNG: Phải đặt trước tất cả các import module nội bộ có chứa lệnh Streamlit
get_language()
st.set_page_config(
    page_title=t("monitor.title"),  # Tên hiển thị trên tab trình duyệt
    layout="wide",                 # Ép trang luôn ở chế độ Full Screen
    initial_sidebar_state="expanded" # Ép sidebar luôn mở ra
)

from src.sidebar import render_sidebar
from src.config import (
    TARGET_TURBINES, TURBINE_LABELS, FEATURE_COLS, CHART_SENSOR_COLS,
    get_sensor_label, get_sensor_unit, AVAILABLE_MODELS, SIMULATION_DELAY,
    TRACE_COLORS, STATUS_COLORS
)

# 🛠️ ĐÃ SỬA: Import đúng hàm từ đúng "nhà" của nó
from src.data_loader import load_data
from src.model_manager import load_model
from src.simulation import run_simulation_step

# ====================== SESSION STATE ======================
def add_system_log(event_message):
    """Hàm tiện ích để ghi lại sự kiện mới vào hệ thống"""
    if "system_logs" not in st.session_state:
        st.session_state.system_logs = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.system_logs.insert(0, {"Timestamp": timestamp, "Event": event_message})

def _init_state():
    defaults = {
        "is_monitoring"  : False,
        "turbine_steps"  : {tid: 0 for tid in TARGET_TURBINES},
        "history_data"   : pd.DataFrame(),
        "anomaly_records": [],
        "selected_model" : next(iter(AVAILABLE_MODELS)),
        "stream_done"    : {tid: False for tid in TARGET_TURBINES},
        "system_logs"    : [{"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Event": t("alerts.monitor_loaded")}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
raw_df = load_data()

# ====================== PAGE HEADER ======================
render_sidebar()

st.title(t("monitor.title"))
st.markdown(t("monitor.description"))
st.divider()


# ====================== CONTROLS ======================
col1, col2, col3 = st.columns([1.2, 2.5, 1.2])

with col1:
    selected_label = st.selectbox(
        t("monitor.choose_turbine"),
        options=[TURBINE_LABELS[tid] for tid in TARGET_TURBINES],
    )
    selected_asset = int(selected_label.split("-")[1])

with col2:
    chosen_sensors = st.multiselect(
        t("monitor.choose_sensors"),
        options=CHART_SENSOR_COLS,
        default=CHART_SENSOR_COLS[:5],
        format_func=get_sensor_label,
        max_selections=10,
    )
    if not chosen_sensors:
        chosen_sensors = CHART_SENSOR_COLS[:4]

with col3:
    model_choice = st.selectbox(
        t("monitor.choose_model"),
        options=list(AVAILABLE_MODELS.keys()),
        disabled=st.session_state.is_monitoring,
    )
    active_model = load_model(model_choice)
    if active_model is not None:
        st.caption(t("monitor.model_loaded", model=model_choice))
    else:
        st.caption(f"⚠️ {t('monitor.model_fallback')}")

# Thêm một chút khoảng trống cho đỡ sát vào nhau
st.markdown("<br>", unsafe_allow_html=True)

# --- HÀNG 2: NÚT START / STOP ---
# Dùng 2 cột rỗng ở hai bên (tỷ lệ 2) để ép 2 nút (tỷ lệ 1) vào giữa màn hình
_, col_start, col_stop, _ = st.columns([2, 1, 1, 2])

with col_start:
    start_clicked = st.button(
        t("monitor.start"), type="primary",
        use_container_width=True,
        disabled=st.session_state.is_monitoring or raw_df.empty,
    )

with col_stop:
    stop_clicked = st.button(
        t("monitor.stop"),
        use_container_width=True,
        disabled=not st.session_state.is_monitoring,
    )

if start_clicked:
    add_system_log(t("alerts.simulation_started", model=model_choice))
    st.session_state.update({
        "is_monitoring"  : True,
        "turbine_steps"  : {tid: 0 for tid in TARGET_TURBINES},
        "history_data"   : pd.DataFrame(),
        "anomaly_records": [],
        "selected_model" : model_choice,
        "stream_done"    : {tid: False for tid in TARGET_TURBINES},
    })
    st.rerun()

if stop_clicked:
    add_system_log(t("alerts.simulation_stopped"))
    st.session_state.is_monitoring = False
    st.rerun()

st.divider()

# ====================== SIMULATION STEP & AUTO RERUN ======================
if st.session_state.is_monitoring:
    prev_anomaly_count = len(st.session_state.anomaly_records)
    
    status = run_simulation_step()
    
    current_anomaly_count = len(st.session_state.anomaly_records)
    if current_anomaly_count > prev_anomaly_count:
        new_anoms = st.session_state.anomaly_records[prev_anomaly_count:]
        for anom in new_anoms:
            add_system_log(t("alerts.new_anomaly", turbine=anom['turbine'], score=anom['score']))

# ====================== DERIVED DATA ======================
history      = st.session_state.history_data
current_data = (
    history[history['asset_id'] == selected_asset].copy()
    if not history.empty else pd.DataFrame()
)

# ====================== STATUS METRICS ======================
if not current_data.empty:
    latest  = current_data.iloc[-1]
    is_anom = int(latest.get('pred_label', 0)) == 1
    score   = float(latest.get('anomaly_score', 0.0))
    n_mine  = sum(
        r['turbine'] == TURBINE_LABELS[selected_asset]
        for r in st.session_state.anomaly_records
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        t("monitor.turbine_status", turbine=TURBINE_LABELS[selected_asset]),
        f"🔴 {t('status.anomaly_caps')}" if is_anom else f"🟢 {t('status.normal_caps')}",
        delta=f"{t('common.score')}: {score:.3f}",
        delta_color="inverse",
    )
    m2.metric(t("common.model"), st.session_state.selected_model)
    m3.metric(t("monitor.alerts_for_turbine", turbine=TURBINE_LABELS[selected_asset]), n_mine)
    m4.metric(t("monitor.total_alerts_all"), len(st.session_state.anomaly_records))
else:
    st.info(f"💡 {t('monitor.press_start')}")

st.divider()

# ====================== TURBINE OVERVIEW (5 turbines) ======================
st.subheader(t("monitor.latest_status"))

ov_cols = st.columns(5)

for i, tid in enumerate(TARGET_TURBINES):
    sub = history[history['asset_id'] == tid].copy() if not history.empty else pd.DataFrame()
    
    if sub.empty:
        ov_cols[i].metric(
            label=f"⏳ {TURBINE_LABELS[tid]}",
            value=t("common.waiting_data"),
            delta=f"{t('monitor.step')} {st.session_state.turbine_steps.get(tid, 0)}"
        )
        continue

    last = sub.iloc[-1]
    is_a = int(last.get('pred_label', 0)) == 1
    n_a = int(sub['pred_label'].sum()) if 'pred_label' in sub.columns else 0
    pct = (n_a / len(sub) * 100) if len(sub) > 0 else 0

    if len(sub) >= 2:
        prev_n_a = int(sub.iloc[-2]['pred_label']) if 'pred_label' in sub.columns else 0
        alert_delta = n_a - prev_n_a
        alert_delta_str = f"{alert_delta:+d}" if alert_delta != 0 else None
        alert_delta_color = "inverse" if alert_delta > 0 else "normal"
    else:
        alert_delta_str = None
        alert_delta_color = "off"

    ov_cols[i].metric(
        label=f"{'🔴' if is_a else '🟢'} {TURBINE_LABELS[tid]}",
        value=t("status.anomaly_caps") if is_a else t("status.normal_caps"),
        delta=f"{n_a} {t('monitor.alerts_suffix')} ({pct:.0f}%) {alert_delta_str if alert_delta_str else ''}",
        delta_color=alert_delta_color
    )

st.divider()

# ====================== LIVE SENSOR CHART ======================
live_flag = f" 🔴 {t('status.live')}" if st.session_state.is_monitoring else f" ⏸ {t('status.paused')}"
st.subheader(t("monitor.live_sensor_trends", turbine=TURBINE_LABELS[selected_asset], flag=live_flag))

if not current_data.empty and chosen_sensors:
    valid_sensors = [c for c in chosen_sensors if c in current_data.columns]
    cd = current_data.sort_values('time_stamp').reset_index(drop=True)

    anomaly_ranges = []
    in_anom, anom_start = False, None
    for idx, row in cd.iterrows():
        is_a = int(row.get('pred_label', 0)) == 1
        if is_a and not in_anom:
            in_anom, anom_start = True, row['time_stamp']
        elif not is_a and in_anom:
            in_anom = False
            anomaly_ranges.append((anom_start, cd.iloc[idx - 1]['time_stamp']))
    if in_anom:
        anomaly_ranges.append((anom_start, cd.iloc[-1]['time_stamp']))

    anom_pts = cd[cd['pred_label'] == 1] if 'pred_label' in cd.columns else pd.DataFrame()

    n_rows = len(valid_sensors)
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        subplot_titles=[get_sensor_label(c) for c in valid_sensors],
    )

    for i, col in enumerate(valid_sensors):
        row_idx    = i + 1
        color      = TRACE_COLORS[i % len(TRACE_COLORS)]
        label_name = get_sensor_label(col)
        unit       = get_sensor_unit(col)
        y_title    = unit if unit else label_name[:12]

        fig.add_trace(
            go.Scatter(
                x=cd['time_stamp'], y=cd[col],
                mode='lines',
                name=label_name,
                line=dict(color=color, width=1.8),
                showlegend=False,
                hovertemplate=(
                    f"<b>{label_name}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    f"{t('common.value')}: %{{y:.3f}} {unit}<extra></extra>"
                ),
            ),
            row=row_idx, col=1,
        )

        if not anom_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=anom_pts['time_stamp'], y=anom_pts[col],
                    mode='markers',
                    marker=dict(color=STATUS_COLORS["Anomaly"], size=5),
                    name=t("monitor.anomaly_label") if row_idx == 1 else None,
                    showlegend=(row_idx == 1),
                    customdata=anom_pts['anomaly_score'],
                    hovertemplate=(
                        f"<b>{t('monitor.anomaly_label').upper()}</b><br>%{{x|%H:%M}}<br>"
                        f"{label_name}: %{{y:.3f}}<br>"
                        f"{t('common.score')}: %{{customdata:.3f}}<extra></extra>"
                    ),
                ),
                row=row_idx, col=1,
            )

        for t0, t1 in anomaly_ranges:
            fig.add_vrect(
                x0=t0, x1=t1,
                fillcolor="rgba(239,68,68,0.10)",
                line_width=0,
                row=row_idx, col=1,
            )

        fig.update_yaxes(title_text=y_title, title_font_size=10, row=row_idx, col=1)

    fig.update_layout(
        height=max(350, 165 * n_rows),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=35, b=40),
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=1, xanchor="right", font_size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", tickformat="%m/%d %H:%M", tickangle=-30)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")

    st.plotly_chart(fig, use_container_width=True)

    total  = len(cd)
    n_anom = int(cd['pred_label'].sum()) if 'pred_label' in cd.columns else 0
    st.caption(
        f"📌 **{TURBINE_LABELS[selected_asset]}** — "
        f"{n_anom}/{total} {t('status.anomaly')} ({n_anom/total*100:.1f}%) · "
        f"{t('common.model')}: **{st.session_state.selected_model}** · "
    )

else:
    st.warning(t("monitor.no_data"))

st.divider()

# ====================== ANOMALY LOG ======================
if st.session_state.anomaly_records:
    st.subheader(t("monitor.anomaly_log"))
    log_df = (
        pd.DataFrame(st.session_state.anomaly_records)
        .sort_values('time', ascending=False)
        .head(50)
        .reset_index(drop=True)
    )
    log_df['time'] = log_df['time'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(
        log_df.rename(columns={'time': t("table.time_stamp"), 'turbine': t("table.turbine"), 'score': t("table.anomaly_score")}),
        use_container_width=True, hide_index=True,
        column_config={
            t("table.anomaly_score"): st.column_config.ProgressColumn(
                t("table.anomaly_score"), min_value=0.0, max_value=1.0, format="%.3f",
            )
        },
    )

# ====================== TRIGGER AUTO RERUN ======================
if st.session_state.is_monitoring:
    time.sleep(SIMULATION_DELAY)
    st.rerun()
