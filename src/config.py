"""
Configuration file for SCADA Anomaly Detection Dashboard (Wind Turbine).
This file contains all paths, model settings, sensor definitions,
and visualization constants used throughout the application.
"""

from pathlib import Path

# ====================== PROJECT PATHS ======================
ROOT_DIR         = Path(__file__).parent.parent
DATA_DIR         = ROOT_DIR / "data"
MODELS_DIR       = ROOT_DIR / "models"
ASSETS_DIR       = ROOT_DIR / "assets"

# Local copied training-result artifacts used for sequence prediction.
PREDICTION_RESULTS_ROOT = MODELS_DIR / "DeepLearning"

# External processing export used for online feature metadata.
TRAINING_REPO_DIR = Path(r"D:\Final Project\scada-fault-prediction")
PREDICTION_CLASSIFIER_EXPORT_DIR = (
    TRAINING_REPO_DIR
    / "Dataset"
    / "processed"
    / "sequence_exports"
    / "window_24h"
    / "classifier"
)

# Local per-asset 21-feature scalers used by the dashboard sequence models.
PREDICTION_SCALER_DIR = MODELS_DIR / "scaler-prediction-model" / "scalers"
DETECTION_SCALER_DIR  = MODELS_DIR / "scaler-detection-model" / "scalers"

# Data files
# df_simulation.csv is the flat-feature stream for current fault detection.
REALTIME_DATA_PATH = DATA_DIR / "df_simulation.csv"
TEST_DATA_PATH     = DATA_DIR / "test_data.csv"
SAMPLE_DATA_PATH   = DATA_DIR / "sample_data.csv"
RAW_WIND_FARM_A_DIR = DATA_DIR / "Wind Farm A"
RAW_WIND_FARM_A_DATASETS_DIR = RAW_WIND_FARM_A_DIR / "datasets"
RAW_WIND_FARM_A_EVENT_INFO = RAW_WIND_FARM_A_DIR / "event_info.csv"
LOCAL_PREDICTION_FEATURE_FILE = DATA_DIR / "feature_screening_per_event" / "final_21_features.csv"
RAW_PREDICTION_SPLIT = "prediction"

# ====================== MODEL CONFIGURATION ======================
AVAILABLE_MODELS = {
    "XGBoost"                      : "Baseline/XGBoost/xgb_model.pkl",
    "Random Forest"                : "Baseline/RandomForest/rf_model.pkl",
    "LSTM"                         : "DeepLearning/results_detection_input_window_24h/window_24h/classifier/lstm/model.keras",
    "GRU"                          : "DeepLearning/results_detection_input_window_24h/window_24h/classifier/gru/model.keras",
    "CNN - LSTM"                   : "DeepLearning/results_detection_input_window_24h/window_24h/classifier/cnn_lstm/model.keras",
    "CNN - GRU"                    : "DeepLearning/results_detection_input_window_24h/window_24h/classifier/cnn_gru/model.keras"
}

MODEL_THRESHOLDS = {
    "XGBoost": {
        "Current": 0.5,
        "12h": 0.5,
        "24h": 0.5,
        "36h": 0.5,
        "48h": 0.5,
        "72h": 0.5
    },
    "Random Forest": {
        "Current": 0.5,
        "12h": 0.5,
        "24h": 0.5,
        "36h": 0.5,
        "48h": 0.5,
        "72h": 0.5
    },
    "LSTM": {
        "Current": 0.353,
        "12h": 0.405,
        "24h": 0.410,
        "36h": 0.425,
        "48h": 0.439,
        "72h": 0.422
    },
    "GRU": {
        "Current": 0.084,
        "12h": 0.274,
        "24h": 0.365,
        "36h": 0.398,
        "48h": 0.260,
        "72h": 0.544
    },
    "CNN - LSTM": {
        "Current": 0.168,
        "12h": 0.377,
        "24h": 0.313,
        "36h": 0.237,
        "48h": 0.471,
        "72h": 0.317
    },
    "CNN - GRU": {
        "Current": 0.069,
        "12h": 0.555,
        "24h": 0.680,
        "36h": 0.335,
        "48h": 0.258,
        "72h": 0.392
    }
}

ML_SCALER_PATH = MODELS_DIR / "Baseline" / "ML_scaler" / "scada_scaler_full.pkl"

XGBOOST_FORECAST_OPTIONS = {
    "Current": "Baseline/XGBoost/xgb_model.pkl",
    "12h": "Baseline/XGBoost/xgb_model_12h.pkl",
    "24h": "Baseline/XGBoost/xgb_model_24h.pkl",
    "36h": "Baseline/XGBoost/xgb_model_36h.pkl",
    "48h": "Baseline/XGBoost/xgb_model_48h.pkl",
    "72h": "Baseline/XGBoost/xgb_model_72h.pkl",
}

XGBOOST_FORECAST_MODEL_PATHS = {
    horizon: str(MODELS_DIR / filename)
    for horizon, filename in XGBOOST_FORECAST_OPTIONS.items()
}

RF_FORECAST_OPTIONS = {
    "Current": "Baseline/RandomForest/rf_model.pkl",
    "12h": "Baseline/RandomForest/rf_model_12h.pkl",
    "24h": "Baseline/RandomForest/rf_model_24h.pkl",
    "36h": "Baseline/RandomForest/rf_model_36h.pkl",
    "48h": "Baseline/RandomForest/rf_model_48h.pkl",
    "72h": "Baseline/RandomForest/rf_model_72h.pkl",
}

RF_FORECAST_MODEL_PATHS = {
    horizon: str(MODELS_DIR / filename)
    for horizon, filename in RF_FORECAST_OPTIONS.items()
}

DL_FORECAST_OPTIONS = {
    "LSTM": {
        "Current": "DeepLearning/results_detection_input_window_24h/window_24h/classifier/lstm/model.keras",
        "12h": "DeepLearning/results_prediction_focal_12h_window_24h_21f/window_24h/classifier/lstm/model.keras",
        "24h": "DeepLearning/results_prediction_focal_24h_window_24h_21f/window_24h/classifier/lstm/model.keras",
        "36h": "DeepLearning/results_prediction_focal_36h_window_24h_21f/window_24h/classifier/lstm/model.keras",
        "48h": "DeepLearning/results_prediction_focal_48h_window_24h_21f/window_24h/classifier/lstm/model.keras",
        "72h": "DeepLearning/results_prediction_focal_72h_window_24h_21f/window_24h/classifier/lstm/model.keras",
    },
    "GRU": {
        "Current": "DeepLearning/results_detection_input_window_24h/window_24h/classifier/gru/model.keras",
        "12h": "DeepLearning/results_prediction_focal_12h_window_24h_21f/window_24h/classifier/gru/model.keras",
        "24h": "DeepLearning/results_prediction_focal_24h_window_24h_21f/window_24h/classifier/gru/model.keras",
        "36h": "DeepLearning/results_prediction_focal_36h_window_24h_21f/window_24h/classifier/gru/model.keras",
        "48h": "DeepLearning/results_prediction_focal_48h_window_24h_21f/window_24h/classifier/gru/model.keras",
        "72h": "DeepLearning/results_prediction_focal_72h_window_24h_21f/window_24h/classifier/gru/model.keras",
    },
    "CNN - LSTM": {
        "Current": "DeepLearning/results_detection_input_window_24h/window_24h/classifier/cnn_lstm/model.keras",
        "12h": "DeepLearning/results_prediction_focal_12h_window_24h_21f/window_24h/classifier/cnn_lstm/model.keras",
        "24h": "DeepLearning/results_prediction_focal_24h_window_24h_21f/window_24h/classifier/cnn_lstm/model.keras",
        "36h": "DeepLearning/results_prediction_focal_36h_window_24h_21f/window_24h/classifier/cnn_lstm/model.keras",
        "48h": "DeepLearning/results_prediction_focal_48h_window_24h_21f/window_24h/classifier/cnn_lstm/model.keras",
        "72h": "DeepLearning/results_prediction_focal_72h_window_24h_21f/window_24h/classifier/cnn_lstm/model.keras",
    },
    "CNN - GRU": {
        "Current": "DeepLearning/results_detection_input_window_24h/window_24h/classifier/cnn_gru/model.keras",
        "12h": "DeepLearning/results_prediction_focal_12h_window_24h_21f/window_24h/classifier/cnn_gru/model.keras",
        "24h": "DeepLearning/results_prediction_focal_24h_window_24h_21f/window_24h/classifier/cnn_gru/model.keras",
        "36h": "DeepLearning/results_prediction_focal_36h_window_24h_21f/window_24h/classifier/cnn_gru/model.keras",
        "48h": "DeepLearning/results_prediction_focal_48h_window_24h_21f/window_24h/classifier/cnn_gru/model.keras",
        "72h": "DeepLearning/results_prediction_focal_72h_window_24h_21f/window_24h/classifier/cnn_gru/model.keras",
    },
}

DL_FORECAST_MODEL_PATHS = {
    model_name: {
        horizon: str(MODELS_DIR / filename)
        for horizon, filename in horizon_map.items()
    }
    for model_name, horizon_map in DL_FORECAST_OPTIONS.items()
}

PREDICTION_HORIZON_RUNS = {
    "12h": "results_prediction_focal_12h_window_24h_21f",
    "24h": "results_prediction_focal_24h_window_24h_21f",
    "36h": "results_prediction_focal_36h_window_24h_21f",
    "48h": "results_prediction_focal_48h_window_24h_21f",
    "72h": "results_prediction_focal_72h_window_24h_21f",
}

DEFAULT_PREDICTION_MODEL_BY_HORIZON = {
    "12h": "CNN - LSTM",
    "24h": "CNN - LSTM",
    "36h": "CNN - LSTM",
    "48h": "CNN - LSTM",
    "72h": "CNN - LSTM",
}

PREDICTION_HORIZONS = list(PREDICTION_HORIZON_RUNS.keys())
DEFAULT_PREDICTION_HORIZON = "24h"
PREDICTION_WINDOW_STEPS = 144

# Full resolved paths — used by load_model()
MODEL_PATHS = {
    name: str(MODELS_DIR / filename)
    for name, filename in AVAILABLE_MODELS.items()
}

DEFAULT_MODEL     = "CNN - LSTM"
ANOMALY_THRESHOLD = 0.75
WARNING_THRESHOLD = 0.45

# ====================== TURBINE CONFIGURATION ======================
TARGET_TURBINES = [0, 10, 11, 13, 21]
TURBINE_LABELS  = {tid: f"WT-{tid:03d}" for tid in TARGET_TURBINES}

# Representative raw Wind Farm A events used by the Real-time Monitor stream.
# Each target turbine contributes one event instead of streaming all 22 events.
REALTIME_REPRESENTATIVE_EVENTS = {
    0: {
        "event_id": 26,
        "event_label": "anomaly",
        "fault_type": "Hydraulic group",
    },
    10: {
        "event_id": 40,
        "event_label": "anomaly",
        "fault_type": "Gearbox bearing",
    },
    11: {
        "event_id": 92,
        "event_label": "normal",
        "fault_type": "",
    },
    13: {
        "event_id": 14,
        "event_label": "normal",
        "fault_type": "",
    },
    21: {
        "event_id": 22,
        "event_label": "anomaly",
        "fault_type": "Hydraulic group",
    },
}

# ====================== SENSOR / FEATURE COLUMNS ======================
# Columns shown on charts (17 base sensor aggregates — before `label` column)
CHART_SENSOR_COLS = [
    'sensor_0_avg', 'sensor_5_avg_sin', 'sensor_5_avg_cos',
    'sensor_5_min_sin', 'sensor_5_min_cos', 'sensor_5_max_cos',
    'sensor_1_avg_sin', 'sensor_10_avg', 'sensor_14_avg',
    'sensor_18_avg', 'sensor_19_avg', 'sensor_33_avg', 'sensor_34_avg',
    'sensor_38_avg', 'sensor_40_avg', 'sensor_41_avg', 'sensor_42_avg_cos',
    'sensor_44', 'reactive_power_28_min', 'reactive_power_28_max',
    'wind_speed_3_min',
]

# Operating limits for warn/critical horizontal bands on the live chart.
# Keys match CHART_SENSOR_COLS. Values: warn_hi, crit_hi, warn_lo, crit_lo (unit matches sensor unit).
SENSOR_OPERATING_LIMITS: dict = {
    "sensor_0_avg":     {"warn_hi": 35.0},
    "sensor_10_avg":    {"warn_hi": 48.0, "crit_hi": 55.0},
    "sensor_41_avg":    {"warn_hi": 48.0, "crit_hi": 55.0},
    "sensor_14_avg":    {"warn_hi": 80.0, "crit_hi": 90.0},
    "sensor_18_avg":    {"warn_lo": 1500.0, "crit_lo": 1400.0},
    "wind_speed_3_min": {"warn_hi": 13.0,  "crit_hi": 25.0},
}


# Rolling-window engineered columns (84 cols: mean/std × window 3 & 6)
_ROLLING_COLS = [
    # --- window = 3 ---
    'sensor_0_avg_mean_3',    'sensor_5_avg_sin_mean_3', 'sensor_5_avg_cos_mean_3',
    'sensor_5_min_sin_mean_3', 'sensor_5_min_cos_mean_3', 'sensor_5_max_cos_mean_3',
    'sensor_1_avg_sin_mean_3', 'sensor_10_avg_mean_3', 'sensor_14_avg_mean_3',
    'sensor_18_avg_mean_3',   'sensor_19_avg_mean_3',   'sensor_33_avg_mean_3',
    'sensor_34_avg_mean_3',   'sensor_38_avg_mean_3',   'sensor_40_avg_mean_3',
    'sensor_41_avg_mean_3',   'sensor_42_avg_cos_mean_3', 'sensor_44_mean_3',
    'reactive_power_28_min_mean_3', 'reactive_power_28_max_mean_3',
    'wind_speed_3_min_mean_3',
    'sensor_0_avg_std_3',     'sensor_5_avg_sin_std_3',  'sensor_5_avg_cos_std_3',
    'sensor_5_min_sin_std_3', 'sensor_5_min_cos_std_3',  'sensor_5_max_cos_std_3',
    'sensor_1_avg_sin_std_3', 'sensor_10_avg_std_3',     'sensor_14_avg_std_3',
    'sensor_18_avg_std_3',   'sensor_19_avg_std_3',     'sensor_33_avg_std_3',
    'sensor_34_avg_std_3',   'sensor_38_avg_std_3',     'sensor_40_avg_std_3',
    'sensor_41_avg_std_3',   'sensor_42_avg_cos_std_3', 'sensor_44_std_3',
    'reactive_power_28_min_std_3', 'reactive_power_28_max_std_3',
    'wind_speed_3_min_std_3',
    # --- window = 6 ---
    'sensor_0_avg_mean_6',    'sensor_5_avg_sin_mean_6', 'sensor_5_avg_cos_mean_6',
    'sensor_5_min_sin_mean_6', 'sensor_5_min_cos_mean_6', 'sensor_5_max_cos_mean_6',
    'sensor_1_avg_sin_mean_6', 'sensor_10_avg_mean_6', 'sensor_14_avg_mean_6',
    'sensor_18_avg_mean_6',   'sensor_19_avg_mean_6',   'sensor_33_avg_mean_6',
    'sensor_34_avg_mean_6',   'sensor_38_avg_mean_6',   'sensor_40_avg_mean_6',
    'sensor_41_avg_mean_6',   'sensor_42_avg_cos_mean_6', 'sensor_44_mean_6',
    'reactive_power_28_min_mean_6', 'reactive_power_28_max_mean_6',
    'wind_speed_3_min_mean_6',
    'sensor_0_avg_std_6',     'sensor_5_avg_sin_std_6',  'sensor_5_avg_cos_std_6',
    'sensor_5_min_sin_std_6', 'sensor_5_min_cos_std_6',  'sensor_5_max_cos_std_6',
    'sensor_1_avg_sin_std_6', 'sensor_10_avg_std_6',     'sensor_14_avg_std_6',
    'sensor_18_avg_std_6',   'sensor_19_avg_std_6',     'sensor_33_avg_std_6',
    'sensor_34_avg_std_6',   'sensor_38_avg_std_6',     'sensor_40_avg_std_6',
    'sensor_41_avg_std_6',   'sensor_42_avg_cos_std_6', 'sensor_44_std_6',
    'reactive_power_28_min_std_6', 'reactive_power_28_max_std_6',
    'wind_speed_3_min_std_6',
]

# Full 105-feature vector fed into XGBoost / Random Forest
FEATURE_COLS = CHART_SENSOR_COLS + _ROLLING_COLS   # 21 + 84 = 105

# ====================== SENSOR LABELS & UNITS ======================
SENSOR_LABELS = {
    # ---- Aggregated base features (used in Real-time Monitor chart) ----
    'sensor_0_avg'          : 'Ambient Temp',
    'sensor_1_avg_sin'      : 'Wind Direction Avg Sin',
    'sensor_5_avg_cos'      : 'Pitch Angle Avg Cos',
    'sensor_5_min_cos'      : 'Pitch Angle Min Cos',
    'sensor_5_max_cos'      : 'Pitch Angle Max Cos',
    'sensor_5_avg_sin'      : 'Pitch Angle Avg Sin',
    'sensor_5_min_sin'      : 'Pitch Angle Min Sin',
    'sensor_43_avg'         : 'Nacelle Temp',
    'sensor_41_avg'         : 'Hydraulic Oil Temp',
    'sensor_14_avg'         : 'Gen Bearing Temp NDE',
    'sensor_18_avg'         : 'Generator RPM',
    'sensor_19_avg'         : 'Split Ring Chamber Temp',
    'sensor_33_avg'         : 'Voltage Phase 2',
    'sensor_34_avg'         : 'Voltage Phase 3',
    'sensor_52_max'         : 'Rotor RPM',
    'wind_speed_3_min'      : 'Wind Speed',
    'sensor_10_avg'         : 'Cooling Water Temp',
    'reactive_power_28_min' : 'Inductive Reactive Power Min',
    'reactive_power_28_max' : 'Inductive Reactive Power Max',
    'reactive_power_27_max' : 'Reactive Power cap',
    'sensor_47'             : 'Reactive Power Disconnected',
    'sensor_38_avg'         : 'HV Transformer Temp L1',
    'sensor_40_avg'         : 'HV Transformer Temp L3',
    'sensor_42_avg_cos'     : 'Nacelle Direction Avg Cos',
    'power_30_std'          : 'Grid Active Power',
    # ---- Raw sensor names (used on other pages) ----
    "sensor_0"          : "Ambient Temperature",
    "sensor_1"          : "Wind Absolute Direction",
    "sensor_2"          : "Wind Relative Direction",
    "wind_speed_3"      : "Wind Speed",
    "wind_speed_4"      : "Estimated Wind Speed",
    "sensor_5"          : "Pitch Angle",
    "sensor_6"          : "Hub Controller Temperature",
    "sensor_7"          : "Top Nacelle Controller Temperature",
    "sensor_8"          : "Choke Coils Temperature",
    "sensor_9"          : "VCP-Board Temperature",
    "sensor_10"         : "VCS Cooling Water Temperature",
    "sensor_11"         : "Gearbox Bearing Temp",
    "sensor_12"         : "Gearbox Oil Temperature",
    "sensor_13"         : "Generator Bearing 2 Temp",
    "sensor_14"         : "Generator Bearing 1 Temp",
    "sensor_15"         : "Generator Stator Windings Temp Phase 1",
    "sensor_16"         : "Generator Stator Windings Temp Phase 2",
    "sensor_17"         : "Generator Stator Windings Temp Phase 3",
    "sensor_18"         : "Generator RPM",
    "sensor_19"         : "Split Ring Chamber Temperature",
    "sensor_20"         : "Busbar Section Temperature",
    "sensor_21"         : "IGBT-Driver Temp",
    "sensor_22"         : "Phase Displacement",
    "sensor_23"         : "Current Phase 1",
    "sensor_24"         : "Current Phase 2",
    "sensor_25"         : "Current Phase 3",
    "sensor_26"         : "Grid Frequency",
    "reactive_power_27" : "Possible Grid Capacitive Reactive Power",
    "reactive_power_28" : "Possible Grid Inductive Reactive Power",
    "power_29"          : "Possible Grid Active Power",
    "power_30"          : "Grid Active Power",
    "sensor_31"         : "Grid Reactive Power",
    "sensor_32"         : "Voltage Phase 1",
    "sensor_33"         : "Voltage Phase 2",
    "sensor_34"         : "Voltage Phase 3",
    "sensor_35"         : "IGBT-Driver Temp (Rotor Side Inverter Phase 1)",
    "sensor_36"         : "IGBT-Driver Temp (Rotor Side Inverter Phase 2)",
    "sensor_37"         : "IGBT-Driver Temp (Rotor Side Inverter Phase 3)",
    "sensor_38"         : "HV Transformer Temperature Phase L1",
    "sensor_39"         : "HV Transformer Temperature Phase L2",
    "sensor_40"         : "HV Transformer Temperature Phase L3",
    "sensor_41"         : "Hydraulic Group Oil Temperature",
    "sensor_42"         : "Nacelle Direction",
    "sensor_43"         : "Nacelle Temperature",
    "sensor_44"         : "Active Power - Generator Disconnected",
    "sensor_45"         : "Active Power - Generator Connected in Delta",
    "sensor_46"         : "Active Power - Generator Connected in Star",
    "sensor_47"         : "Reactive Power - Generator Disconnected",
    "sensor_48"         : "Reactive Power - Generator Connected in Delta",
    "sensor_49"         : "Reactive Power - Generator Connected in Star",
    "sensor_50"         : "Total Active Power",
    "sensor_51"         : "Total Reactive Power",
    "sensor_52"         : "Rotor RPM",
    "sensor_53"         : "Nose Cone Temperature",
}

SENSOR_UNITS = {
    # Temperature sensors
    "sensor_0": "°C", "sensor_0_avg": "°C",
    "sensor_6": "°C", "sensor_7": "°C", "sensor_8": "°C", "sensor_9": "°C",
    "sensor_10": "°C", "sensor_10_avg": "°C",
    "sensor_11": "°C", "sensor_12": "°C", "sensor_13": "°C",
    "sensor_14": "°C", "sensor_14_avg": "°C",
    "sensor_15": "°C", "sensor_16": "°C", "sensor_17": "°C",
    "sensor_19": "°C", "sensor_20": "°C", "sensor_21": "°C",
    "sensor_35": "°C", "sensor_36": "°C", "sensor_37": "°C",
    "sensor_38": "°C", "sensor_38_avg": "°C",
    "sensor_39": "°C",
    "sensor_40": "°C", "sensor_40_avg": "°C",
    "sensor_41": "°C", "sensor_41_avg": "°C",
    "sensor_43": "°C", "sensor_43_avg": "°C",
    "sensor_53": "°C",
    # Wind speed
    "wind_speed_3": "m/s", "wind_speed_4": "m/s", "wind_speed_3_min": "m/s",
    # Angle / direction
    "sensor_5": "°", "sensor_1": "°", "sensor_2": "°", "sensor_42": "°",
    # RPM
    "sensor_18": "rpm", "sensor_18_avg": "rpm", "sensor_52": "rpm", "sensor_52_max": "rpm",
    # Power
    "power_29": "kW", "power_30": "kW", "power_30_std": "kW",
    "sensor_44": "Wh",
    "reactive_power_28_min": "kVAr", "reactive_power_28_max": "kVAr",
    # Current
    "sensor_23": "A", "sensor_24": "A", "sensor_25": "A",
    # Voltage
    "sensor_32": "V", "sensor_33": "V", "sensor_33_avg": "V", "sensor_34": "V", "sensor_34_avg": "V",
    # Frequency
    "sensor_26": "Hz",
}

SENSOR_GROUPS = {
    "Temperatures": [
        "sensor_0_avg",   # Ambient Temp
        "sensor_10_avg",  # Cooling Water Temp
        "sensor_14_avg",  # Gen Bearing Temp
        "sensor_19_avg",  # Split Ring Chamber Temp
        "sensor_38_avg",  # HV Transformer Temp L1
        "sensor_40_avg",  # HV Transformer Temp L3
        "sensor_41_avg",  # Hydraulic Oil Temp
    ],
    "Pitch & Yaw": [
        "sensor_5_avg_cos",  # Pitch angle cos (avg)
        "sensor_5_avg_sin",  # Pitch angle sin (avg)
        "sensor_5_min_cos",  # Pitch angle cos (min)
        "sensor_5_min_sin",  # Pitch angle sin (min)
        "sensor_5_max_cos",  # Pitch angle cos (max)
        "sensor_42_avg_cos", # Nacelle direction (cos)
        "sensor_1_avg_sin",  # Wind direction sin (avg)
    ],
    "Wind": [
        "wind_speed_3_min"  # Wind Speed (min)
    ],
    "Rotational": [
        "sensor_18_avg"     # Generator / Rotor RPM (avg)
    ],
    "Electrical": [
        "sensor_33_avg",      # Voltage Phase 2
        "sensor_34_avg",      # Voltage Phase 3
        "sensor_44",          # Active Power - Generator Disconnected
        "reactive_power_28_min",
        "reactive_power_28_max",
    ],
}

# ====================== VISUALIZATION ======================
STATUS_COLORS = {
    "Normal" : "#22c55e",
    "Warning": "#eab308",
    "Anomaly": "#ef4444",
    "Unknown": "#6b7280",
}

TRACE_COLORS = [
    '#38bdf8', '#f472b6', '#4ade80', '#fb923c', '#a78bfa',
    '#facc15', '#34d399', '#f87171', '#60a5fa', '#e879f9',
    '#fbbf24', '#86efac', '#93c5fd', '#fca5a5', '#c4b5fd',
]

CHART_CONFIG = {
    "height"  : 480,
    "template": "plotly_dark",
}

# ====================== SIMULATION SETTINGS ======================
SIMULATION_DELAY       = 2.0    # seconds between auto-rerun steps
SIMULATION_BATCH_SIZE  = 10     # rows per step per turbine
SIMULATION_MAX_HISTORY = 150    # max data points kept per turbine in live view

# ====================== DASHBOARD META ======================
APP_TITLE          = "SCADA Anomaly Guard - Wind Turbine"
APP_DESCRIPTION    = "Real-time Anomaly Detection for Wind Turbine SCADA"
SIDEBAR_WIDTH      = 300
DEFAULT_TABLE_ROWS = 15

# ====================== HELPER FUNCTIONS ======================
def get_sensor_label(sensor_key: str) -> str:
    """Return human-readable label; fallback to key itself."""
    return SENSOR_LABELS.get(sensor_key, sensor_key)


def get_sensor_unit(sensor_key: str) -> str:
    """Return unit string; fallback to empty string."""
    return SENSOR_UNITS.get(sensor_key, "")


def get_model_path(model_name: str) -> str:
    """Return resolved filesystem path for a model file."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not in AVAILABLE_MODELS.")
    path = MODELS_DIR / AVAILABLE_MODELS[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return str(path)


def get_data_path(use_sample: bool = False) -> str:
    """Return path to test or sample data CSV."""
    if use_sample and SAMPLE_DATA_PATH.exists():
        return str(SAMPLE_DATA_PATH)
    return str(TEST_DATA_PATH)


def init_directories():
    """Create required project directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, ASSETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


init_directories()
