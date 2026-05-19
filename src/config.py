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

# Data files
# df_simulation.csv: 88 cols (17 base + 68 rolling + label + meta)
REALTIME_DATA_PATH = DATA_DIR / "df_simulation.csv"
TEST_DATA_PATH     = DATA_DIR / "test_data.csv"
SAMPLE_DATA_PATH   = DATA_DIR / "sample_data.csv"

# ====================== MODEL CONFIGURATION ======================
AVAILABLE_MODELS = {
    "XGBoost"                      : "Baseline/XGBoost/xgb_model.pkl",
    "Random Forest"                : "Baseline/RandomForest/rf_model.pkl",
    "LSTM"                         : "DeepLearning/LSTM/lstm_model.keras",
    "GRU"                          : "DeepLearning/GRU/gru_model.keras",
    "CNN - LSTM"                   : "DeepLearning/CNN_LSTM/cnn_lstm_hybrid_model.keras",
    "CNN - GRU"                    : "DeepLearning/CNN_GRU/cnn_gru_hybrid_model.keras"
}

MODEL_THRESHOLDS = {
    "LSTM": {
        "Current": 0.418,
        "12h": 0.384,
        "24h": 0.423,
        "36h": 0.459,
        "48h": 0.354,
        "72h": 0.403
    },
    "GRU": {
        "Current": 0.495,
        "12h": 0.382,
        "24h": 0.166,
        "36h": 0.321,
        "48h": 0.528,
        "72h": 0.391
    },
    "CNN - LSTM": {
        "Current": 0.245,
        "12h": 0.27,
        "24h": 0.445,
        "36h": 0.668,
        "48h": 0.192,
        "72h": 0.402
    },
    "CNN - GRU": {
        "Current": 0.333,
        "12h": 0.357,
        "24h": 0.364,
        "36h": 0.545,
        "48h": 0.22,
        "72h": 0.365
    }
}

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
        "Current": "DeepLearning/LSTM/lstm_model.keras",
        "12h": "DeepLearning/LSTM/lstm_model_12h.keras",
        "24h": "DeepLearning/LSTM/lstm_model_24h.keras",
        "36h": "DeepLearning/LSTM/lstm_model_36h.keras",
        "48h": "DeepLearning/LSTM/lstm_model_48h.keras",
        "72h": "DeepLearning/LSTM/lstm_model_72h.keras",
    },
    "GRU": {
        "Current": "DeepLearning/GRU/gru_model.keras",
        "12h": "DeepLearning/GRU/gru_model_12h.keras",
        "24h": "DeepLearning/GRU/gru_model_24h.keras",
        "36h": "DeepLearning/GRU/gru_model_36h.keras",
        "48h": "DeepLearning/GRU/gru_model_48h.keras",
        "72h": "DeepLearning/GRU/gru_model_72h.keras",
    },
    "CNN - LSTM": {
        "Current": "DeepLearning/CNN_LSTM/cnn_lstm_hybrid_model.keras",
        "12h": "DeepLearning/CNN_LSTM/cnn_lstm_hybrid_model_12h.keras",
        "24h": "DeepLearning/CNN_LSTM/cnn_lstm_hybrid_model_24h.keras",
        "36h": "DeepLearning/CNN_LSTM/cnn_lstm_hybrid_model_36h.keras",
        "48h": "DeepLearning/CNN_LSTM/cnn_lstm_hybrid_model_48h.keras",
        "72h": "DeepLearning/CNN_LSTM/cnn_lstm_hybrid_model_72h.keras",
    },
    "CNN - GRU": {
        "Current": "DeepLearning/CNN_GRU/cnn_gru_hybrid_model.keras",
        "12h": "DeepLearning/CNN_GRU/cnn_gru_hybrid_model_12h.keras",
        "24h": "DeepLearning/CNN_GRU/cnn_gru_hybrid_model_24h.keras",
        "36h": "DeepLearning/CNN_GRU/cnn_gru_hybrid_model_36h.keras",
        "48h": "DeepLearning/CNN_GRU/cnn_gru_hybrid_model_48h.keras",
        "72h": "DeepLearning/CNN_GRU/cnn_gru_hybrid_model_72h.keras",
    },
}

DL_FORECAST_MODEL_PATHS = {
    model_name: {
        horizon: str(MODELS_DIR / filename)
        for horizon, filename in horizon_map.items()
    }
    for model_name, horizon_map in DL_FORECAST_OPTIONS.items()
}

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
    'sensor_5_avg_cos'      : 'Pitch angle cos',
    'sensor_5_min_cos'      : 'Pitch angle cos',
    'sensor_5_max_cos'      : 'Pitch angle cos',
    'sensor_5_avg_sin'      : 'Pitch angle sin',
    'sensor_5_min_sin'      : 'Pitch angle sin',
    'sensor_43_avg'         : 'Nacelle Temp',
    'sensor_41_avg'         : 'Hydraulic Oil Temp',
    'sensor_14_avg'         : 'Gen Bearing Temp NDE',
    'sensor_52_max'         : 'Rotor RPM',
    'wind_speed_3_min'      : 'Wind Speed',
    'sensor_10_avg'         : 'Cooling Water Temp',
    'reactive_power_27_max' : 'Reactive Power cap',
    'sensor_47'             : 'Reactive Power Disconnected',
    'sensor_38_avg'         : 'HV Transformer Temp L1',
    'sensor_40_avg'         : 'HV Transformer Temp L3',
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
    "sensor_18": "rpm", "sensor_52": "rpm", "sensor_52_max": "rpm",
    # Power
    "power_29": "kW", "power_30": "kW", "power_30_std": "kW",
    # Current
    "sensor_23": "A", "sensor_24": "A", "sensor_25": "A",
    # Voltage
    "sensor_32": "V", "sensor_33": "V", "sensor_34": "V",
    # Frequency
    "sensor_26": "Hz",
}

SENSOR_GROUPS = {
    "Temperatures": [
        "sensor_0_avg",   # Ambient Temp
        "sensor_10_avg",  # Cooling Water Temp
        "sensor_14_avg",  # Gen Bearing Temp
        "sensor_38_avg",  # HV Transformer Temp L1
        "sensor_40_avg",  # HV Transformer Temp L3
        "sensor_41_avg",  # Hydraulic Oil Temp
        "sensor_43_avg"   # Nacelle Temp
    ],
    "Wind": [
        "wind_speed_3_min" # Wind Speed (min)
    ],
    "Rotational Speed": [
        "sensor_52_max"    # Rotor RPM (max)
    ],
    "Power": [
        "power_30_std"     # Grid Active Power (std)
    ],
    "Electrical": [
        "reactive_power_27_max", # Reactive Power cap (max)
        "sensor_47"              # Reactive Power Disconnected
    ],
    "Pitch & Yaw": [
        "sensor_5_avg_cos", # Pitch angle cos (avg)
        "sensor_5_max_cos", # Pitch angle cos (max)
        "sensor_5_min_cos", # Pitch angle cos (min)
        "sensor_5_avg_sin", # Pitch angle sin (avg)
        "sensor_5_min_sin"  # Pitch angle sin (min)
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
SIMULATION_DELAY       = 3.0    # seconds between auto-rerun steps
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