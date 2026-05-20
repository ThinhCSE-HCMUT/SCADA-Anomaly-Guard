import pandas as pd
import streamlit as st
from src.config import (
    RAW_PREDICTION_SPLIT,
    RAW_WIND_FARM_A_DATASETS_DIR,
    RAW_WIND_FARM_A_EVENT_INFO,
    REALTIME_DATA_PATH,
    TARGET_TURBINES,
)
from src.online_preprocessor import (
    load_online_feature_contract,
    prepare_raw_prediction_frame,
    required_raw_columns_for_features,
)


def _event_sort_key(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


@st.cache_data(show_spinner="Loading Wind Farm A event list, please wait...")
def load_event_info() -> pd.DataFrame:
    try:
        if not RAW_WIND_FARM_A_EVENT_INFO.exists():
            st.error(f"Wind Farm A event metadata is missing: {RAW_WIND_FARM_A_EVENT_INFO}")
            return pd.DataFrame()

        df = pd.read_csv(RAW_WIND_FARM_A_EVENT_INFO, sep=";")
        df["asset"] = pd.to_numeric(df["asset"], errors="coerce").astype("Int64")
        df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["asset", "event_id"]).copy()
        df["asset"] = df["asset"].astype(int)
        df["event_id"] = df["event_id"].astype(int)

        for col in ("event_start", "event_end"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        event_label = df["event_label"] if "event_label" in df.columns else pd.Series("", index=df.index)
        event_description = (
            df["event_description"]
            if "event_description" in df.columns
            else pd.Series("", index=df.index)
        )
        df["event_label"] = event_label.fillna("").astype(str).str.strip()
        df["event_description"] = (
            event_description.fillna("").astype(str).str.strip()
        )
        df["event_file"] = df["event_id"].astype(str) + ".csv"
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"Loading Wind Farm A event metadata failed: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading legacy detection data, please wait...")
def load_legacy_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(str(REALTIME_DATA_PATH))
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        df = df[df['asset_id'].isin(TARGET_TURBINES)].copy()
        return df.sort_values(['asset_id', 'time_stamp']).reset_index(drop=True)
    except Exception as e:
        st.error(f"Loading CSV is failed: {e}")
        return pd.DataFrame()


def load_data() -> pd.DataFrame:
    """Backward-compatible alias for the legacy flat-feature stream."""
    return load_legacy_data()


@st.cache_data(show_spinner="Loading raw Wind Farm A prediction rows, please wait...")
def load_online_prediction_data(
    event_id: int | None = None,
    split: str = RAW_PREDICTION_SPLIT,
) -> pd.DataFrame:
    try:
        if not RAW_WIND_FARM_A_DATASETS_DIR.exists():
            st.error(f"Raw Wind Farm A folder is missing: {RAW_WIND_FARM_A_DATASETS_DIR}")
            return pd.DataFrame()

        feature_cols = load_online_feature_contract()
        required_cols = required_raw_columns_for_features(feature_cols)
        parts = []
        if event_id is not None:
            csv_paths = [RAW_WIND_FARM_A_DATASETS_DIR / f"{int(event_id)}.csv"]
        else:
            csv_paths = sorted(RAW_WIND_FARM_A_DATASETS_DIR.glob("*.csv"), key=_event_sort_key)

        for csv_path in csv_paths:
            if not csv_path.exists():
                st.error(f"Selected Wind Farm A event file is missing: {csv_path}")
                continue

            header = pd.read_csv(csv_path, sep=";", nrows=0).columns.tolist()
            usecols = [col for col in header if col in required_cols]
            if "time_stamp" not in usecols or "asset_id" not in usecols:
                continue

            df = pd.read_csv(csv_path, sep=";", usecols=usecols)
            if "sequence_id" not in df.columns:
                try:
                    df["sequence_id"] = int(csv_path.stem)
                except ValueError:
                    df["sequence_id"] = csv_path.stem
            parts.append(df)

        if not parts:
            return pd.DataFrame()

        raw_df = pd.concat(parts, ignore_index=True)
        if "train_test" in raw_df.columns and split:
            raw_df["train_test"] = raw_df["train_test"].astype(str).str.strip().str.lower()
            raw_df = raw_df[raw_df["train_test"].eq(split.lower())].copy()

        raw_df["asset_id"] = pd.to_numeric(raw_df["asset_id"], errors="coerce")
        raw_df = raw_df[raw_df["asset_id"].isin(TARGET_TURBINES)].copy()

        prepared = prepare_raw_prediction_frame(raw_df, feature_cols)
        if event_id is not None and not prepared.empty:
            event_info = load_event_info()
            event_row = event_info[event_info["event_id"].eq(int(event_id))]
            if not event_row.empty:
                meta = event_row.iloc[0]
                prepared["event_label"] = meta.get("event_label", "")
                prepared["event_description"] = meta.get("event_description", "")
                prepared["event_start"] = meta.get("event_start", pd.NaT)
                prepared["event_end"] = meta.get("event_end", pd.NaT)

        sort_cols = [col for col in ("asset_id", "sequence_id", "time_stamp", "id") if col in prepared.columns]
        return prepared.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    except Exception as e:
        st.error(f"Loading raw Wind Farm A data failed: {e}")
        return pd.DataFrame()


@st.cache_data
def split_by_turbine(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        tid: df[df['asset_id'] == tid].reset_index(drop=True)
        for tid in TARGET_TURBINES
    }
