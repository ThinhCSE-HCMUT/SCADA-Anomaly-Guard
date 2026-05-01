import pandas as pd
import streamlit as st
from src.config import REALTIME_DATA_PATH, TARGET_TURBINES

@st.cache_data(show_spinner="Loading data, please wait...")
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(str(REALTIME_DATA_PATH))
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        df = df[df['asset_id'].isin(TARGET_TURBINES)].copy()
        return df.sort_values(['asset_id', 'time_stamp']).reset_index(drop=True)
    except Exception as e:
        st.error(f"Loading CSV is failed: {e}")
        return pd.DataFrame()

@st.cache_data
def split_by_turbine(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        tid: df[df['asset_id'] == tid].reset_index(drop=True)
        for tid in TARGET_TURBINES
    }