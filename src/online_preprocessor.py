"""Online preprocessing for raw Wind Farm A prediction rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CHART_SENSOR_COLS,
    LOCAL_PREDICTION_FEATURE_FILE,
    PREDICTION_CLASSIFIER_EXPORT_DIR,
)


BASE_RAW_COLUMNS = {
    "time_stamp",
    "asset_id",
    "id",
    "train_test",
    "status_type_id",
    "sequence_id",
}


def _read_feature_list(path: Path) -> list[str]:
    feature_df = pd.read_csv(path)
    if feature_df.empty:
        return []

    if "feature_cols" in feature_df.columns:
        column_name = "feature_cols"
    elif "feature" in feature_df.columns:
        column_name = "feature"
    elif "final_feature" in feature_df.columns:
        column_name = "final_feature"
    else:
        column_name = feature_df.columns[0]

    return (
        feature_df[column_name]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .tolist()
    )


@st.cache_data(show_spinner=False)
def load_online_feature_contract() -> list[str]:
    metadata_path = PREDICTION_CLASSIFIER_EXPORT_DIR / "metadata.json"
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            feature_cols = metadata.get("feature_cols") or []
            if feature_cols:
                return [str(col) for col in feature_cols]
        except Exception:
            pass

    if LOCAL_PREDICTION_FEATURE_FILE.exists():
        feature_cols = _read_feature_list(LOCAL_PREDICTION_FEATURE_FILE)
        if feature_cols:
            return feature_cols

    return list(CHART_SENSOR_COLS)


def _derived_source(feature_name: str) -> tuple[str, str | None]:
    for suffix in ("_sin", "_cos"):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)], suffix
    return feature_name, None


def required_raw_columns_for_features(feature_cols: list[str]) -> set[str]:
    required = set(BASE_RAW_COLUMNS)
    for feature_name in feature_cols:
        source_col, _ = _derived_source(feature_name)
        required.add(source_col)
    return required


def _add_angle_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for feature_name in feature_cols:
        if feature_name in out.columns:
            continue

        source_col, suffix = _derived_source(feature_name)
        if suffix is None or source_col not in out.columns:
            continue

        radians = pd.to_numeric(out[source_col], errors="coerce") * (np.pi / 180.0)
        if suffix == "_sin":
            out[feature_name] = np.sin(radians)
        else:
            out[feature_name] = np.cos(radians)
    return out


def prepare_raw_prediction_frame(
    raw_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert raw CARE rows into the configured online prediction frame."""
    feature_cols = list(feature_cols or load_online_feature_contract())
    if raw_df.empty:
        return pd.DataFrame(columns=["time_stamp", "asset_id", *feature_cols])

    df = raw_df.copy()
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df["asset_id"] = pd.to_numeric(df["asset_id"], errors="coerce")
    df = df.dropna(subset=["time_stamp", "asset_id"]).copy()
    df["asset_id"] = df["asset_id"].astype(int)

    if "sequence_id" in df.columns:
        df["sequence_id"] = pd.to_numeric(df["sequence_id"], errors="coerce").astype("Int64")
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
    if "status_type_id" in df.columns:
        df["status_type_id"] = pd.to_numeric(df["status_type_id"], errors="coerce").fillna(-1).astype(int)
    if "train_test" in df.columns:
        df["train_test"] = df["train_test"].astype(str).str.strip().str.lower()

    df = _add_angle_features(df, feature_cols)

    for feature_name in feature_cols:
        if feature_name not in df.columns:
            df[feature_name] = np.nan
        df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce")

    group_cols = [col for col in ("asset_id", "sequence_id") if col in df.columns]
    sort_cols = [col for col in ("asset_id", "sequence_id", "time_stamp", "id") if col in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort")

    if group_cols:
        df[feature_cols] = df.groupby(group_cols, group_keys=False)[feature_cols].ffill()
    else:
        df[feature_cols] = df[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].fillna(0.0).astype(np.float32)

    meta_cols = [
        col
        for col in ("time_stamp", "asset_id", "sequence_id", "id", "train_test", "status_type_id")
        if col in df.columns
    ]
    return df[meta_cols + feature_cols].reset_index(drop=True)
