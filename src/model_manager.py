import os
import joblib
import streamlit as st
from src.i18n import t
from src.config import MODEL_PATHS

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name, "")
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(t("common.model_load_failed", model=model_name, error=e))
        return None
