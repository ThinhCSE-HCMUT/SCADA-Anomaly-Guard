import os
import joblib
import streamlit as st
from src.config import MODEL_PATHS

@st.cache_resource(show_spinner="Loading AI model, please wait...")
def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name, "")
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Cannot load the model '{model_name}': {e}")
        return None