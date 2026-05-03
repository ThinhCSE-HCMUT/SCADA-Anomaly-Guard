import os
import joblib
import streamlit as st
from src.config import MODEL_PATHS
from tensorflow.keras.models import load_model as keras_load_model
import onnxruntime as ort

@st.cache_resource(show_spinner="Loading AI model, please wait...")
def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name, "")
    if not path or not os.path.exists(path):
        print(f"Model file for '{model_name}' not found at path: {path} muahahaa")
        return None
        
    try:
        # Kiểm tra đuôi file để quyết định công cụ load
        if path.endswith(".keras") or path.endswith(".h5"):
            # Dùng Keras để load Deep Learning & Hybrid models
            return keras_load_model(path)
        elif path.endswith(".pkl"):
            # Dùng Joblib để load Machine Learning models
            return joblib.load(path)
        
        elif path.endswith(".onnx"):
            # Dùng ONNX Runtime để load Deep Learning & Hybrid models
            # Khởi tạo InferenceSession để chạy dự đoán
            return ort.InferenceSession(path)
        else:
            st.warning(f"Unsupported model format for file: {path}")
            return None
            
    except Exception as e:
        st.warning(f"Cannot load the model '{model_name}': {e}")
        return None