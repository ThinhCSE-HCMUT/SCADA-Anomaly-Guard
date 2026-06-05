import os
import joblib
import streamlit as st
import zipfile
import json
import hashlib
from pathlib import Path
from src.config import MODEL_PATHS, ROOT_DIR
from tensorflow.keras.models import load_model as keras_load_model

@st.cache_resource(show_spinner="Loading AI model, please wait...")
def load_model(model_name: str, use_for_real_time_monitor: bool = False):
    # 🌟 Áp dụng tham số của bạn để phân luồng logic
    if use_for_real_time_monitor:
        # Nếu là Real-time Monitor, model_name chính là đường dẫn tuyệt đối trực tiếp từ UI
        path = model_name
    else:
        # Nếu là False (mặc định cho trang Testing cũ), giữ nguyên logic tra cứu từ dict ban đầu
        path = MODEL_PATHS.get(model_name, "")
        
    if not path or not os.path.exists(path):
        print(f"Model file for '{model_name}' not found at path: {path} muahahaa")
        return None
        
    try:
        # Kiểm tra đuôi file để quyết định công cụ load
        if path.endswith(".keras") or path.endswith(".h5"):
            # Dùng Keras để load Deep Learning & Hybrid models
            model_path = Path(path)
            
            def _load_keras_path(p: Path):
                try:
                    return keras_load_model(str(p), compile=False, safe_mode=False)
                except TypeError:
                    return keras_load_model(str(p), compile=False)

            try:
                return _load_keras_path(model_path)
            except Exception as exc:
                # attempt to build a compatibility archive by stripping incompatible config
                try:
                    # only zip-backed saved models need this
                    if zipfile.is_zipfile(model_path):
                        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:12]
                        compat_dir = Path(ROOT_DIR) / ".streamlit" / "keras_compat"
                        compat_dir.mkdir(parents=True, exist_ok=True)
                        compat_path = compat_dir / f"{model_path.stem}-{digest}.keras"
                        
                        if not compat_path.exists():
                            with zipfile.ZipFile(model_path, "r") as src, zipfile.ZipFile(compat_path, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                                for info in src.infolist():
                                    if info.filename == "config.json":
                                        cfg = json.loads(src.read(info.filename).decode("utf-8"))
                                        # remove known incompatible entries
                                        def _strip_cfg(v):
                                            if isinstance(v, dict):
                                                return {k: _strip_cfg(val) for k, val in v.items() if k != "quantization_config"}
                                            if isinstance(v, list):
                                                return [_strip_cfg(i) for i in v]
                                            return v

                                        cfg = _strip_cfg(cfg)
                                        dst.writestr(info, json.dumps(cfg, ensure_ascii=False, separators=(",", ":")))
                                    else:
                                        dst.writestr(info, src.read(info.filename))
                        return _load_keras_path(compat_path)
                except Exception:
                    pass
                # final fallback: re-raise original exception to be handled by caller
                raise
                
        elif path.endswith(".pkl"):
            # Dùng Joblib để load Machine Learning models
            return joblib.load(path)
        
        else:
            st.warning(f"Unsupported model format for file: {path}")
            return None
            
    except Exception as e:
        st.warning(f"Cannot load the model '{model_name}': {e}")
        return None
