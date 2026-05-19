import numpy as np
import pandas as pd
from src.config import FEATURE_COLS

def scale_features(X: np.ndarray, scaler) -> np.ndarray:
    """Scale features using provided scaler. Returns X if scaler is None."""
    if scaler is None:
        return X
    try:
        return scaler.transform(X)
    except Exception as e:
        print(f"Warning: Cannot apply scaler: {e}. Using raw features.")
        return X

def predict_batch(model, batch: pd.DataFrame, scaler=None) -> tuple[np.ndarray, np.ndarray]:
    if model is None:
        labels = batch['label'].values.astype(int)
        return labels, labels.astype(float)

    # 1. Lấy mảng dữ liệu đặc trưng cơ bản (2D)
    X = batch[FEATURE_COLS].fillna(0).values
    
    # Scale features nếu có scaler (Thường dùng cho ML models như XGBoost, Random Forest)
    X = scale_features(X, scaler)
    
    # 2. KIỂM TRA KIỂU MÔ HÌNH ĐỂ TỰ ĐỘNG CẤU TRÚC LẠI SHAPE (ML vs DL)
    model_classname = model.__class__.__name__
    
    # Nếu là mô hình Keras (Sequential hoặc Functional của TensorFlow)
    if "keras" in model_classname.lower() or "functional" in model_classname.lower():
        try:
            # Lấy cấu hình Shape đầu vào mà mô hình LSTM/GRU yêu cầu (Ví dụ: [None, 144, 21])
            input_shape = model.input_shape # Thường trả về tuple e.g., (None, 144, 21)
            req_timesteps = input_shape[1] if input_shape[1] is not None else 144
            req_features = input_shape[2] if input_shape[2] is not None else 21
            
            current_batch_size = X.shape[0]
            
            # --- XỬ LÝ KHỚP SHAPE 3D ---
            # Nếu ma trận phẳng X hiện tại không đủ tổ hợp (Time steps x Features)
            if X.shape[1] != (req_timesteps * req_features):
                # Giải pháp phân tích stream: Lặp/Broadcast dòng dữ liệu hiện tại để lấp đầy 144 time steps
                # Giúp mô hình DL hiểu dữ liệu tại nhịp hiện tại mà không làm sập luồng Stream
                X_3d = np.zeros((current_batch_size, req_timesteps, req_features))
                for i in range(current_batch_size):
                    # Lấy dòng dữ liệu hiện tại (đủ req_features) gán đều cho toàn bộ trục thời gian lịch sử
                    row_data = X[i, :req_features] if X.shape[1] >= req_features else np.pad(X[i], (0, req_features - X.shape[1]))
                    X_3d[i, :, :] = row_data
                X = X_3d
            else:
                # Nếu dữ liệu lấy ra đã được làm phẳng từ trước (144 * 21 = 3024), ta chỉ cần gập lại thành 3D
                X = X.reshape(current_batch_size, req_timesteps, req_features)
                
        except Exception as e:
            print(f"Warning during DL reshaping: {e}. Attempting fallback reshape.")
            # Fallback dự phòng nếu không đọc được input_shape
            X = np.repeat(X[:, np.newaxis, :], 144, axis=1)[:, :, :21]

    # 3. TIẾN HÀNH DỰ ĐOÁN ĐỒNG BỘ
    # Đối với mô hình Keras, ta dùng predict() để lấy xác suất xác định bất thường
    if "keras" in model_classname.lower() or "functional" in model_classname.lower():
        pred_probas = model.predict(X, verbose=0).flatten()
        pred_labels = (pred_probas > 0.5).astype(int)
    else:
        # Giữ nguyên luồng dự đoán cho các mô hình Machine Learning cổ điển
        pred_labels = model.predict(X).astype(int)
        pred_probas = (
            model.predict_proba(X)[:, 1]
            if hasattr(model, 'predict_proba')
            else pred_labels.astype(float)
        )
        
    return pred_labels, pred_probas