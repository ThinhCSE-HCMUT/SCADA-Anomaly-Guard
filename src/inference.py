import numpy as np
import pandas as pd
from src.config import FEATURE_COLS

def predict_batch(model, batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if model is None:
        labels = batch['label'].values.astype(int)
        return labels, labels.astype(float)

    X = batch[FEATURE_COLS].fillna(0).values
    pred_labels = model.predict(X).astype(int)
    
    pred_probas = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, 'predict_proba')
        else pred_labels.astype(float)
    )
    return pred_labels, pred_probas