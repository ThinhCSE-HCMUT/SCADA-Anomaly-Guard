import numpy as np
import pandas as pd
from src.config import FEATURE_COLS


def _fallback_from_labels(batch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if 'label' in batch.columns:
        labels = batch['label'].fillna(0).values.astype(int)
    else:
        labels = np.zeros(len(batch), dtype=int)
    return labels, labels.astype(float)


def _expects_sequence_input(model) -> bool:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]
    return isinstance(input_shape, tuple) and len(input_shape) == 3


def predict_batch(model, batch: pd.DataFrame, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    if model is None:
        return _fallback_from_labels(batch)

    # The dashboard's current-detection path is flat-row based. Sequence Keras
    # models are handled by src.prediction, which builds [batch, 144, 21] inputs.
    if _expects_sequence_input(model):
        return _fallback_from_labels(batch)

    X = batch[FEATURE_COLS].fillna(0).values
    try:
        if hasattr(model, 'predict_proba'):
            pred_probas = model.predict_proba(X)[:, 1]
            pred_labels = (pred_probas >= threshold).astype(int)
        else:
            raw_pred = np.asarray(model.predict(X)).reshape(-1)
            pred_probas = raw_pred.astype(float)
            pred_labels = (pred_probas >= threshold).astype(int)
    except Exception:
        return _fallback_from_labels(batch)
    return pred_labels, pred_probas
