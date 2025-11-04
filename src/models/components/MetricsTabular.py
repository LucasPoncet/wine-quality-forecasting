from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def _binarise(preds: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (preds >= thr).astype(int)


def accuracy(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    return (_binarise(y_pred_prob) == y_true).mean() * 100.0


def f1(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    return float(f1_score(y_true, _binarise(y_pred_prob)))


def roc_auc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_pred_prob))
    except ValueError:
        return float("nan")
