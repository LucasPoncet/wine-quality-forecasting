from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_metrics(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_proba: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_proba))
    return acc, f1, auc


def print_confusion(y_true: NDArray[np.int_], y_pred: NDArray[np.int_]) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion matrix:\n[[TN {tn}  FP {fp}]\n [FN {fn}  TP {tp}]]")
