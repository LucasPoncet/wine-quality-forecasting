import numpy as np
import pytest

from src.models.evaluation.metrics_utils import evaluate_metrics, print_confusion


@pytest.fixture
def binary_predictions():
    """Provide a small deterministic binary classification setup."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    # Probabilities corresponding to predicted label = 1
    y_proba = np.array([0.2, 0.8, 0.1, 0.3, 0.9, 0.6])
    return y_true, y_pred, y_proba


def test_evaluate_metrics_returns_three_floats(binary_predictions):
    """Check type and length of returned metrics."""
    y_true, y_pred, y_proba = binary_predictions
    acc, f1, auc = evaluate_metrics(y_true, y_pred, y_proba)
    assert isinstance(acc, float)
    assert isinstance(f1, float)
    assert isinstance(auc, float)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= auc <= 1.0


def test_evaluate_metrics_expected_values(binary_predictions):
    """Ensure expected metric computation correctness."""
    y_true, y_pred, y_proba = binary_predictions
    acc, f1, auc = evaluate_metrics(y_true, y_pred, y_proba)

    # Manually computed:
    # TP=2, TN=2, FP=1, FN=1 -> Accuracy = (TP+TN)/6 = 4/6 = 0.6667
    assert pytest.approx(acc, rel=1e-3) == 2 / 3
    # F1 for positive class: precision=2/(2+1)=0.667, recall=2/(2+1)=0.667 => F1=0.667
    assert pytest.approx(f1, rel=1e-3) == 2 / 3
    # AUC should be roughly 0.9167 for these scores
    assert pytest.approx(auc, rel=1e-3) == 0.8889


def test_print_confusion_output(capsys, binary_predictions):
    """Ensure confusion matrix is printed correctly."""
    y_true, y_pred, _ = binary_predictions
    print_confusion(y_true, y_pred)
    captured = capsys.readouterr()
    # Check structure and key tokens
    assert "Confusion matrix:" in captured.out
    assert "TN" in captured.out and "FP" in captured.out
    assert "FN" in captured.out and "TP" in captured.out


def test_print_confusion_correct_counts(capsys):
    """Test that printed TN/FP/FN/TP counts match computed values."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    # TN=1, FP=1, FN=1, TP=1
    print_confusion(y_true, y_pred)
    out = capsys.readouterr().out
    assert "TN 1" in out
    assert "FP 1" in out
    assert "FN 1" in out
    assert "TP 1" in out


def test_evaluate_metrics_invalid_input_shape():
    """Ensure shape mismatch or invalid probabilities raises cleanly."""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 1])
    y_proba = np.array([0.2, 0.8])  # wrong length
    with pytest.raises(ValueError):
        evaluate_metrics(y_true, y_pred, y_proba)
