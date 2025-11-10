import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.builders.classical_estimators import RANDOM_STATE, _make_estimator, _param_grid


@pytest.mark.parametrize(
    "model_key, expected_class",
    [
        ("lr", LogisticRegression),
        ("rf", RandomForestClassifier),
        ("xgb", XGBClassifier),
        ("hgb", HistGradientBoostingClassifier),
        ("unknown", HistGradientBoostingClassifier),
    ],
)
def test_make_estimator_returns_correct_type(model_key, expected_class):
    model = _make_estimator(model_key)
    assert isinstance(model, expected_class)


def test_lr_estimator_parameters():
    model = _make_estimator("lr")
    assert model.max_iter == 1000  # type: ignore[attr-defined]
    assert model.class_weight == "balanced"  # type: ignore[attr-defined]
    assert model.random_state == RANDOM_STATE  # type: ignore[attr-defined]
    assert model.solver in ["lbfgs", "liblinear"]  # type: ignore[attr-defined]


def test_rf_estimator_parameters():
    model = _make_estimator("rf")
    assert model.n_estimators == 200  # type: ignore[attr-defined]
    assert model.class_weight == "balanced"  # type: ignore[attr-defined]
    assert model.random_state == RANDOM_STATE  # type: ignore[attr-defined]
    assert model.n_jobs == -1  # type: ignore[attr-defined]


def test_xgb_estimator_parameters():
    model = _make_estimator("xgb")
    assert isinstance(model, XGBClassifier)
    assert model.n_estimators == 200  # type: ignore[attr-defined]
    assert model.learning_rate == 0.1  # type: ignore[attr-defined]
    assert model.subsample == 0.8  # type: ignore[attr-defined]
    assert model.colsample_bytree == 0.8  # type: ignore[attr-defined]
    assert model.random_state == RANDOM_STATE  # type: ignore[attr-defined]


def test_hgb_estimator_default_behavior():
    model = _make_estimator("something_else")
    assert isinstance(model, HistGradientBoostingClassifier)
    assert model.learning_rate == 0.1  # type: ignore[attr-defined]
    assert model.class_weight == "balanced"  # type: ignore[attr-defined]


@pytest.mark.parametrize("model_key", ["lr", "rf", "xgb", "hgb", "something_else"])
def test_param_grid_keys_are_prefixed(model_key):
    grid = _param_grid(model_key)
    assert all(key.startswith("model__") for key in grid)


def test_lr_param_grid_contents():
    grid = _param_grid("lr")
    assert "model__C" in grid
    assert any(v > 0 for v in grid["model__C"])
    assert grid["model__solver"] == ["lbfgs", "liblinear"]
    assert "model__penalty" in grid


def test_rf_param_grid_valid_ranges():
    grid = _param_grid("rf")
    assert all(isinstance(v, list) for v in grid.values())
    assert max(grid["model__n_estimators"]) >= 600
    assert "model__max_depth" in grid


def test_xgb_param_grid_valid_values():
    grid = _param_grid("xgb")
    assert "model__n_estimators" in grid
    assert all(0 < lr < 0.1 for lr in grid["model__learning_rate"])
    assert any(depth >= 3 for depth in grid["model__max_depth"])
    assert "model__subsample" in grid
    assert "model__reg_lambda" in grid


def test_default_param_grid_hgb():
    grid = _param_grid("hgb")
    assert "model__learning_rate" in grid
    assert "model__l2_regularization" in grid
    assert all(isinstance(x, float) for x in grid["model__learning_rate"])


def test_invalid_model_key_still_returns_dict():
    grid = _param_grid("invalid_key")
    assert isinstance(grid, dict)
    assert "model__learning_rate" in grid
