# pyright: reportAttributeAccessIssue=false

import builtins
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from scripts.run_baselines import run_baselines


@pytest.fixture
def mock_data(tmp_path: Path):
    """Create fake train/test parquet datasets."""
    train_df = pd.DataFrame(
        {
            "f1": np.random.randn(10),
            "f2": np.random.randn(10),
            "label": np.random.randint(0, 2, size=10),
        }
    )
    test_df = pd.DataFrame(
        {
            "f1": np.random.randn(5),
            "f2": np.random.randn(5),
            "label": np.random.randint(0, 2, size=5),
        }
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    return SimpleNamespace(train=train_path, test=test_path)


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Patch build_pipeline to return a simple mock classifier with fit/predict."""
    mock_clf = MagicMock()
    mock_clf.fit.return_value = mock_clf
    mock_clf.predict.return_value = np.random.randint(0, 2, 5)
    mock_clf.predict_proba.return_value = np.random.rand(5, 2)

    # simulate nested estimator access for best_estimator_ handling
    mock_search = MagicMock()
    mock_search.fit.return_value = mock_search
    mock_search.predict.return_value = np.random.randint(0, 2, 5)
    mock_search.predict_proba.return_value = np.random.rand(5, 2)
    mock_search.best_estimator_ = MagicMock(named_steps={"model": MagicMock()})

    def fake_build_pipeline(model_key, grid_search, **kwargs):
        return mock_search if grid_search else mock_clf

    monkeypatch.setattr("scripts.run_baselines.build_pipeline", fake_build_pipeline)
    return mock_clf


def test_run_baselines_executes_without_error(mock_data, mock_pipeline, caplog):
    """Ensure run_baselines executes end-to-end with mocks and logs metrics."""
    caplog.set_level(logging.INFO)
    run_baselines(mock_data.train, mock_data.test, model_keys=["rf", "lr"], grid_search=True)

    # Verify logging occurred
    assert any("Training RF" in rec.message for rec in caplog.records)
    assert any("acc=" in rec.message for rec in caplog.records)


def test_run_baselines_handles_no_grid(mock_data, mock_pipeline):
    """Ensure pipeline runs correctly when grid search is disabled."""
    run_baselines(mock_data.train, mock_data.test, model_keys=["xgb"], grid_search=False)
    # The pipeline should still call fit and predict
    mock_pipeline.fit.assert_called()
    mock_pipeline.predict.assert_called()


def test_main_invokes_run_baselines(monkeypatch):
    """Test CLI main() integration with argument parsing."""
    from scripts import run_baselines as rb

    called = {}

    def fake_run_baselines(train_path, test_path, model_keys, grid_search):
        called["ok"] = (train_path, test_path, model_keys, grid_search)

    monkeypatch.setattr(rb, "run_baselines", fake_run_baselines)
    fake_args = [
        "--train",
        "data/train.parquet",
        "--test",
        "data/test.parquet",
        "--models",
        "rf",
        "lr",
        "--no-grid",
    ]
    monkeypatch.setattr(builtins, "exit", lambda _: None)
    monkeypatch.setattr("sys.argv", ["prog"] + fake_args)

    rb.main()
    assert "ok" in called
    assert called["ok"][2] == ["rf", "lr"]
    assert called["ok"][3] is False
