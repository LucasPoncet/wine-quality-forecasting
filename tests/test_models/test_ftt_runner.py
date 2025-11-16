from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import src.models.training.ftt_runner as ftt_runner


@pytest.fixture
def mock_loaders():
    """Return dummy TensorDatasets and mappings."""
    ds = torch.utils.data.TensorDataset(
        torch.rand(10, 3),  # x_num
        torch.zeros((10, 0), dtype=torch.long),  # x_cat
        torch.randint(0, 2, (10,)),  # y
    )
    return ds, ds, ds, {"region": {"A": 0}}, None


@pytest.fixture
def mock_load_parquet():
    """Mock parquet dataset (X_train, y_train, X_test, y_test)."""
    import pandas as pd

    X = pd.DataFrame({"f1": np.random.rand(10), "f2": np.random.rand(10), "region": ["A"] * 10})
    y = np.random.randint(0, 2, 10)
    return X, y, X.copy(), y.copy()


def test_run_ftt_pipeline_executes(monkeypatch, mock_loaders, tmp_path):
    """Ensure the FTT pipeline executes end-to-end without errors."""
    # --- Patch DatasetLoader to return dummy datasets ---
    monkeypatch.setattr(
        ftt_runner,
        "DatasetLoader",
        MagicMock(return_value=MagicMock(load_tabular_data=lambda: mock_loaders)),
    )

    # --- Patch feature engineering ---
    monkeypatch.setattr(
        ftt_runner,
        "add_engineered_features",
        MagicMock(return_value=(mock_loaders[:3], ["f1", "f2"], ["region"])),
    )

    # --- Patch model builder ---
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model

    # When model(...) is called, return a fake tensor-like output
    fake_output = torch.randn(10, 2)  # 10 samples, 2 classes
    fake_model.side_effect = None  # in case any previous side effects exist
    fake_model.return_value = fake_output

    monkeypatch.setattr(
        ftt_runner,
        "make_ftt_model",
        MagicMock(return_value=(fake_model, {"region": (10, 8)})),
    )

    # --- Patch scope and trainer ---
    fake_scope = MagicMock()
    monkeypatch.setattr(ftt_runner, "ScopeClassifier", MagicMock(return_value=fake_scope))

    class FakeTrainer:
        def __init__(self, *_, **__):
            self.model = fake_model
            self.scope = fake_scope

        def set_model(self, *_, **__):
            pass

        def set_scope(self, *_, **__):
            pass

        def set_data(self, *_, **__):
            pass

        def run(self):
            return [1.0], [1.0]

    monkeypatch.setattr(ftt_runner, "TrainerClassifier", FakeTrainer)

    # --- Patch plotting (avoid matplotlib) ---
    fake_plot = MagicMock()
    monkeypatch.setattr(ftt_runner, "plot_fold_mean", fake_plot)

    # --- Run pipeline ---
    acc, f1 = ftt_runner.run_ftt_pipeline(
        train_path=tmp_path / "train.parquet",
        test_path=tmp_path / "test.parquet",
        feature_ids=["A", "B"],
        device=torch.device("cpu"),
        max_epoch=2,
        plot=True,
    )

    # --- Assertions ---
    assert isinstance(acc, float)
    assert isinstance(f1, float)
    fake_plot.assert_called_once()
