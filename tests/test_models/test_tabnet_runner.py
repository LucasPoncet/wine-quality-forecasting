from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import src.models.training.tabnet_runner as tabnet_runner


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
    """Mocked parquet dataset (X_train, y_train, X_test, y_test)."""
    import pandas as pd

    X = pd.DataFrame(
        {
            "f1": np.random.rand(10),
            "f2": np.random.rand(10),
            "region": ["A"] * 10,
        }
    )
    y = np.random.randint(0, 2, 10)
    return X, y, X.copy(), y.copy()


def test_run_tabnet_pipeline_executes(monkeypatch, mock_loaders, mock_load_parquet, tmp_path):
    """Ensure the TabNet pipeline executes end-to-end without errors."""
    # --- Patch heavy functions/classes ---
    monkeypatch.setattr(
        tabnet_runner,
        "DatasetLoader",
        MagicMock(return_value=MagicMock(load_tabular_data=lambda: mock_loaders)),
    )
    monkeypatch.setattr(
        tabnet_runner, "load_parquet_dataset", MagicMock(return_value=mock_load_parquet)
    )
    monkeypatch.setattr(
        tabnet_runner,
        "add_engineered_features",
        MagicMock(return_value=(mock_loaders[:3], ["f1", "f2"], ["region"])),
    )
    monkeypatch.setattr(
        tabnet_runner,
        "make_tabnet_model",
        MagicMock(return_value=(MagicMock(), {"region": (10, 8)})),
    )

    class DummyTabNet:
        def __init__(self, *_, **__):
            pass

        def forward(self, *a, **kw):
            # Use the first tensor’s batch dimension if available
            batch_size = a[0].shape[0] if a and hasattr(a[0], "shape") else 10
            return torch.rand(batch_size, 2)

        def __call__(self, *a, **kw):
            return self.forward(*a, **kw)

        def to(self, device):
            return self

    monkeypatch.setattr(tabnet_runner, "TabNetClassifier", DummyTabNet)

    # Fake Scope + Trainer
    class FakeTrainer:
        def __init__(self, *_, **__):
            self.model = MagicMock(return_value=torch.rand(10, 2))
            self.scope = None

        def set_model(self, model, device=None):
            self.model = model

        def set_scope(self, scope):
            self.scope = scope

        def set_data(self, **kwargs):
            pass

        def run(self):
            return [1.0], [1.0]

    monkeypatch.setattr(tabnet_runner, "TrainerClassifier", FakeTrainer)
    monkeypatch.setattr(tabnet_runner, "ScopeClassifier", MagicMock())
    monkeypatch.setattr(tabnet_runner, "evaluate_metrics", MagicMock(return_value=(0.8, 0.75, 0.9)))
    monkeypatch.setattr(tabnet_runner, "plot_fold_mean", MagicMock())

    # --- Run ---
    acc, f1, auc = tabnet_runner.run_tabnet_pipeline(
        train_path=tmp_path / "train.parquet",
        test_path=tmp_path / "test.parquet",
        feature_ids=["A", "B"],
        device=torch.device("cpu"),
        max_epoch=2,
        plot=True,
    )

    # --- Assertions ---
    assert all(isinstance(x, float) for x in (acc, f1, auc))
    assert (acc, f1, auc) == (0.8, 0.75, 0.9)
    mock_evaluate = tabnet_runner.plot_fold_mean
    assert isinstance(mock_evaluate, MagicMock)
    mock_evaluate.assert_called_once()
