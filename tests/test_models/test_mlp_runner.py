import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.models.training import mlp_runner


@pytest.fixture
def mock_dataset(tmp_path):
    """Create minimal fake tensors & dataset."""
    n, d = 20, 5
    x_num = torch.rand(n, d)
    x_cat = torch.zeros((n, 0), dtype=torch.long)
    y = torch.randint(0, 2, (n,))
    ds = torch.utils.data.TensorDataset(x_num, x_cat, y)
    return ds


@pytest.fixture
def mock_loaders(mock_dataset):
    """Return three identical fake datasets."""
    return mock_dataset, mock_dataset, mock_dataset, {}, None


@pytest.fixture
def mock_load_parquet():
    """Mock load_parquet_dataset to return simple arrays."""
    X = np.random.rand(10, 3)
    y = np.random.randint(0, 2, 10)
    import pandas as pd

    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    X_df["cat"] = ["a"] * len(X_df)
    return X_df, y, X_df, y


def test_mlp_runner_executes(monkeypatch, mock_loaders, mock_load_parquet, tmp_path):
    """Ensure run_mlp_pipeline executes end-to-end without crash."""

    # Patch data loaders and preprocessors

    monkeypatch.setattr(
        mlp_runner,
        "DatasetLoader",
        MagicMock(return_value=MagicMock(load_tabular_data=lambda: mock_loaders)),
    )
    monkeypatch.setattr(
        mlp_runner, "load_parquet_dataset", MagicMock(return_value=mock_load_parquet)
    )
    monkeypatch.setattr(
        mlp_runner,
        "add_engineered_features",
        MagicMock(return_value=(mock_loaders[:3], ["f1", "f2"], [])),
    )
    monkeypatch.setattr(
        mlp_runner, "build_cat_mapping", MagicMock(return_value=({}, torch.empty((10, 0)), []))
    )

    # Fake model behaving like a PyTorch module

    def fake_forward(x_num, x_cat):
        batch_size = x_num.shape[0]
        return torch.rand(batch_size, 2)

    fake_model = MagicMock()
    fake_model.side_effect = fake_forward
    fake_model.to = MagicMock(return_value=fake_model)  # Ensure .to() returns itself

    # TabularMLP always returns this fake model
    monkeypatch.setattr(mlp_runner, "TabularMLP", MagicMock(return_value=fake_model))

    # Fake TrainerClassifier implementation

    class FakeTrainer:
        def __init__(self, *_, **__):
            self.model = fake_model
            self.scope = None
            self.run = lambda: ([1.0], [1.0])

        def set_model(self, model, device=None):
            self.model = model

        def set_scope(self, scope):
            self.scope = scope

        def set_data(self, **kwargs):
            pass

    monkeypatch.setattr(mlp_runner, "TrainerClassifier", FakeTrainer)

    # Patch remaining heavy or external components

    monkeypatch.setattr(mlp_runner, "ScopeClassifier", MagicMock())
    monkeypatch.setattr(mlp_runner, "evaluate_metrics", MagicMock(return_value=(0.8, 0.75, 0.9)))
    monkeypatch.setattr(mlp_runner.plt, "show", lambda: None)

    # Run the actual pipeline

    device = torch.device("cpu")

    mlp_runner.run_mlp_pipeline(
        train_path=tmp_path / "train.parquet",
        test_path=tmp_path / "test.parquet",
        feature_ids=["A", "B"],
        device=device,
        max_epoch=3,
    )

    # Validation

    mock_evaluate = mlp_runner.evaluate_metrics
    assert isinstance(mock_evaluate, MagicMock)
    mock_evaluate.assert_called_once()
    logging.info("run_mlp_pipeline executed successfully")
