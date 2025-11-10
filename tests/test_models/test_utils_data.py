from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.training import utils_data


@pytest.fixture
def parquet_data(tmp_path: Path):
    """Create minimal parquet train/test data."""
    train_df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "label": [0, 1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "f1": [10.0, 20.0, 30.0],
            "f2": [40.0, 50.0, 60.0],
            "label": [1, 0, 1],
        }
    )
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    return train_path, test_path, train_df, test_df


def test_load_parquet_dataset(parquet_data):
    """Ensure parquet datasets are loaded and split correctly."""
    train_path, test_path, train_df, test_df = parquet_data

    X_train, y_train, X_test, y_test = utils_data.load_parquet_dataset(
        train_path, test_path, "label"
    )

    # --- Shapes and types ---
    assert list(X_train.columns) == ["f1", "f2"]
    assert list(X_test.columns) == ["f1", "f2"]
    assert y_train.dtype == int
    assert y_test.dtype == int

    # --- Values ---
    np.testing.assert_array_equal(X_train.values, train_df[["f1", "f2"]].values)
    np.testing.assert_array_equal(y_test.values, test_df["label"].values)


def test_clean_tensor_nan_replaces_invalid_values():
    """Ensure NaN and Inf are replaced with zeros."""
    x_num = torch.tensor([[1.0, float("nan")], [float("inf"), -float("inf")]])
    x_cat = torch.zeros((2, 0))
    y = torch.tensor([0, 1])

    ds = torch.utils.data.TensorDataset(x_num.clone(), x_cat, y)
    cleaned = utils_data.clean_tensor_nan(ds)

    x_cleaned, _, _ = cleaned.tensors
    assert torch.all(torch.isfinite(x_cleaned))
    assert torch.all(x_cleaned >= 0.0) or torch.all(x_cleaned <= 0.0)
    assert torch.allclose(x_cleaned, torch.tensor([[1.0, 0.0], [0.0, 0.0]]), atol=1e-8)


def test_clean_tensor_nan_preserves_shape():
    """The dataset must keep the same tensor shapes."""
    x_num = torch.randn(3, 2)
    x_cat = torch.zeros((3, 1), dtype=torch.long)
    y = torch.randint(0, 2, (3,))

    ds = torch.utils.data.TensorDataset(x_num, x_cat, y)
    cleaned = utils_data.clean_tensor_nan(ds)

    for original, cleaned_tensor in zip(ds.tensors, cleaned.tensors, strict=True):
        assert original.shape == cleaned_tensor.shape
