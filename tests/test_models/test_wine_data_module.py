import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset

from src.models.data.wine_data_module import DatasetLoader


@pytest.fixture
def sample_data(tmp_path):
    """Create synthetic train/test parquet files for testing."""
    train_df = pd.DataFrame(
        {
            "feature_num": [1.0, 2.5, None, float("inf"), -float("inf")],
            "feature_cat": ["A", "B", "A", None, "C"],
            "label": [0, 1, 1, 0, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "feature_num": [0.5, 1.5],
            "feature_cat": ["A", "C"],
            "label": [0, 1],
        }
    )

    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)
    return str(train_path), str(test_path)


def test_create_index_mapping(sample_data):
    train_path, test_path = sample_data
    df = pd.DataFrame({"cat": ["B", "A", "C", "A"]})
    loader = DatasetLoader(train_path, test_path, cat_cols=["cat"])
    mapping = loader.create_index_mapping(df, ["cat"])
    assert set(mapping["cat"].keys()) == {"A", "B", "C"}
    assert all(isinstance(v, int) for v in mapping["cat"].values())


def test_df_to_dataset_numeric_and_categorical(sample_data):
    train_path, test_path = sample_data
    loader = DatasetLoader(
        train_path,
        test_path,
        target_col="label",
        num_cols=["feature_num"],
        cat_cols=["feature_cat"],
    )
    df = pd.read_parquet(train_path)
    loader.cat_mapping = loader.create_index_mapping(df, ["feature_cat"])
    ds = loader._df_to_dataset(df)
    assert isinstance(ds, TensorDataset)
    x_num, x_cat, y = ds.tensors
    assert x_num.shape[1] == 1
    assert x_cat.shape[1] == 1
    assert torch.all((x_cat >= 0) & (x_cat < len(loader.cat_mapping["feature_cat"])))


def test_load_tabular_data_with_auto_split(sample_data):
    train_path, test_path = sample_data
    loader = DatasetLoader(
        train_path,
        test_path,
        target_col="label",
        num_cols=["feature_num"],
        cat_cols=["feature_cat"],
        valid_frac=0.4,
    )
    train_ds, valid_ds, test_ds, mapping, n_classes = loader.load_tabular_data()

    # Check dataset types and outputs
    assert all(isinstance(ds, TensorDataset) for ds in [train_ds, valid_ds, test_ds])
    assert isinstance(mapping, dict)
    assert n_classes == 2
    assert all(isinstance(v, dict) for v in mapping.values())
    assert len(train_ds.tensors) == 3
    assert len(valid_ds.tensors) == 3


def test_load_tabular_data_with_split_column(sample_data):
    train_path, test_path = sample_data
    df = pd.read_parquet(train_path)
    df["split"] = ["train", "train", "valid", "valid", "valid"]
    df.to_parquet(train_path)
    loader = DatasetLoader(
        train_path,
        test_path,
        target_col="label",
        num_cols=["feature_num"],
        cat_cols=["feature_cat"],
    )
    train_ds, valid_ds, test_ds, _, _ = loader.load_tabular_data()
    assert len(train_ds.tensors[0]) + len(valid_ds.tensors[0]) == len(df)


def test_load_tabular_data_raises_for_missing_cols(sample_data):
    train_path, test_path = sample_data
    loader = DatasetLoader(train_path, test_path)
    with pytest.raises(ValueError, match="At least one of num_cols or cat_cols"):
        loader.load_tabular_data()
