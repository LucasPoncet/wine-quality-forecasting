import pytest
import torch
from torch.utils.data import TensorDataset

from src.preprocessing.feature_engineering import (
    add_engineered_features,
    drop_columns,
)


@pytest.fixture
def base_dataset():
    # 2 samples, with all numeric features needed by A and B
    x_num = torch.tensor(
        [
            [100.0, 10.0, 5.0, 25.0, 20.0],  # GDD, rain_June, rain_SepOct, TX_summer, TM_summer
            [50.0, 20.0, 10.0, 30.0, 25.0],
        ]
    )
    num_cols = ["GDD", "rain_June", "rain_SepOct", "TX_summer", "TM_summer"]

    x_cat = torch.tensor([[1, 2], [3, 4]])
    cat_cols = ["region", "station"]

    y = torch.tensor([1.0, 0.0])
    return TensorDataset(x_num, x_cat, y), num_cols, cat_cols


def test_add_engineered_features_adds_numeric_and_categorical(base_dataset):
    ds, num_cols, cat_cols = base_dataset
    new_ds, new_num_cols, new_cat_cols = add_engineered_features(
        (ds,), num_cols, cat_cols, feature_ids=["A", "B", "J"]
    )

    assert isinstance(new_ds, tuple)
    assert len(new_ds) == 1

    x_num, x_cat, y = new_ds[0].tensors
    # numeric features should increase by 2 ("A" and "B")
    assert x_num.shape[1] == len(num_cols) + 2
    # categorical features should increase by 1 ("J")
    assert x_cat.shape[1] == len(cat_cols) + 1

    assert "heat_to_rain_ratio" in new_num_cols
    assert "region_station_id" in new_cat_cols


def test_add_engineered_features_empty_feature_list(base_dataset):
    ds, num_cols, cat_cols = base_dataset
    new_ds, new_num_cols, new_cat_cols = add_engineered_features(
        (ds,), num_cols, cat_cols, feature_ids=[]
    )

    # nothing should change
    assert new_num_cols == num_cols
    assert new_cat_cols == cat_cols
    assert new_ds[0].tensors[0].shape[1] == len(num_cols)


def test_drop_columns_removes_correct_columns(base_dataset):
    ds, num_cols, cat_cols = base_dataset
    new_ds, new_num_cols = drop_columns((ds,), num_cols, drop_list=["rain_June"])
    assert "rain_June" not in new_num_cols
    x_num, _, _ = new_ds[0].tensors
    assert x_num.shape[1] == len(num_cols) - 1


def test_drop_columns_no_drops(base_dataset):
    ds, num_cols, cat_cols = base_dataset
    new_ds, new_num_cols = drop_columns((ds,), num_cols, drop_list=[])
    assert new_num_cols == num_cols
    x_num, _, _ = new_ds[0].tensors
    assert x_num.shape[1] == len(num_cols)
