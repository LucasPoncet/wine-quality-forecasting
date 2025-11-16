import pytest
import torch

from src.preprocessing.utils.feature_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    compute_categorical_feature,
    compute_numeric_feature,
)


@pytest.fixture
def mock_data():
    return {
        "GDD": torch.tensor([100.0]),
        "rain_June": torch.tensor([10.0]),
        "rain_SepOct": torch.tensor([5.0]),
        "TX_summer": torch.tensor([25.0]),
        "TM_summer": torch.tensor([20.0]),
        "hot_days": torch.tensor([5.0]),
        "frost_days_Apr": torch.tensor([3.0]),
        "avg_TM_Apr": torch.tensor([1.0]),
        "price": torch.tensor([200.0]),
        "region": torch.tensor([4]),
        "station": torch.tensor([7]),
    }


def test_compute_numeric_feature_valid(mock_data):
    # "A" = GDD / (rain_June + rain_SepOct + 1)
    val_A = compute_numeric_feature("A", mock_data)
    assert torch.isclose(val_A, torch.tensor([100 / (10 + 5 + 1)]))
    # "B" = TX_summer - TM_summer
    assert compute_numeric_feature("B", mock_data).item() == 5.0
    # "I" = log(1 + price)
    assert torch.isclose(
        compute_numeric_feature("I", mock_data), torch.log1p(torch.tensor([200.0]))
    )


def test_compute_numeric_feature_invalid(mock_data):
    with pytest.raises(ValueError):
        compute_numeric_feature("Z", mock_data)


def test_compute_categorical_feature_valid(mock_data):
    # region * 1000 + station
    assert compute_categorical_feature("J", mock_data).item() == 4007


def test_compute_categorical_feature_invalid(mock_data):
    with pytest.raises(ValueError):
        compute_categorical_feature("X", mock_data)


def test_feature_dicts_are_consistent():
    assert all(isinstance(v, tuple) and len(v) == 2 for v in NUMERIC_FEATURES.values())
    assert all(isinstance(v, tuple) and len(v) == 2 for v in CATEGORICAL_FEATURES.values())
