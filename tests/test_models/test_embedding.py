import pandas as pd
import pytest
import torch

from src.models.components.embedding import build_cat_mapping


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["Bordeaux", "Burgundy", "Bordeaux", "Loire"],
            "grape": ["Cabernet", "Pinot", "Cabernet", "Chenin"],
            "year": [2020, 2021, 2020, 2022],  # numeric column (should ignore if not in cat_cols)
        }
    )


def test_build_cat_mapping_creates_mapping(sample_df):
    mapping, x_cat, vocab_sizes = build_cat_mapping(sample_df, ["region", "grape"])
    # mapping keys
    assert set(mapping.keys()) == {"region", "grape"}
    # vocab sizes match uniques
    assert vocab_sizes == [3, 3]
    # tensor shape = (rows, categorical columns)
    assert x_cat.shape == (len(sample_df), 2)
    # tensor dtype
    assert x_cat.dtype == torch.long


def test_existing_mapping_is_reused(sample_df):
    # first mapping
    mapping1, _, _ = build_cat_mapping(sample_df, ["region"])
    # simulate new data with an unseen value
    df2 = pd.DataFrame({"region": ["Bordeaux", "Alsace"]})
    mapping2, x_cat2, vocab2 = build_cat_mapping(df2, ["region"], existing=mapping1)
    assert mapping2 is mapping1  # same object, updated
    assert "Alsace" in mapping2["region"]
    assert vocab2[0] == len(mapping2["region"])
    # encoding should reflect new index
    assert x_cat2[-1, 0].item() == mapping2["region"]["Alsace"]


def test_build_cat_mapping_from_dict_input():
    data = {"color": ["red", "white", "red"]}
    mapping, x_cat, vocab_sizes = build_cat_mapping(data, ["color"])
    assert "color" in mapping
    assert vocab_sizes == [2]
    assert torch.equal(x_cat[:, 0], torch.tensor([0, 1, 0]))


def test_missing_column_raises_keyerror(sample_df):
    with pytest.raises(KeyError):
        build_cat_mapping(sample_df, ["nonexistent"])


def test_empty_cat_cols_returns_empty_tensor(sample_df):
    mapping, x_cat, vocab = build_cat_mapping(sample_df, [])
    assert x_cat.numel() == 0
    assert vocab == []
