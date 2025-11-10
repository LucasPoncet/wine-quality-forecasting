import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.models.builders.preprocessors import (
    CAT_FEATURES,
    NUM_FEATURES,
    _categorical_encoder,
    _make_preprocessor,
)


@pytest.fixture
def sample_dataframe():
    """Return a minimal DataFrame with all required numeric and categorical features."""
    data = {
        **{col: np.random.randn(5) for col in NUM_FEATURES},
        **{col: np.random.choice(["A", "B", "C"], 5) for col in CAT_FEATURES},
    }
    return pd.DataFrame(data)


# Tests for _categorical_encoder


def test_categorical_encoder_linear():
    enc = _categorical_encoder("linear")
    assert isinstance(enc, OneHotEncoder)
    assert getattr(enc, "handle_unknown", None) == "ignore"
    assert getattr(enc, "sparse_output", True) is True


def test_categorical_encoder_tree():
    enc = _categorical_encoder("tree")
    assert isinstance(enc, OrdinalEncoder)
    assert getattr(enc, "handle_unknown", None) == "use_encoded_value"
    assert getattr(enc, "unknown_value", None) == -1


def test_categorical_encoder_invalid_fallback():
    """Invalid input should default to tree-style (OrdinalEncoder)."""
    enc = _categorical_encoder("something_else")
    assert isinstance(enc, OrdinalEncoder)


# Tests for _make_preprocessor


@pytest.mark.parametrize("model_key", ["lr", "xgb", "rf", "hgb"])
def test_make_preprocessor_valid(model_key, sample_dataframe):
    """Ensure the preprocessor builds and transforms data without errors."""
    pre = _make_preprocessor(model_key)
    assert isinstance(pre, ColumnTransformer)

    # Fit and transform on the sample data
    Xt = pre.fit_transform(sample_dataframe)
    if sp.issparse(Xt):
        assert Xt.shape[0] == len(sample_dataframe)
    else:
        assert len(Xt.shape) == 2


def test_lr_preprocessor_includes_scaler():
    pre = _make_preprocessor("lr")
    transformer_names = [name for name, _, _ in getattr(pre, "transformers", [])]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_xgb_preprocessor_no_standard_scaler(sample_dataframe):
    """Tree-based models shouldn't include scaling pipeline."""
    pre = _make_preprocessor("xgb")
    pre.fit(sample_dataframe)
    num_transformer = dict(pre.named_transformers_)["num"]
    assert not isinstance(num_transformer, Pipeline)
    assert hasattr(num_transformer, "fit")


def test_preprocessor_missing_column_raises(sample_dataframe):
    """Dropping one column should cause an informative error."""
    df_missing = sample_dataframe.drop(columns=[NUM_FEATURES[0]])
    pre = _make_preprocessor("lr")
    with pytest.raises(ValueError, match="not a column of the dataframe"):
        pre.fit_transform(df_missing)
