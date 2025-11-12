import numpy as np
import pytest

from src.models.architectures.tabnet import TabNetClassifier
from src.models.builders.tabnet_builder import make_tabnet_model


@pytest.fixture
def dummy_hyperparams():
    return {
        "n_steps": 5,
        "n_d": 64,
        "n_a": 64,
        "shared_layers": 1,
        "step_layers": 2,
        "emb_dropout": 0.1,
        "virtual_batch": 32,
    }


def test_make_tabnet_model_with_valid_mapping(dummy_hyperparams):
    """Ensure model and embedding sizes are built correctly when mapping is valid."""
    num_cols = ["f1", "f2", "f3"]
    cat_cols = ["region", "cepage"]
    onehot_mapping = {
        "region": {"A": 0, "B": 1, "C": 2},
        "cepage": {"x": 0, "y": 1},
    }

    model, embedding_sizes = make_tabnet_model(
        dummy_hyperparams, num_cols, cat_cols, onehot_mapping
    )

    # Type and key checks
    assert isinstance(model, TabNetClassifier)
    assert set(embedding_sizes.keys()) == set(cat_cols)

    # Dimension logic check
    for col, (vocab, dim) in embedding_sizes.items():
        assert vocab == len(onehot_mapping[col])
        assert dim == max(8, int(np.sqrt(vocab)))  # matches function logic


def test_make_tabnet_model_skips_unmapped_categories(dummy_hyperparams):
    """If cat_cols includes names missing in mapping, they should be skipped."""
    num_cols = ["f1"]
    cat_cols = ["region", "station"]  # station not in mapping
    onehot_mapping = {"region": {"A": 0, "B": 1, "C": 2}}

    model, embedding_sizes = make_tabnet_model(
        dummy_hyperparams, num_cols, cat_cols, onehot_mapping
    )

    assert "region" in embedding_sizes
    assert "station" not in embedding_sizes
    assert isinstance(model, TabNetClassifier)


def test_make_tabnet_model_with_empty_mapping(dummy_hyperparams):
    """If mapping is empty, embedding_sizes should be empty and model still valid."""
    num_cols = ["x1", "x2"]
    cat_cols = ["region"]
    onehot_mapping = {}

    model, embedding_sizes = make_tabnet_model(
        dummy_hyperparams, num_cols, cat_cols, onehot_mapping
    )

    assert isinstance(model, TabNetClassifier)
    assert embedding_sizes == {}


def test_make_tabnet_model_embedding_dimensions_are_min_8(dummy_hyperparams):
    """Embedding dims must not go below 8, even for small vocabularies."""
    num_cols = ["x1"]
    cat_cols = ["region"]
    onehot_mapping = {"region": {"A": 0, "B": 1}}  # sqrt(2)=1.4 < 8 ⇒ must use 8

    _, embedding_sizes = make_tabnet_model(dummy_hyperparams, num_cols, cat_cols, onehot_mapping)
    vocab, dim = embedding_sizes["region"]

    assert vocab == 2
    assert dim == 8
