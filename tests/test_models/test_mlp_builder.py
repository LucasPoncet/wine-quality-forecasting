from unittest.mock import MagicMock

import pytest

from src.models.builders import mlp_builder


@pytest.fixture
def dummy_hparams():
    """Provide minimal hyperparameter dictionary."""
    return {
        "hidden_layers_size": [64, 32],
        "activation": "relu",
        "output_dim": 2,
        "num_numeric_features": 5,
    }


def test_make_mlp_model_no_cat_cols(dummy_hparams):
    """When no categorical columns are given, embedding_sizes must be empty."""
    model, embedding_sizes = mlp_builder.make_mlp_model(dummy_hparams, cat_cols=[])
    assert isinstance(embedding_sizes, dict)
    assert embedding_sizes == {}
    assert model is not None


def test_make_mlp_model_no_mapping(dummy_hparams):
    """When categorical columns exist but mapping is None, embeddings must be empty."""
    cat_cols = ["region", "type"]
    model, embedding_sizes = mlp_builder.make_mlp_model(dummy_hparams, cat_cols, mapping=None)
    assert isinstance(embedding_sizes, dict)
    assert embedding_sizes == {}
    assert model is not None


def test_make_mlp_model_with_mapping(dummy_hparams):
    """When mapping is provided, embedding sizes must be computed correctly."""
    cat_cols = ["region", "category"]
    mapping = {"region": list(range(10)), "category": list(range(5))}
    model, embedding_sizes = mlp_builder.make_mlp_model(dummy_hparams, cat_cols, mapping)
    assert isinstance(embedding_sizes, dict)
    assert set(embedding_sizes.keys()) == set(cat_cols)
    for col, (vocab_size, emb_dim) in embedding_sizes.items():
        assert vocab_size == len(mapping[col])
        assert emb_dim >= 4  # as defined in the builder logic
    assert model is not None


def test_make_mlp_model_invokes_tabular_mlp(monkeypatch, dummy_hparams):
    """Ensure TabularMLP is called with expected arguments."""
    fake_model = MagicMock()
    called_args = {}

    def fake_tabular_mlp(hparams, emb_sizes):
        called_args["hparams"] = hparams
        called_args["emb_sizes"] = emb_sizes
        return fake_model

    monkeypatch.setattr(mlp_builder, "TabularMLP", fake_tabular_mlp)

    cat_cols = ["a"]
    mapping = {"a": [0, 1, 2, 3]}
    model, emb_sizes = mlp_builder.make_mlp_model(dummy_hparams, cat_cols, mapping)

    assert model is fake_model
    assert called_args["hparams"] == dummy_hparams
    assert "a" in called_args["emb_sizes"]
    assert isinstance(emb_sizes, dict)
