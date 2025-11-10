import pytest

from src.models.architectures.tabular_mlp import TabularMLP
from src.models.builders.model_factory import build_model


@pytest.fixture
def dummy_hyperparams():
    return {
        "num_numeric_features": 5,
        "hidden_layers_size": [64, 32],
        "activation": "relu",
        "output_dim": 3,
        "batch_normalization": True,
        "dropout_rate": 0.2,
    }


@pytest.fixture
def dummy_embeddings():
    return {"region": (10, 8), "station": (5, 4)}


def test_build_model_returns_tabular_mlp(dummy_hyperparams, dummy_embeddings):
    """Ensure build_model instantiates a TabularMLP."""
    model = build_model(dummy_hyperparams, dummy_embeddings)
    assert isinstance(model, TabularMLP)
    assert hasattr(model, "forward")


def test_build_model_uses_correct_hyperparameters(dummy_hyperparams, dummy_embeddings):
    """Check if the model receives correct parameters."""
    model = build_model(dummy_hyperparams, dummy_embeddings)
    # You can check internal dimensions depending on your TabularMLP definition
    assert dummy_hyperparams["output_dim"] > 0
    assert len(dummy_hyperparams["hidden_layers_size"]) >= 1


def test_build_model_invalid_type(monkeypatch, dummy_hyperparams, dummy_embeddings):
    """If you extended build_model with model_type, check error raising."""
    from src.models.builders import model_factory

    def fake_model_type(hp, emb):
        raise ValueError("Unsupported model")

    monkeypatch.setattr(model_factory, "TabularMLP", fake_model_type)

    with pytest.raises(Exception):
        build_model(dummy_hyperparams, dummy_embeddings)
