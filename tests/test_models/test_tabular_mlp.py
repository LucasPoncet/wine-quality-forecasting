import pytest
import torch
from torch import nn

from src.models.architectures.tabular_mlp import TabularMLP


@pytest.fixture
def sample_hyperparams():
    return {
        "hidden_layers_size": [16, 8],
        "activation": "relu",
        "batch_normalization": True,
        "dropout_rate": 0.1,
        "num_numeric_features": 4,
        "output_dim": 3,
    }


@pytest.fixture
def sample_embeddings():
    return {"region": (5, 3), "station": (4, 2)}


def test_model_initialization(sample_hyperparams, sample_embeddings):
    """Ensure the model initializes correctly with embeddings."""
    model = TabularMLP(sample_hyperparams, sample_embeddings)
    assert isinstance(model, nn.Module)
    # Check embedding layers were created
    assert set(model.emb_layers.keys()) == {"region", "station"}
    # Check classifier has expected output size
    last_layer = list(model.classifier.children())[-1]
    assert isinstance(last_layer, nn.Linear)
    assert last_layer.out_features == sample_hyperparams["output_dim"]


def test_forward_with_embeddings(sample_hyperparams, sample_embeddings):
    """Forward pass should return correct output shape when embeddings exist."""
    model = TabularMLP(sample_hyperparams, sample_embeddings)
    x_num = torch.randn(10, sample_hyperparams["num_numeric_features"])
    x_cat = torch.randint(0, 4, (10, len(sample_embeddings)))
    out = model(x_num, x_cat)
    assert out.shape == (10, sample_hyperparams["output_dim"])
    assert not torch.isnan(out).any()


def test_forward_without_embeddings(sample_hyperparams):
    """Model should handle empty embedding dict."""
    model = TabularMLP(sample_hyperparams, {})
    x_num = torch.randn(5, sample_hyperparams["num_numeric_features"])
    out = model(x_num, None)
    assert out.shape == (5, sample_hyperparams["output_dim"])
    assert not torch.isnan(out).any()


def test_forward_requires_numeric_input(sample_hyperparams, sample_embeddings):
    """Should raise if x_numeric is missing or has wrong shape."""
    model = TabularMLP(sample_hyperparams, sample_embeddings)
    x_cat = torch.stack(
        [torch.randint(0, vocab, (8,)) for vocab, _ in sample_embeddings.values()], dim=1
    )
    with pytest.raises(RuntimeError):
        _ = model(torch.randn(8, 2), x_cat)  # wrong num features


def test_model_parameter_count(sample_hyperparams, sample_embeddings):
    """Check that model has trainable parameters and they are all finite."""
    model = TabularMLP(sample_hyperparams, sample_embeddings)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0
    for p in model.parameters():
        assert torch.isfinite(p).all()
