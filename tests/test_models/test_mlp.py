import pytest
import torch
from torch import nn

from src.models.architectures.mlp import MLP


@pytest.fixture
def basic_hparams():
    return {
        "input_dim": 10,
        "output_dim": 3,
        "hidden_layers_size": [8, 4],
        "activation": "relu",
        "batch_normalization": True,
        "dropout_rate": 0.1,
    }


def test_mlp_initialization(basic_hparams):
    model = MLP(basic_hparams)
    assert isinstance(model, nn.Module)
    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == basic_hparams["output_dim"]


def test_mlp_forward_shape(basic_hparams):
    model = MLP(basic_hparams)
    x = torch.randn(16, basic_hparams["input_dim"])
    out = model(x)
    assert out.shape == (16, basic_hparams["output_dim"])
    assert torch.isfinite(out).all()


def test_mlp_fails_on_wrong_input_dim(basic_hparams):
    model = MLP(basic_hparams)
    x = torch.randn(8, basic_hparams["input_dim"] - 2)
    with pytest.raises(RuntimeError):
        _ = model(x)


def test_mlp_parameter_count(basic_hparams):
    model = MLP(basic_hparams)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
