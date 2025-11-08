import pytest
import torch.nn as nn

from src.models.components.activation import get_activation


@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("relu", nn.ReLU),
        ("sigmoid", nn.Sigmoid),
        ("tanh", nn.Tanh),
        ("gelu", nn.GELU),
        ("leakyrelu", nn.LeakyReLU),
        ("elu", nn.ELU),
    ],
)
def test_get_activation_valid(name, expected_type):
    """Check that known activation names return the right nn.Module."""
    act = get_activation(name)
    assert isinstance(act, expected_type), f"{name} should return {expected_type.__name__}"


def test_get_activation_none_and_linear():
    """None and 'linear' should both return None."""
    assert get_activation(None) is None
    assert get_activation("linear") is None


def test_get_activation_case_insensitive():
    """Activation lookup should not depend on case."""
    assert isinstance(get_activation("ReLU"), nn.ReLU)
    assert isinstance(get_activation("SIGMOID"), nn.Sigmoid)
    assert isinstance(get_activation("tAnH"), nn.Tanh)


def test_get_activation_invalid_raises():
    """Unsupported names should raise ValueError."""
    with pytest.raises(ValueError):
        get_activation("not_an_activation")
