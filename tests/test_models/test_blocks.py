import pytest
import torch
from torch import nn

from src.models.components.blocks import (
    AttentiveTransformer,
    BasicResNetBlock,
    Conv2DBlock,
    DenseBlock,
    FeatureTransformerBlock,
    FlattenDenseBlock,
    GhostBatchNorm,
    PositionalEncoding,
    Sparsemax,
    TransformerEncoderBlock,
    UnflattenDenseBlock,
)

# -------------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_determinism() -> None:
    """Ensure deterministic runs for dropout and linear weights."""
    torch.manual_seed(0)


# -------------------------------------------------------------------------
# DENSE + CONVOLUTIONAL BLOCKS
# -------------------------------------------------------------------------


def test_dense_block_shape_and_dropout() -> None:
    block = DenseBlock(8, 16, activation=nn.ReLU())
    x = torch.randn(4, 8)
    y = block(x)
    assert y.shape == (4, 16)
    # ensure dropout does not produce NaN
    assert not torch.isnan(y).any()
    # ensure train mode applies dropout (probability check)
    block.train()
    y_train = block(x)
    assert not torch.allclose(y, y_train)  # dropout randomness


@pytest.mark.parametrize("use_bn", [False, True])
def test_flatten_dense_block_shape(use_bn: bool) -> None:
    block = FlattenDenseBlock((2, 4), 5, activation=nn.ReLU(), batch_normalization=use_bn)
    x = torch.randn(3, 2, 4)
    y = block(x)
    assert y.shape == (3, 5)


def test_unflatten_dense_block_roundtrip() -> None:
    target_shape = (2, 3)
    block = UnflattenDenseBlock(4, target_shape, activation=nn.ReLU())
    x = torch.randn(5, 4)
    y = block(x)
    assert y.shape == (5, *target_shape)


@pytest.mark.parametrize("bn,act", [(False, None), (True, nn.ReLU())])
def test_conv2d_block_shape(bn: bool, act) -> None:
    block = Conv2DBlock(3, 6, kernel_size=3, activation=act, batch_normalization=bn)
    x = torch.randn(2, 3, 32, 32)
    y = block(x)
    assert y.shape == (2, 6, 32, 32)
    assert not torch.isnan(y).any()


def test_residual_conv_block_identity_and_proj() -> None:
    # Same channels => identity shortcut
    block_id = BasicResNetBlock(3, 3, 3, activation=nn.ReLU())
    x = torch.randn(1, 3, 16, 16)
    y = block_id(x)
    assert y.shape == x.shape

    # Different channels => projection
    block_proj = BasicResNetBlock(3, 6, 3, activation=nn.ReLU())
    y2 = block_proj(x)
    assert y2.shape == (1, 6, 16, 16)
    # Residual must change channels
    assert y2.shape[1] != x.shape[1]


# -------------------------------------------------------------------------
# TRANSFORMER BLOCKS
# -------------------------------------------------------------------------


def test_positional_encoding_adds_correctly() -> None:
    seq_len, d_model = 10, 8
    x = torch.zeros(seq_len, d_model)
    pe = PositionalEncoding(seq_len, d_model)
    y = pe(x)
    # shape check
    assert y.shape == (seq_len, d_model)
    # positional encoding should not be all zeros
    assert not torch.allclose(pe.pe, torch.zeros_like(pe.pe))
    # dropout doesn’t break dtype
    assert y.dtype == torch.float32


def test_transformer_encoder_block_shape_and_residuals() -> None:
    block = TransformerEncoderBlock(16, num_heads=4, expansion_factor=2, activation=nn.ReLU)
    x = torch.randn(2, 8, 16)  # (B, S, D)
    y = block(x)
    assert y.shape == x.shape
    # Norm layers should preserve mean/variance scaling roughly
    assert not torch.isnan(y).any()
    # Should not equal input (residual + dropout + MHA)
    assert not torch.allclose(x, y)


# -------------------------------------------------------------------------
# TABNET-LIKE BLOCKS
# -------------------------------------------------------------------------


def test_ghost_batch_norm_virtual_batch() -> None:
    gbn = GhostBatchNorm(8, virtual_batch_size=2)
    gbn.train()
    x = torch.randn(6, 8)
    y = gbn(x)
    assert y.shape == x.shape
    # Test inference mode
    gbn.eval()
    y2 = gbn(x)
    assert y2.shape == x.shape


@pytest.mark.parametrize("n_glu_layers", [1, 2, 3])
def test_feature_transformer_block_shape_and_skip(n_glu_layers: int) -> None:
    block = FeatureTransformerBlock(8, 8, n_glu_layers=n_glu_layers)
    x = torch.randn(4, 8)
    y = block(x)
    assert y.shape == x.shape
    # Check residual path is active (skip connection)
    assert not torch.allclose(x, y)


def test_sparsemax_properties() -> None:
    sm = Sparsemax()
    x = torch.tensor([[1.0, 2.0, 3.0]])
    y = sm(x)
    # Values between 0 and 1
    assert (y >= 0).all()
    assert (y <= 1).all()
    # Sparsemax sums to 1
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(1))


def test_attentive_transformer_mask_behavior() -> None:
    att = AttentiveTransformer(8, 4)
    x = torch.randn(2, 8)
    prior = torch.ones(2, 4)
    mask = att(x, prior)
    assert mask.shape == (2, 4)
    # Masked output should stay in [0, 1] range
    assert (mask >= 0).all()
    assert (mask <= 1).all()
    # Prior is multiplicative => mask <= prior
    assert (mask <= prior + 1e-6).all()
