import pytest
import torch

from src.models.components.losses import FocalLoss


@pytest.fixture
def logits_and_target():
    # Two-class example (batch_size=4)
    logits = torch.tensor([[1.0, 2.0], [0.5, 1.5], [2.0, 0.1], [1.0, 3.0]], dtype=torch.float32)
    targets = torch.tensor([1, 1, 0, 1], dtype=torch.long)
    return logits, targets


def test_focal_loss_basic_shape(logits_and_target):
    logits, targets = logits_and_target
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0, reduction="mean")
    loss = loss_fn(logits, targets)
    assert isinstance(loss.item(), float)
    assert loss.ndim == 0  # scalar output


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_reduction_modes(logits_and_target, reduction):
    logits, targets = logits_and_target
    loss_fn = FocalLoss(alpha=0.75, gamma=2.0, reduction=reduction)
    loss = loss_fn(logits, targets)
    if reduction == "none":
        assert loss.shape == targets.shape
    else:
        assert loss.ndim == 0


def test_invalid_reduction_raises(logits_and_target):
    logits, targets = logits_and_target
    with pytest.raises(ValueError):
        FocalLoss(reduction="bad")(logits, targets)


def test_invalid_target_dtype(logits_and_target):
    logits, _ = logits_and_target
    bad_targets = torch.tensor([0.5, 1.0, 0.0, 1.0])  # float
    loss_fn = FocalLoss()
    with pytest.raises(TypeError):
        loss_fn(logits, bad_targets)


def test_invalid_logit_shape_raises():
    logits = torch.randn(4, 2, 2)  # invalid 3D
    targets = torch.tensor([0, 1, 0, 1])
    loss_fn = FocalLoss()
    with pytest.raises(ValueError):
        loss_fn(logits, targets)


def test_alpha_list_supports_multiclass():
    logits = torch.randn(3, 3)
    targets = torch.tensor([0, 1, 2])
    loss_fn = FocalLoss(alpha=[1.0, 0.5, 0.2])
    loss = loss_fn(logits, targets)
    assert loss.shape == ()  # scalar
    assert loss.dtype == torch.float32


def test_device_consistency_cpu_to_cuda(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    logits = torch.randn(2, 2).cuda()
    targets = torch.tensor([0, 1], device=logits.device)
    loss_fn = FocalLoss()
    out = loss_fn(logits, targets)
    assert out.is_cuda
