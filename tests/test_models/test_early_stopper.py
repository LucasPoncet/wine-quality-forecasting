import pytest
import torch
from torch import nn

from src.models.components.early_stopper import EarlyStopper


class DummyModel(nn.Module):
    """Minimal model to test state_dict save/restore."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)


@pytest.fixture
def model() -> nn.Module:
    torch.manual_seed(0)
    return DummyModel()


@pytest.fixture
def hyperparams() -> dict:
    return {"max_epoch": 10, "min_epoch_es": 2, "patience_es": 3}


def test_initialization_defaults(hyperparams):
    stopper = EarlyStopper(hyperparams)
    assert stopper.max_epoch == 10
    assert stopper.min_epoch == 2
    assert stopper.patience == 3
    assert stopper.metric_history == []
    assert stopper.wait == 0


def test_first_epoch_saves_weights(model, hyperparams):
    stopper = EarlyStopper(hyperparams)
    keep_training = stopper.step(model, epoch=0, metric_value=0.5)
    assert keep_training is True
    assert stopper.best_model_weights is not None
    assert len(stopper.metric_history) == 1


def test_improvement_resets_wait_and_saves(model, hyperparams):
    stopper = EarlyStopper(hyperparams)
    stopper.step(model, 0, 0.5)
    stopper.wait = 2
    new_metric = 0.8
    stopper.step(model, 3, new_metric)
    assert stopper.wait == 0
    assert abs(max(stopper.metric_history) - new_metric) < 1e-9


def test_no_improvement_increments_wait(model, hyperparams):
    stopper = EarlyStopper(hyperparams)
    stopper.step(model, 0, 1.0)
    stopper.metric_history.extend([1.0, 0.9])
    stopper.wait = 1
    stopper.step(model, 4, 0.8)
    assert stopper.wait == 2


def test_early_stopping_triggers_and_restores(model, hyperparams):
    stopper = EarlyStopper(hyperparams)
    stopper.step(model, 0, 0.5)
    # Alter model weights so restoration can be detected
    with torch.no_grad():
        model.fc.weight.fill_(10.0)
    original_weights = model.fc.weight.clone()
    # Sequence of non-improving metrics
    for e in range(3, 7):
        keep_training = stopper.step(model, e, 0.1)
        if not keep_training:
            break
    # Should have restored best (different) weights
    assert not torch.allclose(model.fc.weight, original_weights)
    assert keep_training is False


def test_stop_on_max_epoch(model, hyperparams):
    stopper = EarlyStopper(hyperparams)
    stopper.step(model, 0, 0.1)
    keep_training = stopper.step(model, stopper.max_epoch - 1, 0.1)
    assert keep_training is False


def test_restore_does_nothing_without_saved_weights(model, hyperparams):
    stopper = EarlyStopper(hyperparams)
    stopper.best_model_weights = None
    stopper._restore_weights(model)  # should not crash
