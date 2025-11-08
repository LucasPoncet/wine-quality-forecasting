import pytest
import torch

from src.models.components.early_stopper import EarlyStopper
from src.models.components.scope import ScopeClassifier


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def base_hparams():
    return {"learning_rate": 0.001, "max_epoch": 5}


def test_optimizer_initialization(model, base_hparams):
    scope = ScopeClassifier(model, base_hparams)
    assert isinstance(scope.optimizer, torch.optim.Adam)
    assert isinstance(scope.criterion, torch.nn.Module)


def test_scheduler_reduce_on_plateau(model):
    hparams = {"learning_rate": 0.001, "patience_lr": 2, "max_epoch": 10}
    scope = ScopeClassifier(model, hparams)
    assert isinstance(scope.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    scope.step_scheduler(metric_value=0.8)  # should not raise


def test_scheduler_one_cycle(model):
    hparams = {"learning_rate": 0.001, "cycle_lr": 0.01, "max_epoch": 5}
    scope = ScopeClassifier(model, hparams, steps_per_epoch=10)
    assert isinstance(scope.scheduler, torch.optim.lr_scheduler.OneCycleLR)
    scope.optimizer.step()
    scope.step_scheduler()  # one cycle scheduler does not need metric


def test_early_stopping_integration(model):
    hparams = {"learning_rate": 0.001, "max_epoch": 5, "early_stopping": True}
    scope = ScopeClassifier(model, hparams)
    assert isinstance(scope.early_stopping, EarlyStopper)
    should_continue = scope.maybe_stop(epoch=0, val_metric=0.5)
    assert isinstance(should_continue, bool)


def test_state_dict_and_restore(model):
    hparams = {"learning_rate": 0.001, "max_epoch": 5}
    scope = ScopeClassifier(model, hparams)
    state = scope.state_dict()
    scope.load_state_dict(state)  # should reload cleanly
    assert "optimizer" in state
