import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.training.trainer_tabular import TrainerClassifier

# Fixtures


@pytest.fixture
def dummy_model():
    """A minimal model that takes x_num, x_cat and returns logits."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(6, 3)

        def forward(self, x_num, x_cat):
            x = torch.cat([x_num, x_cat], dim=1)
            return self.fc(x)

    return DummyModel()


@pytest.fixture
def dummy_data():
    """Return small numeric & categorical tensors."""
    x_num = torch.randn(20, 3)
    x_cat = torch.randint(0, 3, (20, 3))
    y = torch.randint(0, 3, (20,))
    return x_num, x_cat, y


@pytest.fixture
def dummy_scope(dummy_model):
    """A minimal scope object with optimizer, criterion, scheduler, and early stopping."""
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)
    criterion = nn.CrossEntropyLoss()

    class DummyEarlyStopping:
        def __init__(self):
            self.calls = []

        def step(self, model, epoch, metric):
            self.calls.append((epoch, metric))
            return True

    class DummyScope:
        def __init__(self):
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.criterion = criterion
            self.early_stopping = DummyEarlyStopping()

    return DummyScope()


@pytest.fixture
def trainer(dummy_model, dummy_scope):
    hp = {"max_epoch": 3, "batch_size": 8}
    t = TrainerClassifier(hp)
    t.set_model(dummy_model, torch.device("cpu"))
    t.set_scope(dummy_scope)
    return t


# Tests


def test_setters_and_dataloaders(trainer, dummy_data):
    x_num, x_cat, y = dummy_data
    trainer.set_data((x_num, x_cat), y, (x_num, x_cat), y)
    assert isinstance(trainer.train_loader, DataLoader)
    assert isinstance(trainer.valid_loader, DataLoader)
    assert len(trainer.train_loader) > 0


def test_epoch_loop_train_and_eval(trainer, dummy_data):
    x_num, x_cat, y = dummy_data
    ds = TensorDataset(x_num, x_cat, y)
    loader = DataLoader(ds, batch_size=5)
    loss, acc = trainer._epoch_loop(loader, train_phase=True)
    assert 0 <= acc <= 100
    assert loss >= 0


def test_run_training_simple(trainer, dummy_data):
    x_num, x_cat, y = dummy_data
    trainer.set_data((x_num, x_cat), y, (x_num, x_cat), y)
    train_hist, valid_hist = trainer.run()
    assert len(train_hist) == len(valid_hist)
    assert max(valid_hist) <= 100
    assert all(isinstance(v, float) for v in valid_hist)


def test_scheduler_and_early_stopping_called(trainer, dummy_data):
    x_num, x_cat, y = dummy_data
    trainer.set_data((x_num, x_cat), y, (x_num, x_cat), y)
    trainer.run()
    assert len(trainer.scope.early_stopping.calls) > 0


def test_guard_clauses(dummy_model):
    trainer = TrainerClassifier({"max_epoch": 1})
    with pytest.raises(RuntimeError):
        trainer.run()
