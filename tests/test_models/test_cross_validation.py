import numpy as np
import pytest
import torch

from src.models.training.cross_validation import cross_val_run


class DummyModel(torch.nn.Module):
    """Tiny model for mocking."""

    def __init__(self, n_in=4, n_out=2):
        super().__init__()
        self.fc = torch.nn.Linear(n_in, n_out)

    def forward(self, x_num, x_cat):
        # Simulate output logits
        batch_size = x_num.size(0)
        return torch.randn(batch_size, 2)


@pytest.fixture
def dummy_data():
    n_samples = 20
    x_num = torch.randn(n_samples, 4)
    x_cat = torch.randint(0, 3, (n_samples, 2))
    y = torch.randint(0, 2, (n_samples,))
    return x_num, x_cat, y


@pytest.fixture
def dummy_hyperparams():
    return {"max_epoch": 1, "batch_size": 4}


def test_cross_val_basic(monkeypatch, dummy_data, dummy_hyperparams):
    """Ensure cross_val_run returns the mean of best validation accuracies."""

    x_num, x_cat, y = dummy_data

    # --- Mock ScopeClassifier ---
    class MockScope:
        def __init__(self, *_, **__):
            pass

    # --- Mock TrainerClassifier ---
    class MockTrainer:
        called_folds = 0

        def __init__(self, hyper):
            self.hyper = hyper

        def set_model(self, model, device):  # noqa: ARG002
            pass

        def set_scope(self, scope):  # noqa: ARG002
            pass

        def set_data(self, *_, **__):
            pass

        def run(self):
            # Return fake validation history increasing per fold
            MockTrainer.called_folds += 1
            fake_val_hist = [10 * MockTrainer.called_folds, 20 * MockTrainer.called_folds]
            return [], fake_val_hist

    # --- Monkeypatch dependencies ---
    import src.models.training.cross_validation as cv_module

    monkeypatch.setattr(cv_module, "ScopeClassifier", MockScope)
    monkeypatch.setattr(cv_module, "TrainerClassifier", MockTrainer)

    # --- Define model builder ---
    def build_model_fn():
        return DummyModel()

    # --- Run ---
    mean_acc = cross_val_run(
        x_num, x_cat, y, build_model_fn, dummy_hyperparams, n_splits=3, seed=42
    )

    # --- Assertions ---
    # Each fold should have val_hist = [10, 20], [20, 40], [30, 60]
    # best per fold = 20, 40, 60 → mean = 40
    assert np.isclose(mean_acc, 40.0, atol=1e-6)
    assert isinstance(mean_acc, float)


def test_cross_val_different_splits(monkeypatch, dummy_data, dummy_hyperparams):
    """Check the number of splits and correct averaging logic."""
    x_num, x_cat, y = dummy_data

    class MockScope:
        def __init__(self, *_, **__):
            pass

    class MockTrainer:
        calls = 0

        def __init__(self, *_):
            pass

        def set_model(self, *_, **__):
            pass

        def set_scope(self, *_, **__):
            pass

        def set_data(self, *_, **__):
            pass

        def run(self):
            MockTrainer.calls += 1
            return [], [MockTrainer.calls * 5.0]

    import src.models.training.cross_validation as cv_module

    monkeypatch.setattr(cv_module, "ScopeClassifier", MockScope)
    monkeypatch.setattr(cv_module, "TrainerClassifier", MockTrainer)

    def build_model_fn():
        return DummyModel()

    mean_acc = cross_val_run(x_num, x_cat, y, build_model_fn, dummy_hyperparams, n_splits=4)
    # Fold best accuracies = [5, 10, 15, 20] → mean = 12.5
    assert np.isclose(mean_acc, 12.5, atol=1e-6)


def test_cross_val_empty_inputs(monkeypatch, dummy_hyperparams):
    """Edge case: empty dataset should raise ValueError."""
    x_num = torch.empty((0, 3))
    x_cat = torch.empty((0, 2))
    y = torch.empty((0,), dtype=torch.long)

    import src.models.training.cross_validation as cv_module

    monkeypatch.setattr(cv_module, "ScopeClassifier", object)
    monkeypatch.setattr(cv_module, "TrainerClassifier", object)

    def build_model_fn():
        return DummyModel()

    with pytest.raises(ValueError):
        cross_val_run(x_num, x_cat, y, build_model_fn, dummy_hyperparams)
