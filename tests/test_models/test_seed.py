import random

import numpy as np
import pytest
import torch

from src.utils.seed import set_seed


def test_seed_reproducibility_int():
    set_seed(123)
    a1 = random.random()
    b1 = np.random.rand()
    t1 = torch.rand(3)

    set_seed(123)
    a2 = random.random()
    b2 = np.random.rand()
    t2 = torch.rand(3)

    assert a1 == pytest.approx(a2)
    assert b1 == pytest.approx(b2)
    assert torch.allclose(t1, t2)


def test_seed_reproducibility_string():
    set_seed("experiment_A")
    arr1 = np.random.randint(0, 100, size=5)
    set_seed("experiment_A")
    arr2 = np.random.randint(0, 100, size=5)
    assert np.array_equal(arr1, arr2)


def test_invalid_seed_type_raises():
    with pytest.raises(TypeError):
        set_seed(3.14)  # type: ignore


def test_deterministic_flag(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    set_seed(999, deterministic=False)
