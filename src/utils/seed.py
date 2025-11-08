import logging
import os
import random

import numpy as np
import torch

from src.utils.config_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def set_seed(seed: int | str = 100, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility across Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int | str, default=100
        The seed value to use. If a string is provided, it is hashed to derive an integer.
    deterministic : bool, default=True
        If True, forces deterministic algorithms for PyTorch (slightly slower but reproducible).

    Examples
    --------
    >>> from utils.seed import set_seed
    >>> set_seed(123)
    >>> # or with string
    >>> set_seed("experiment_A")
    """
    if isinstance(seed, str):
        # Simple stable hash → integer seed
        seed_value = abs(hash(seed)) % (2**32)
        logger.debug("Derived integer seed %d from string '%s'", seed_value, seed)
    elif isinstance(seed, int):
        seed_value = seed
    else:
        raise TypeError(f"Seed must be int or str, got {type(seed)}")

    # Core reproducibility setup
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("CUDA seeds and deterministic backend set")

    logger.info("Random seed set to %s", seed_value)
