import logging
from typing import Any, cast

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR, ReduceLROnPlateau

from src.models.components.early_stopper import EarlyStopper
from src.utils.config_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class ScopeClassifier:
    """Encapsulate optimizer, scheduler, and early stopping for model training.

    Parameters
    ----------
    model : nn.Module
        The model to optimize.
    hyperparameters : dict[str, Any]
        Dictionary defining training configuration, may contain:
          - 'learning_rate' (float)
          - 'weight_decay' (float, optional)
          - 'criterion' (nn.Module, optional)
          - 'patience_lr' (int, optional)
          - 'cycle_lr' (float, optional)
          - 'max_epoch' (int, optional)
          - 'early_stopping' (bool, optional)
    steps_per_epoch : int | None, optional
        Required for OneCycleLR.
    """

    def __init__(
        self,
        model: nn.Module,
        hyperparameters: dict[str, object],
        steps_per_epoch: int | None = None,
    ) -> None:
        self.model = model
        self.hyperparameters = hyperparameters

        # Criterion
        crit_obj = hyperparameters.get("criterion", nn.CrossEntropyLoss())
        self.criterion: nn.Module = (
            crit_obj if isinstance(crit_obj, nn.Module) else nn.CrossEntropyLoss()
        )

        # Optimizer
        lr = float(cast(float, hyperparameters["learning_rate"]))
        weight_decay = float(cast(float, hyperparameters.get("weight_decay", 1e-5)))
        self.optimizer: optim.Optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        logger.debug("Initialized Adam optimizer (lr=%.5f, wd=%.1e)", lr, weight_decay)

        # Scheduler
        self.scheduler: LRScheduler | ReduceLROnPlateau | None = None

        if "patience_lr" in hyperparameters:
            patience = int(cast(int, hyperparameters["patience_lr"]))
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", patience=patience, factor=0.1
            )

            logger.info(
                "Using ReduceLROnPlateau scheduler (patience=%d)", hyperparameters["patience_lr"]
            )
        elif "cycle_lr" in hyperparameters and steps_per_epoch is not None:
            max_lr = float(cast(float, hyperparameters["cycle_lr"]))
            max_epoch = int(cast(int, hyperparameters["max_epoch"]))
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=max_epoch,
            )

            logger.info("Using OneCycleLR scheduler (max_lr=%.5f)", hyperparameters["cycle_lr"])

        # Early Stopping
        self.early_stopping: EarlyStopper | None = None
        if hyperparameters.get("early_stopping", False):
            self.early_stopping = EarlyStopper(hyperparameters)
            logger.info("Initialized EarlyStopper")

    # Utility methods for integration with the training loop

    def step_scheduler(self, metric_value: float | None = None) -> None:
        """Update scheduler state after each epoch."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric_value is None:
                raise ValueError("ReduceLROnPlateau requires validation metric to step.")
            self.scheduler.step(metric_value)
        else:
            self.scheduler.step()

    def maybe_stop(self, epoch: int, val_metric: float) -> bool:
        """Check early stopping condition."""
        if self.early_stopping is None:
            return True
        return self.early_stopping.step(self.model, epoch, val_metric)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step_optimizer(self) -> None:
        self.optimizer.step()

    def state_dict(self) -> dict[str, object]:
        """Return full training control state for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore optimizer/scheduler state from checkpoint."""
        self.optimizer.load_state_dict(cast(dict[str, Any], state["optimizer"]))
        if self.scheduler and state.get("scheduler"):
            sched_state = cast(dict[str, Any], state["scheduler"])
            self.scheduler.load_state_dict(sched_state)
