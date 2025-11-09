import logging
from typing import Any, cast

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TrainerClassifier:
    """Supervised classifier trainer for tabular deep learning models."""

    def __init__(self, hyperparameters: dict[str, Any]) -> None:
        self.hp = hyperparameters
        self.model: nn.Module | None = None
        self.device: torch.device | None = None
        self.scope: Any | None = None
        self.train_loader: DataLoader | None = None
        self.valid_loader: DataLoader | None = None

    def set_model(self, model, device):
        """Attach model and device."""
        self.model, self.device = model, device
        logger.info("Model set on device: %s", device)

    def set_scope(self, scope):
        """Attach training scope (optimizer, scheduler, early stopping)."""
        self.scope = scope

    def set_data(
        self,
        x_train: torch.Tensor | tuple[torch.Tensor, ...],
        y_train: torch.Tensor,
        x_valid: torch.Tensor | tuple[torch.Tensor, ...],
        y_valid: torch.Tensor,
    ) -> None:
        """Initialize DataLoaders for training and validation datasets."""
        batch_size = self.hp.get("batch_size", 512)

        if isinstance(x_train, (tuple, list)):
            train_tensors = (*x_train, y_train)
            valid_tensors = (*x_valid, y_valid)
        else:
            train_tensors = (x_train, y_train)
            valid_tensors = (x_valid, y_valid)

        self.train_loader = DataLoader(
            TensorDataset(*cast(tuple[torch.Tensor, ...], train_tensors)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        self.valid_loader = DataLoader(
            TensorDataset(*cast(tuple[torch.Tensor, ...], valid_tensors)),
            batch_size=batch_size * 2,
            shuffle=False,
        )
        logger.info("DataLoaders initialized: batch_size=%d", batch_size)

    def _epoch_loop(self, loader: DataLoader, train_phase: bool = True) -> tuple[float, float]:
        assert self.model is not None and self.device is not None
        assert self.scope is not None, "Scope must be set before training."
        phase = "train" if train_phase else "valid"
        self.model.train(mode=train_phase)

        total_loss, total_correct, total_samples = 0.0, 0, 0

        context = torch.enable_grad() if train_phase else torch.no_grad()
        with context:
            for batch in loader:
                # support (x_num, x_cat, y)
                if len(batch) == 3:
                    x_num, x_cat, y = batch
                    x_num, x_cat, y = (
                        x_num.to(self.device),
                        x_cat.to(self.device),
                        y.to(self.device),
                    )
                    y_hat = self.model(x_num, x_cat)
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)

                loss = self.scope.criterion(y_hat, y)

                if train_phase:
                    self.scope.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.scope.optimizer.step()
                    if isinstance(self.scope.scheduler, OneCycleLR):
                        self.scope.scheduler.step()

                total_loss += loss.item() * y.size(0)
                preds = y_hat.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100.0
        logger.debug("[%s] loss=%.4f, acc=%.2f%%", phase, avg_loss, accuracy)
        return avg_loss, accuracy

    def run(self):
        """Execute full training loop."""
        if self.scope is None:
            raise RuntimeError("Scope must be set before training.")
        if self.train_loader is None or self.valid_loader is None:
            raise RuntimeError("Train and validation DataLoaders must be set before training.")

        if not all([self.model, self.scope, self.train_loader, self.valid_loader]):
            raise RuntimeError("Model, scope, and data loaders must be set before training.")
        logger.info("Starting training loop...")
        train_hist, valid_hist = [], []
        n_epochs = self.hp["max_epoch"]

        for epoch in range(1, n_epochs + 1):
            tr_loss, tr_acc = self._epoch_loop(self.train_loader, train_phase=True)
            va_loss, va_acc = self._epoch_loop(self.valid_loader, train_phase=False)

            train_hist.append(tr_acc)
            valid_hist.append(va_acc)

            logger.info(
                "Epoch %2d/%d | train loss %.4f acc %.2f%% | valid loss %.4f acc %.2f%%",
                epoch,
                n_epochs,
                tr_loss,
                tr_acc,
                va_loss,
                va_acc,
            )

            if isinstance(self.scope.scheduler, ReduceLROnPlateau):
                self.scope.scheduler.step(va_acc)
            elif self.scope.scheduler is not None:
                self.scope.scheduler.step()

            if getattr(self.scope, "early_stopping", None) and not self.scope.early_stopping.step(
                self.model, epoch, va_acc
            ):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        best_epoch = int(torch.tensor(valid_hist).argmax()) + 1
        logger.info("Best validation accuracy %.2f%% at epoch %d", max(valid_hist), best_epoch)

        return train_hist, valid_hist
