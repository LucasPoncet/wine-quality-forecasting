import copy
import logging

import numpy as np

from src.utils.config_logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class EarlyStopper:
    """Early stopping callback for model training.

    Parameters
    ----------
    hyperparameters : dict
        Dictionary containing:
        - 'max_epoch': int, total number of epochs planned.
        - 'min_epoch_es': int, minimum epochs before early stopping can trigger.
        - 'patience_es': int, number of epochs to wait after last improvement.

    Attributes
    ----------
    best_model_weights : dict[str, Any] | None
        Deep copy of model weights corresponding to best validation metric.
    metric_history : list[float]
        History of metric values across epochs.
    wait : int
        Number of epochs since last improvement.
    """

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.max_epoch = self.hyperparameters["max_epoch"]
        self.metric_history = []
        self.min_epoch = self.hyperparameters.get("min_epoch_es", 10)
        self.patience = self.hyperparameters.get("patience_es", 10)
        self.best_model_weights = None
        self.wait = 0

    def step(self, model, epoch, metric_value):
        """Evaluate metric and decide whether to continue training.

        Parameters
        ----------
        model : nn.Module
            Model being trained.
        epoch : int
            Current epoch index.
        metric_value : float
            Validation metric for this epoch (higher is better).

        Returns
        -------
        bool
            True if training should continue, False otherwise.
        """
        keep_training = True
        self.metric_history.append(metric_value)
        if epoch == 0:
            self.wait = 0
            logger.debug("Epoch %d: initial weights saved", epoch)
            self._save_weights(model)
        if epoch >= self.min_epoch:
            if metric_value > np.max(self.metric_history[:-1]):
                self.wait = 0
                self._save_weights(model)
                logger.info("Epoch %d: metric improved to %.5f", epoch, metric_value)
            else:
                self.wait += 1
                logger.debug("Epoch %d: no improvement (wait=%d)", epoch, self.wait)
                if self.wait >= self.patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    keep_training = False
                    self._restore_weights(model)

        if epoch == (self.max_epoch - 1):
            logger.info("Max epoch %d reached; restoring best weights", self.max_epoch)
            keep_training = False
            self._restore_weights(model)
        return keep_training

    def _save_weights(self, model):
        self.best_model_weights = copy.deepcopy(model.state_dict())

    def _restore_weights(self, model):
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
