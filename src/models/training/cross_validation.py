import logging

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from src.models.components.scope import ScopeClassifier
from src.models.training.trainer_tabular import TrainerClassifier

logger = logging.getLogger(__name__)


def cross_val_run(
    x_num: torch.Tensor,
    x_cat: torch.Tensor,
    y: torch.Tensor,
    build_model_fn,
    hyperparameters: dict,
    n_splits: int = 5,
    seed: int = 100,
) -> float:
    """Run stratified k-fold cross-validation for a tabular classifier.

    Parameters
    ----------
    x_num, x_cat : torch.Tensor
        Numerical and categorical input tensors.
    y : torch.Tensor
        Target tensor (class labels).
    build_model_fn : callable
        Function returning a new model instance.
    hyperparameters : dict
        Hyperparameters for ScopeClassifier and TrainerClassifier.
    n_splits : int, default=5
        Number of folds.
    seed : int, default=100
        Random seed for StratifiedKFold reproducibility.

    Returns
    -------
    float
        Mean validation accuracy (%) across folds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_accuracies: list[float] = []

    y_np = y.cpu().numpy()
    dummy_X = np.zeros((len(y_np), 1))
    for fold, (idx_tr, idx_va) in enumerate(skf.split(dummy_X, y_np)):
        logger.info("── Fold %d/%d ──", fold, n_splits)

        model = build_model_fn().to(device)
        batch_size = hyperparameters["batch_size"]
        scope = ScopeClassifier(model, hyperparameters, steps_per_epoch=len(idx_tr) // batch_size)

        trainer = TrainerClassifier(hyperparameters)
        trainer.set_model(model, device)
        trainer.set_scope(scope)

        trainer.set_data(
            (x_num[idx_tr], x_cat[idx_tr]), y[idx_tr], (x_num[idx_va], x_cat[idx_va]), y[idx_va]
        )

        _, val_hist = trainer.run()
        best_val_acc = max(val_hist)
        logger.info("Fold %d best validation accuracy: %.2f%%", fold, best_val_acc)
        fold_accuracies.append(best_val_acc)

    mean_acc = float(np.mean(fold_accuracies))
    logger.info("Mean CV accuracy: %.2f%%", mean_acc)
    return mean_acc
