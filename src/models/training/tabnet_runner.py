import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch import nn

from src.models.architectures.tabnet import TabNetClassifier
from src.models.builders.tabnet_builder import make_tabnet_model
from src.models.components.scope import ScopeClassifier
from src.models.data.wine_data_module import DatasetLoader
from src.models.evaluation.metrics_utils import evaluate_metrics
from src.models.training.trainer_tabular import TrainerClassifier
from src.models.training.utils_data import clean_tensor_nan, ensure_cat_tensor, load_parquet_dataset
from src.preprocessing.feature_engineering import add_engineered_features
from src.visualization.plots.plot_metrics import plot_fold_mean


def run_tabnet_pipeline(
    train_path: Path,
    test_path: Path,
    feature_ids: Sequence[str],
    device: torch.device,
    max_epoch: int = 100,
    plot: bool = True,
) -> tuple[float, float, float]:
    """
    Train a TabNet classifier on tabular data with cross-validation.
    Returns (acc, f1, auc).
    """

    logging.info("=== Loading parquet data ===")
    X_train, y_train, X_test, y_test = load_parquet_dataset(train_path, test_path, "label")

    num_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

    loader = DatasetLoader(
        train_path=str(train_path),
        test_path=str(test_path),
        target_col="label",
        num_cols=num_cols,
        cat_cols=cat_cols,
        valid_frac=0.2,
        dtype=torch.float32,
    )
    train_ds, valid_ds, test_ds, onehot_mapping, _ = loader.load_tabular_data()

    # Add engineered features
    (train_ds, valid_ds, test_ds), num_cols, cat_cols = add_engineered_features(
        datasets=(train_ds, valid_ds, test_ds),
        num_cols=num_cols,
        cat_cols=cat_cols,
        feature_ids=list(feature_ids),
    )

    for ds in (train_ds, valid_ds, test_ds):
        ensure_cat_tensor(ds)
        clean_tensor_nan(ds)

    x_num_train, x_cat_train, y_train_t = train_ds.tensors
    x_num_valid, x_cat_valid, y_valid_t = valid_ds.tensors
    x_num_test, x_cat_test, y_test_t = test_ds.tensors

    # ---------------- Model setup ----------------
    hyperparameters = {
        "learning_rate": 1.5e-3,
        "max_epoch": max_epoch,
        "n_steps": 8,
        "n_d": 64,
        "n_a": 64,
        "shared_layers": 1,
        "step_layers": 2,
        "gamma": 1.8,
        "lambda_sparse": 1e-4,
        "virtual_batch": 32,
        "emb_dropout": 0.2,
    }

    model, embedding_sizes = make_tabnet_model(hyperparameters, num_cols, cat_cols, onehot_mapping)
    scope = ScopeClassifier(model, hyperparameters, steps_per_epoch=1)

    # Class weighting
    cnt = Counter(y_train_t.cpu().numpy())
    total = len(y_train_t)
    weights = [total / cnt[cls] for cls in [0, 1]]
    scope.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    # ---------------- Cross-validation ----------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
    y_np = y_train_t.cpu().numpy()
    probs_test_folds, best_thrs = [], []
    train_hist_all = np.zeros(max_epoch)
    valid_hist_all = np.zeros(max_epoch)
    fold_counts = np.zeros(max_epoch)

    logging.info("=== Starting 5-Fold training ===")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y_np), y_np)):
        logging.info(f"── Fold {fold + 1}/5 ──")

        x_num_tr, x_cat_tr, y_tr = (
            x_num_train[tr_idx],
            x_cat_train[tr_idx],
            y_train_t[tr_idx],
        )
        x_num_va, x_cat_va, y_va = (
            x_num_train[va_idx],
            x_cat_train[va_idx],
            y_train_t[va_idx],
        )

        trainer = TrainerClassifier(hyperparameters=hyperparameters)
        model_typed = cast(
            nn.Module,
            TabNetClassifier(
                embedding_sizes=embedding_sizes,
                num_numeric_features=len(num_cols),
                output_dim=2,
                n_steps=hyperparameters["n_steps"],
                shared_layers=hyperparameters["shared_layers"],
                step_layers=hyperparameters["step_layers"],
                emb_dropout=hyperparameters["emb_dropout"],
                virtual_batch_size=hyperparameters["virtual_batch"],
            ).to(device),
        )
        trainer.set_model(model_typed, device=device)
        assert trainer.model is not None
        trainer.set_scope(ScopeClassifier(trainer.model, hyperparameters, steps_per_epoch=1))
        assert trainer.scope is not None
        trainer.scope.criterion = scope.criterion

        trainer.set_data(
            x_train=(x_num_tr, x_cat_tr),
            y_train=y_tr,
            x_valid=(x_num_va, x_cat_va),
            y_valid=y_va,
        )

        train_acc_hist, valid_acc_hist = trainer.run()
        L = len(train_acc_hist)
        train_hist_all[:L] += np.array(train_acc_hist)
        valid_hist_all[:L] += np.array(valid_acc_hist)
        fold_counts[:L] += 1

        with torch.no_grad():
            p_va = (
                torch.softmax(trainer.model(x_num_va.to(device), x_cat_va.to(device)), 1)[:, 1]
                .cpu()
                .numpy()
            )

        pr, rc, th = precision_recall_curve(y_va.cpu().numpy(), p_va)
        rc_, pr_ = rc[:-1], pr[:-1]
        mask = rc_ >= 0.75
        best_thr = th[mask][np.argmax(pr_[mask])] if mask.any() else 0.5
        best_thrs.append(best_thr)

        with torch.no_grad():
            p_te = (
                torch.softmax(trainer.model(x_num_test.to(device), x_cat_test.to(device)), 1)[:, 1]
                .cpu()
                .numpy()
            )
        probs_test_folds.append(p_te)

    # ---------------- Aggregate results ----------------
    mask = fold_counts > 0

    best_thr_global = float(np.median(best_thrs))
    probs_test_mean = np.mean(probs_test_folds, axis=0)
    y_pred = (probs_test_mean >= best_thr_global).astype(int)

    acc, f1, auc = evaluate_metrics(y_test_t.cpu().numpy(), y_pred, probs_test_mean)
    logging.info(f"TabNet Test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    print(classification_report(y_test_t.cpu().numpy(), y_pred))

    if plot:
        plot_fold_mean(
            train_hist_all, valid_hist_all, fold_counts, "TabNet – mean 5 folds accuracy"
        )

    return acc, f1, auc
