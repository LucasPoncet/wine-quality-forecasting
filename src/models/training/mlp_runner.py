import logging
from collections import Counter
from collections.abc import Mapping, Sequence, Sized
from pathlib import Path
from typing import cast

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import TensorDataset

from src.models.architectures.tabular_mlp import TabularMLP
from src.models.components.embedding import build_cat_mapping
from src.models.components.scope import ScopeClassifier
from src.models.data.wine_data_module import DatasetLoader
from src.models.evaluation.metrics_utils import evaluate_metrics
from src.models.training.trainer_tabular import TrainerClassifier
from src.models.training.utils_data import clean_tensor_nan, load_parquet_dataset
from src.preprocessing.feature_engineering import add_engineered_features


def _ensure_cat_tensor(ds: TensorDataset) -> None:
    """Guarantee categorical tensor shape consistency."""
    x_num, x_cat, y = ds.tensors
    if x_cat.numel() == 0:
        x_cat = torch.empty((len(x_num), 0), dtype=torch.long)
    ds.tensors = (x_num, x_cat, y)


def _make_mlp_model(
    hparams: dict, cat_cols: list[str], mapping: Mapping[str, object] | None = None
):
    """Construct a TabularMLP with appropriate embedding sizes."""
    if not cat_cols or mapping is None:
        embedding_sizes: dict[str, tuple[int, int]] = {}
    else:
        embedding_sizes = {
            col: (
                len(cast(Sized, mapping[col])),
                int(max(4, np.sqrt(len(cast(Sized, mapping[col]))) // 2)),
            )
            for col in cat_cols
            if col in mapping
        }
    model = TabularMLP(hparams, embedding_sizes)
    return model, embedding_sizes


def run_mlp_pipeline(
    train_path: Path,
    test_path: Path,
    feature_ids: Sequence[str],
    device: torch.device,
    max_epoch: int = 1000,
) -> None:
    """Train an MLP classifier on tabular data with CV and LightGBM baseline."""

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

    # Map categorical features to embeddings
    if cat_cols:
        mapping, x_cat_train, vocab_sizes = build_cat_mapping(
            {col: train_ds.tensors[1][:, i].cpu().numpy() for i, col in enumerate(cat_cols)},
            cat_cols,
        )
        _, x_cat_valid, _ = build_cat_mapping(
            {col: valid_ds.tensors[1][:, i].cpu().numpy() for i, col in enumerate(cat_cols)},
            cat_cols,
            mapping,
        )
        _, x_cat_test, _ = build_cat_mapping(
            {col: test_ds.tensors[1][:, i].cpu().numpy() for i, col in enumerate(cat_cols)},
            cat_cols,
            mapping,
        )

        train_ds.tensors = (train_ds.tensors[0], x_cat_train, train_ds.tensors[2])
        valid_ds.tensors = (valid_ds.tensors[0], x_cat_valid, valid_ds.tensors[2])
        test_ds.tensors = (test_ds.tensors[0], x_cat_test, test_ds.tensors[2])
    else:
        mapping = None

    for ds in (train_ds, valid_ds, test_ds):
        _ensure_cat_tensor(ds)
        clean_tensor_nan(ds)

    x_num_train, x_cat_train, y_train_t = train_ds.tensors
    x_num_valid, x_cat_valid, y_valid_t = valid_ds.tensors
    x_num_test, x_cat_test, y_test_t = test_ds.tensors

    # ---------------- Model setup ----------------
    hyperparameters = {
        "hidden_layers_size": [128, 64],
        "activation": "relu",
        "batch_normalization": False,
        "dropout_rate": 0.1,
        "output_dim": 2,
        "num_numeric_features": len(num_cols),
        "learning_rate": 1e-4,
        "max_epoch": max_epoch,
    }

    model, embedding_sizes = _make_mlp_model(hyperparameters, cat_cols, mapping)
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
        model_typed = cast(nn.Module, TabularMLP(hyperparameters, embedding_sizes).to(device))
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

        assert trainer.run is not None
        train_acc_hist, valid_acc_hist = trainer.run()

        L = len(train_acc_hist)
        train_hist_all[:L] += np.array(train_acc_hist)
        valid_hist_all[:L] += np.array(valid_acc_hist)
        fold_counts[:L] += 1

        # Validation inference
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
    train_mean = train_hist_all[mask] / fold_counts[mask]
    valid_mean = valid_hist_all[mask] / fold_counts[mask]

    plt.figure(figsize=(6, 3))
    plt.plot(train_mean, label="train (mean)")
    plt.plot(valid_mean, label="valid (mean)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.title("Accuracy – mean of 5 folds")
    plt.legend()
    plt.tight_layout()
    plt.show()

    best_thr_global = float(np.median(best_thrs))
    probs_test_mean = np.mean(probs_test_folds, axis=0)
    y_pred = (probs_test_mean >= best_thr_global).astype(int)

    acc, f1, auc = evaluate_metrics(y_test_t.cpu().numpy(), y_pred, probs_test_mean)
    logging.info(f"\nMLP Test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    print(classification_report(y_test_t.cpu().numpy(), y_pred))

    # ---------------- LightGBM baseline ----------------
    logging.info("\n=== LightGBM baseline ===")
    if cat_cols:
        onehots_tr, onehots_va = [], []
        for idx, col in enumerate(cat_cols):
            n_cls = embedding_sizes[col][0]
            onehots_tr.append(torch.nn.functional.one_hot(x_cat_train[:, idx], n_cls).float())
            onehots_va.append(torch.nn.functional.one_hot(x_cat_valid[:, idx], n_cls).float())
        x_1hot_train = torch.cat(onehots_tr, 1)
        x_1hot_valid = torch.cat(onehots_va, 1)
    else:
        x_1hot_train = torch.empty((len(x_num_train), 0))
        x_1hot_valid = torch.empty((len(x_num_valid), 0))

    X_train = torch.cat([x_num_train, x_1hot_train], 1).cpu().numpy()
    X_valid = torch.cat([x_num_valid, x_1hot_valid], 1).cpu().numpy()
    y_train_np, y_valid_np = y_train_t.cpu().numpy(), y_valid_t.cpu().numpy()

    lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05)
    lgbm.fit(X_train, y_train_np)

    importances = lgbm.booster_.feature_importance(importance_type="gain")
    feat_names = num_cols + cat_cols
    top = sorted(
        zip(feat_names, importances[: len(feat_names)], strict=False),  # ✅ Ruff fix
        key=lambda x: x[1],
        reverse=True,
    )[:15]

    logging.info("Top LightGBM gains:")
    for name, gain in top:
        logging.info(f"{name:25s} {gain:,.0f}")

    lgbm_pred = np.asarray(lgbm.predict(X_valid))
    lgbm_acc = accuracy_score(y_valid_np, lgbm_pred)
    logging.info(f"LGBM valid acc={lgbm_acc:.4f}")

    lgbm_test_preds = np.asarray(
        lgbm.predict(torch.cat([x_num_test, x_cat_test.float()], 1).cpu().numpy())
    )
    lgbm_test_acc = accuracy_score(y_test_t.cpu().numpy(), lgbm_test_preds)
    lgbm_test_f1 = f1_score(y_test_t.cpu().numpy(), lgbm_test_preds)
    logging.info(f"LGBM test acc={lgbm_test_acc:.4f}, f1={lgbm_test_f1:.4f}")
