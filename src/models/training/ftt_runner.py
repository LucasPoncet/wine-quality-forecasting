import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.models.builders.ftt_builder import make_ftt_model
from src.models.components.scope import ScopeClassifier
from src.models.data.wine_data_module import DatasetLoader
from src.models.training.trainer_tabular import TrainerClassifier
from src.models.training.utils_data import clean_tensor_nan, ensure_cat_tensor
from src.preprocessing.feature_engineering import add_engineered_features
from src.visualization.plots.plot_metrics import plot_fold_mean


def run_ftt_pipeline(
    train_path: Path,
    test_path: Path,
    feature_ids: Sequence[str],
    device: torch.device,
    max_epoch: int = 300,
    plot: bool = True,
) -> tuple[float, float]:
    """
    Train an FT-Transformer classifier on tabular data with validation split.
    Returns (accuracy, f1).
    """

    logging.info("=== Loading parquet data ===")
    loader = DatasetLoader(
        train_path=str(train_path),
        test_path=str(test_path),
        target_col="label",
        valid_frac=0.2,
        dtype=torch.float32,
    )
    train_ds, valid_ds, test_ds, onehot_mapping, _ = loader.load_tabular_data()

    # Add engineered features
    (train_ds, valid_ds, test_ds), num_cols, cat_cols = add_engineered_features(
        datasets=(train_ds, valid_ds, test_ds),
        num_cols=loader.num_cols or [],
        cat_cols=loader.cat_cols or [],
        feature_ids=list(feature_ids),
    )

    # Clean tensors
    for ds in (train_ds, valid_ds, test_ds):
        ensure_cat_tensor(ds)
        clean_tensor_nan(ds)

    x_num_train, x_cat_train, y_train = train_ds.tensors
    x_num_valid, x_cat_valid, y_valid = valid_ds.tensors
    x_num_test, x_cat_test, y_test = test_ds.tensors

    # --- Hyperparameters ---
    hyperparams = {
        "d_model": 128,
        "n_layers": 6,
        "n_heads": 8,
        "activation": "gelu",
        "dropout_rate": 0.3,
        "output_dim": 2,
        "learning_rate": 3e-4,
        "max_epoch": max_epoch,
        "weight_decay": 5e-4,
    }

    # --- Model & Scope ---
    model, embedding_sizes = make_ftt_model(hyperparams, num_cols, cat_cols, onehot_mapping)
    model = model.to(device)
    scope = ScopeClassifier(model, hyperparams, steps_per_epoch=1)

    # Class weights
    cnt = Counter(y_train.cpu().numpy())
    total = len(y_train)
    weights = [total / cnt[c] for c in sorted(cnt.keys())]
    scope.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    # --- Training ---
    trainer = TrainerClassifier(hyperparameters=hyperparams)
    trainer.set_model(model=model, device=device)
    trainer.set_scope(scope=scope)
    trainer.set_data(
        x_train=(x_num_train, x_cat_train),
        y_train=y_train,
        x_valid=(x_num_valid, x_cat_valid),
        y_valid=y_valid,
    )

    train_hist, valid_hist = trainer.run()

    # --- Evaluation ---
    with torch.no_grad():
        preds = model(x_num_test.to(device), x_cat_test.to(device)).argmax(dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    logging.info(f"FTTransformer Test: acc={acc:.4f}, f1={f1:.4f}")
    print(classification_report(y_true, preds))

    if plot:
        plot_fold_mean(
            np.array(train_hist),
            np.array(valid_hist),
            np.ones_like(train_hist),
            "FTTransformer – Accuracy",
        )

    return float(acc), float(f1)
