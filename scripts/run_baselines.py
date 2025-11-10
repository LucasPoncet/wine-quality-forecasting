import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from src.models.evaluation.metrics_utils import evaluate_metrics, print_confusion
from src.models.training.classical_pipeline import build_pipeline


def run_baselines(
    train_path: Path, test_path: Path, model_keys: list[str], grid_search: bool = True
):
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"].astype(int).to_numpy()
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"].astype(int).to_numpy()

    for key in tqdm(model_keys, desc="Models"):
        logging.info(f"=== Training {key.upper()} ===")
        clf = build_pipeline(key, grid_search=grid_search, search_type="random", n_iter=200)

        with parallel_backend("loky"):
            clf.fit(X_train, y_train)

        if isinstance(clf, (GridSearchCV, RandomizedSearchCV)):
            best_est: Pipeline = clf.best_estimator_  # type: ignore[attr-defined]
        else:
            best_est: Pipeline = clf  # type: ignore[assignment]
        y_pred = clf.predict(X_test)
        y_proba = (
            clf.predict_proba(X_test)[:, 1]
            if hasattr(clf, "predict_proba")
            else clf.decision_function(X_test)
        )

        y_pred = np.asarray(y_pred).astype(int)
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]

        y_proba = np.asarray(y_proba, dtype=np.float64)
        if isinstance(y_proba, tuple):
            y_proba = y_proba[0]

        acc, f1, auc = evaluate_metrics(y_test, y_pred, y_proba)
        logging.info(f"{key.upper()} — acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")
        print_confusion(y_test, y_pred)

        model = best_est.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            logging.info("Top 10 feature importances:")
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:10]
            for i in top_idx:
                logging.info(f"  f{i}: {importances[i]:.3f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train classical baseline models.")
    parser.add_argument("--train", type=Path, default="data/vivino_wine_train_label.parquet")
    parser.add_argument("--test", type=Path, default="data/vivino_wine_test_label.parquet")
    parser.add_argument("--models", nargs="+", default=["xgb", "rf", "lr", "hgb"])
    parser.add_argument("--no-grid", action="store_true", help="Disable hyperparameter search")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run_baselines(args.train, args.test, args.models, grid_search=not args.no_grid)


if __name__ == "__main__":
    main()
