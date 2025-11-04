from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import parallel_backend
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

# --------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------- #
NUM_FEATURES: list[str] = [
    "GDD",
    "TM_summer",
    "TX_summer",
    "temp_amp_summer",
    "hot_days",
    "rainy_days_summer",
    "rain_June",
    "rain_SepOct",
    "frost_days_Apr",
    "avg_TM_Apr",
    "price",
]
CAT_FEATURES: list[str] = ["cepages", "region"]
TARGET = "label"  # Binary target: 1 for top 65% wines, 0 otherwise
RANDOM_STATE = 64  # For reproducibility


def _categorical_encoder(kind: str):
    if kind == "linear":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def _make_preprocessor(model_key: str) -> ColumnTransformer:
    if model_key == "lr":
        transformers = [
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
                ),
                NUM_FEATURES,
            ),
            ("cat", _categorical_encoder("linear"), CAT_FEATURES),
        ]
    elif model_key == "xgb":
        transformers = [
            ("num", SimpleImputer(strategy="mean"), NUM_FEATURES),
            ("cat", _categorical_encoder("tree"), CAT_FEATURES),
        ]
    else:
        transformers = [
            ("num", SimpleImputer(strategy="mean"), NUM_FEATURES),
            ("cat", _categorical_encoder("tree"), CAT_FEATURES),
        ]
    return ColumnTransformer(transformers)


def _make_estimator(model_key: str):
    if model_key == "lr":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_key == "xgb":
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=None,
        l2_regularization=0.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _param_grid(model_key: str) -> dict[str, list]:
    if model_key == "lr":
        return {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
        }
    if model_key == "rf":
        return {
            "model__n_estimators": [100, 300, 600],
            "model__max_depth": [None, 5, 10, 20],
            "model__max_features": ["sqrt"],
            "model__min_samples_split": [2, 10, 20],
            "model__min_samples_leaf": [1, 4],
            "model__bootstrap": [True],
        }
    if model_key == "xgb":
        return {
            "model__n_estimators": [80, 100, 120],  # around 100
            "model__max_depth": [2, 3, 4],  # around 3
            "model__learning_rate": [0.03, 0.05, 0.07],  # around 0.05
            "model__subsample": [0.7, 0.8, 0.9],  # around 0.8
            "model__colsample_bytree": [0.7, 0.8, 0.9],  # around 0.8
            "model__gamma": [0, 0.05, 0.1],  # around 0
            "model__reg_alpha": [0.05, 0.1, 0.2],  # around 0.1
            "model__reg_lambda": [0.8, 1, 1.2],  # around 1
        }
    return {
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
        "model__max_leaf_nodes": [31, 63],
        "model__l2_regularization": [0.0, 0.1],
    }


def build_pipeline(model_key: str, grid_search: bool, search_type: str = "grid", n_iter: int = 50):
    pipe = Pipeline(
        [
            ("pre", _make_preprocessor(model_key)),
            ("model", _make_estimator(model_key)),
        ]
    )

    if not grid_search:
        return pipe
    param_grid = _param_grid(model_key)

    if search_type == "random":
        return RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=RANDOM_STATE,
        )

    else:
        return GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )


def _evaluate(
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    y_proba: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_proba))
    return acc, f1, auc


def _print_confusion(y_true: NDArray[np.int_], y_pred: NDArray[np.int_]) -> None:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion matrix:\n[[TN {tn}  FP {fp}]\n [FN {fn}  TP {tp}]]")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser("baseline models")
    p.add_argument("--train", type=Path, required=True, help="Train parquet file path")
    p.add_argument("--test", type=Path, required=True, help="Test parquet file path")
    p.add_argument(
        "--model",
        choices=["xgb", "lr", "rf", "hgb", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    p.add_argument("--grid_search", action="store_true", help="Enable GridSearchCV")
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    args = _parse_args(argv)

    df_train = pd.read_parquet(args.train)
    df_test = pd.read_parquet(args.test)

    X_train = df_train.drop(columns=[TARGET])
    y_train: NDArray[np.int_] = df_train[TARGET].astype(int).to_numpy()
    X_test = df_test.drop(columns=[TARGET])
    y_test: NDArray[np.int_] = df_test[TARGET].astype(int).to_numpy()

    model_keys = ["xgb", "lr", "rf", "hgb", "all"] if args.model == "all" else [args.model]

    for key in tqdm(model_keys, desc="Models"):
        print(f"\n=== Training {key.upper()} ===")
        clf = build_pipeline(key, args.grid_search, search_type="random", n_iter=200)
        with parallel_backend("loky"):
            clf.fit(X_train, y_train)

        if isinstance(clf, RandomizedSearchCV):
            print("Best params:", clf.best_params_)
            best_est = clf.best_estimator_
        else:
            best_est = clf

        y_pred = np.asarray(clf.predict(X_test)).astype(int)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = clf.decision_function(X_test)
        y_proba = np.asarray(y_proba, dtype=np.float64)

        acc, f1, auc = _evaluate(y_test, y_pred, y_proba)
        print(f"Accuracy : {acc:.4f}")
        print(f"F1 score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        _print_confusion(y_test, y_pred)

        # Print feature importances if available
        model = best_est.named_steps["model"]
        pre = best_est.named_steps["pre"]
        if hasattr(model, "feature_importances_"):
            # Get feature names after preprocessing
            try:
                feature_names = (
                    pre.get_feature_names_out()
                    if hasattr(pre, "get_feature_names_out")
                    else [f"f{i}" for i in range(len(model.feature_importances_))]
                )
            except Exception:
                feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print("Top 10 Feature Importances:")
            for idx in sorted_idx[:10]:
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
        elif hasattr(model, "coef_"):
            print("Feature coefficients (Logistic Regression):")
            print(model.coef_)


if __name__ == "__main__":
    # Hardcoded file paths for direct execution
    TRAIN_PATH = "data/vivino_wine_train_label.parquet"
    TEST_PATH = "data/vivino_wine_test_label.parquet"
    MODEL = "all"  # or "lr", "rf", "hgb"
    GRID_SEARCH = True  # Set to False to disable grid search

    class Args:
        train = Path(TRAIN_PATH)
        test = Path(TEST_PATH)
        model = MODEL
        grid_search = GRID_SEARCH

    # Simulate argparse.Namespace
    args = Args()

    df_train = pd.read_parquet(args.train)
    df_test = pd.read_parquet(args.test)

    X_train = df_train.drop(columns=[TARGET])
    y_train: NDArray[np.int_] = df_train[TARGET].astype(int).to_numpy()
    X_test = df_test.drop(columns=[TARGET])
    y_test: NDArray[np.int_] = df_test[TARGET].astype(int).to_numpy()

    model_keys = ["xgb", "lr", "rf", "hgb", "all"] if args.model == "all" else [args.model]

    for key in tqdm(model_keys, desc="Models"):
        print(f"\n=== Training {key.upper()} ===")
        clf = build_pipeline(key, args.grid_search, search_type="random", n_iter=200)
        with parallel_backend("loky"):
            clf.fit(X_train, y_train)

        if isinstance(clf, GridSearchCV | RandomizedSearchCV):
            print("Best params:", clf.best_params_)
            best_est = clf.best_estimator_
        else:
            best_est = clf

        y_pred = np.asarray(clf.predict(X_test)).astype(int)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = clf.decision_function(X_test)
        y_proba = np.asarray(y_proba, dtype=np.float64)

        acc, f1, auc = _evaluate(y_test, y_pred, y_proba)
        print(f"Accuracy : {acc:.4f}")
        print(f"F1 score : {f1:.4f}")
        print(f"ROC-AUC  : {auc:.4f}")
        _print_confusion(y_test, y_pred)
