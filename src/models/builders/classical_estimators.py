from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_STATE = 100


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
