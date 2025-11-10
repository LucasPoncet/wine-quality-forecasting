from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.models.builders.classical_estimators import _make_estimator, _param_grid
from src.models.builders.preprocessors import _make_preprocessor

RANDOM_STATE = 100


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
