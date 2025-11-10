# pyright: reportAttributeAccessIssue=false

import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.models.builders.classical_estimators import _param_grid
from src.models.training.classical_pipeline import RANDOM_STATE, build_pipeline


@pytest.mark.parametrize("model_key", ["lr", "rf", "xgb", "hgb"])
def test_build_pipeline_no_search_returns_pipeline(model_key):
    """Ensure base pipeline structure without GridSearch."""
    pipe = build_pipeline(model_key, grid_search=False)
    assert isinstance(pipe, Pipeline)
    steps = dict(pipe.named_steps)
    assert "pre" in steps and "model" in steps
    # Ensure both subcomponents exist
    assert steps["pre"] is not None
    assert steps["model"] is not None


@pytest.mark.parametrize("model_key", ["lr", "rf", "xgb", "hgb"])
def test_build_pipeline_with_grid_search(model_key):
    """Ensure grid_search=True returns GridSearchCV by default."""
    grid = build_pipeline(model_key, grid_search=True)
    assert isinstance(grid, GridSearchCV)
    assert isinstance(grid.estimator, Pipeline)
    # Each GridSearchCV should have a parameter grid prefixed with model__
    assert all(k.startswith("model__") for k in grid.param_grid)
    assert grid.cv == 5
    assert grid.scoring == "roc_auc"
    assert grid.n_jobs == -1


@pytest.mark.parametrize("model_key", ["lr", "rf", "xgb", "hgb"])
def test_build_pipeline_with_random_search(model_key):
    """Ensure random search uses RandomizedSearchCV with correct params."""
    search = build_pipeline(model_key, grid_search=True, search_type="random", n_iter=25)
    assert isinstance(search, RandomizedSearchCV)
    assert isinstance(search.estimator, Pipeline)
    assert search.n_iter == 25
    assert search.random_state == RANDOM_STATE
    # The parameter distributions must match _param_grid
    expected_grid = _param_grid(model_key)
    assert set(expected_grid.keys()) == set(search.param_distributions.keys())


def test_invalid_search_type_defaults_to_grid():
    """Passing an invalid search_type still yields a GridSearchCV object."""
    search = build_pipeline("lr", grid_search=True, search_type="invalid_type")
    assert isinstance(search, GridSearchCV)
    # Should fall back to default grid param behavior
    assert all(k.startswith("model__") for k in search.param_grid)


def test_random_and_grid_search_are_distinct_objects():
    """RandomizedSearchCV and GridSearchCV are separate configurations."""
    grid_obj = build_pipeline("rf", grid_search=True, search_type="grid")
    rand_obj = build_pipeline("rf", grid_search=True, search_type="random")
    assert isinstance(grid_obj, GridSearchCV)
    assert isinstance(rand_obj, RandomizedSearchCV)
    # Ensure random search uses n_iter attr while grid does not
    assert hasattr(rand_obj, "n_iter")
    assert not hasattr(grid_obj, "n_iter")
