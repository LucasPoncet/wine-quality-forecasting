from types import SimpleNamespace


def test_search_optuna_smoke(monkeypatch):
    """Smoke test: ensure script can import and run its entry logic."""
    import importlib

    # --- Mock heavy dependencies ---
    import src.models.training.cross_validation as cv

    monkeypatch.setattr(cv, "cross_val_run", lambda *_, **__: 42.0)

    import src.models.builders.model_factory as factory

    monkeypatch.setattr(factory, "build_model", lambda *_: SimpleNamespace())

    import src.models.components.scope as scope

    monkeypatch.setattr(scope, "ScopeClassifier", lambda *_, **__: SimpleNamespace())

    import src.models.training.trainer_tabular as trainer

    monkeypatch.setattr(
        trainer,
        "TrainerClassifier",
        lambda *_: SimpleNamespace(
            set_model=lambda *_, **__: None,
            set_scope=lambda *_, **__: None,
            set_data=lambda *_, **__: None,
            run=lambda *_, **__: None,
        ),
    )

    # --- Import and reload your script ---
    module = importlib.import_module("scripts.search_optuna")

    # --- Sanity check ---
    assert hasattr(module, "objective")
    assert callable(module.objective)
