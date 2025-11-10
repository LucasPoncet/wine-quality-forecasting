import logging
from pathlib import Path

import optuna
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.models.builders.model_factory import build_model
from src.models.components.scope import ScopeClassifier
from src.models.data.wine_data_module import DatasetLoader
from src.models.training.cross_validation import cross_val_run
from src.models.training.trainer_tabular import TrainerClassifier
from src.utils.config_logger import setup_logging

# --------------------  GLOBAL DATA & CONFIG  --------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "vivino_wine_train_label.parquet"
TEST_PATH = DATA_DIR / "vivino_wine_test_label.parquet"

NUM_COLS = [
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
CAT_COLS = ["region", "station", "cepages"]


# --------------------  OBJECTIVE FUNCTION  --------------------
def objective(trial):
    """Optuna objective for MLP hyperparameter search."""

    loader = DatasetLoader(
        train_path=str(TRAIN_PATH),
        test_path=str(TEST_PATH),
        target_col="label",
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        valid_frac=0.0,
        dtype=torch.float32,
    )

    train_ds, _, _, mapping, _ = loader.load_tabular_data()
    x_num, x_cat, y = train_ds.tensors
    y = y.long()

    embedding_sizes = {
        "region": (len(mapping["region"]), 8),
        "station": (len(mapping["station"]), 16),
        "cepages": (len(mapping["cepages"]), 8),
    }
    n_classes = int(y.max().item() + 1)

    base_hp = {
        "hidden_layers_size": [128, 64],
        "activation": "relu",
        "batch_normalization": False,
        "dropout_rate": 0.1,
        "output_dim": n_classes,
        "num_numeric_features": x_num.shape[1],
        "learning_rate": 1e-4,
        "max_epoch": 1500,
        "model_type": "mlp",
    }

    # Dynamic trial params
    hp_arch = {
        "dropout_rate": trial.suggest_float("drop", 0.0, 0.4),
        "hidden_layers_size": trial.suggest_categorical(
            "layers", [[256, 128], [128, 64], [256, 256, 128]]
        ),
    }
    hp_scope = base_hp.copy()
    hp_scope["learning_rate"] = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    hp_scope["batch_size"] = trial.suggest_categorical("bs", [128, 256, 512])
    hp_scope.update({"early_stopping": True, "min_epoch_es": 100, "patience_es": 15})

    sched_type = trial.suggest_categorical("sched", ["plateau", "onecycle"])
    if sched_type == "plateau":
        hp_scope["patience_lr"] = 5
    else:
        hp_scope["cycle_lr"] = 3e-3

    return cross_val_run(
        x_num,
        x_cat,
        y,
        build_model_fn=lambda: build_model({**base_hp, **hp_arch}, embedding_sizes),
        hyperparameters=hp_scope,
        n_splits=5,
        seed=trial.number,
    )


# --------------------  MAIN EXECUTION  --------------------
def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, timeout=3 * 60 * 60)

    logger.info("Best params: %s", study.best_trial.params)
    logger.info("Best 5-fold accuracy: %.2f%%", study.best_value)

    # Retrain on full data
    loader = DatasetLoader(
        train_path=str(TRAIN_PATH),
        test_path=str(TEST_PATH),
        target_col="label",
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        valid_frac=0.0,
        dtype=torch.float32,
    )
    train_ds, _, test_ds, mapping, _ = loader.load_tabular_data()
    x_num, x_cat, y = train_ds.tensors
    y = y.long()

    embedding_sizes = {
        "region": (len(mapping["region"]), 8),
        "station": (len(mapping["station"]), 16),
        "cepages": (len(mapping["cepages"]), 8),
    }
    n_classes = int(y.max().item() + 1)

    best_arch = {
        "hidden_layers_size": study.best_trial.params.get("layers", [128, 64]),
        "dropout_rate": study.best_trial.params.get("drop", 0.1),
        "activation": "relu",
        "batch_normalization": False,
        "output_dim": n_classes,
        "num_numeric_features": x_num.shape[1],
        "model_type": "mlp",
    }

    best_scope = {
        "learning_rate": study.best_trial.params["lr"],
        "batch_size": study.best_trial.params["bs"],
        "early_stopping": True,
    }
    if "patience_lr" in study.best_trial.params:
        best_scope["patience_lr"] = study.best_trial.params["patience_lr"]
    else:
        best_scope["cycle_lr"] = 3e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_model = build_model(best_arch, embedding_sizes).to(device)
    full_scope = ScopeClassifier(
        full_model, best_scope, steps_per_epoch=len(y) // best_scope["batch_size"]
    )

    trainer = TrainerClassifier(best_scope)
    trainer.set_model(full_model, device)
    trainer.set_scope(full_scope)
    trainer.set_data((x_num, x_cat), y, (x_num, x_cat), y)
    trainer.run()

    # Test evaluation
    x_num_test, x_cat_test, y_test = test_ds.tensors
    full_model.eval()
    with torch.no_grad():
        logits = full_model(x_num_test.to(device), x_cat_test.to(device))
        y_pred = logits.argmax(dim=1).cpu()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logger.info("Final test metrics — acc: %.4f, f1: %.4f", acc, f1)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

    model_dir = ROOT / "models" / "optuna"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(full_model.state_dict(), model_dir / "mlp_optuna.pth")
    logger.info("Model saved to %s", model_dir / "mlp_optuna.pth")


if __name__ == "__main__":
    main()
