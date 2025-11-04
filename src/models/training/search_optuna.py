# search_optuna.py  — run this file
# ---------------------------------------------------------------
import optuna
import torch
from ClassesData.WineDataModule import DatasetLoader
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerTabular import TrainerClassifier
from cv_wrapper import cross_val_run
from model_factory import build_model
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ---------------- 1. LOAD DATA ------------------------------------------------
train_path = "data/vivino_wine_train_label.parquet"
test_path = "data/vivino_wine_test_label.parquet"

num_cols = [
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
cat_cols = ["region", "station", "cepages"]

loader = DatasetLoader(
    train_path=train_path,
    test_path=test_path,
    target_col="label",
    num_cols=num_cols,
    cat_cols=cat_cols,
    valid_frac=0.0,
    dtype=torch.float32,
)
train_ds, _, test_ds, onehot_map, _ = loader.load_tabular_data()

x_num, x_cat, y = train_ds.tensors
y = y.long()

num_regions = len(onehot_map["region"])
num_stations = len(onehot_map["station"])
num_cepages = len(onehot_map["cepages"])

embedding_sizes = {
    "region": (num_regions, 8),
    "station": (num_stations, 16),
    "cepages": (num_cepages, 8),
}
n_classes = int(y.max().item() + 1)

# ---------------- 2. BASELINE HYPER-PARAMS ------------------------------------
base_hp = {
    "hidden_layers_size": [128, 64],
    "activation": "relu",
    "batch_normalization": False,
    "dropout_rate": 0.1,
    "output_dim": n_classes,
    "num_numeric_features": x_num.shape[1],
    "learning_rate": 1e-4,
    "max_epoch": 1500,
}


# ---------------- 3. OPTUNA OBJECTIVE ----------------------------------------
def objective(trial):
    hp_arch = {
        "dropout_rate": trial.suggest_float("drop", 0.0, 0.4),
        "hidden_layers_size": trial.suggest_categorical(
            "layers", [[256, 128], [128, 64], [256, 256, 128]]
        ),
    }
    hp_scope = base_hp.copy()
    hp_scope["learning_rate"] = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    hp_scope["batch_size"] = trial.suggest_categorical("bs", [128, 256, 512])
    hp_scope["early_stopping"] = True
    hp_scope["min_epoch_es"] = 100
    hp_scope["patience_es"] = 15

    if trial.suggest_categorical("sched", ["plateau", "onecycle"]) == "plateau":
        hp_scope["patience_lr"] = 5
    else:
        hp_scope["cycle_lr"] = 3e-3

    return cross_val_run(
        x_num,
        x_cat,
        y,
        build_model_fn=lambda: build_model({**base_hp, **hp_arch}, embedding_sizes),
        hyper=hp_scope,
        n_splits=5,
        seed=trial.number,
    )


# ---------------- 4. RUN OPTUNA ----------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, timeout=3 * 60 * 60)

print("🔎  Best params :", study.best_trial.params)
print("🔎  Best 5-fold accuracy:", study.best_value)

# =============================================================================
# 5. RETRAIN ON FULL TRAIN SET WITH BEST PARAMS  +  TEST-SET REPORT
# =============================================================================
best_arch = base_hp.copy()
best_arch.update(
    {
        "hidden_layers_size": study.best_trial.params.get("layers", base_hp["hidden_layers_size"]),
        "dropout_rate": study.best_trial.params.get("drop", base_hp["dropout_rate"]),
    }
)
best_scope = base_hp.copy()
best_scope.update(
    {
        "learning_rate": study.best_trial.params["lr"],
        "batch_size": study.best_trial.params["bs"],
        "early_stopping": True,
    }
)
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
trainer.set_data(
    x_train=(x_num, x_cat),
    y_train=y,
    x_valid=(x_num, x_cat),
    y_valid=y,  # dummy – only for early stop
)
trainer.run()

# ---------------- 6. EVALUATE ON HELD-OUT TEST -------------------------------
x_num_test, x_cat_test, y_test = test_ds.tensors
full_model.eval()
with torch.no_grad():
    logits = full_model(x_num_test.to(device), x_cat_test.to(device))
    y_pred = logits.argmax(dim=1).cpu()

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊  FINAL TEST METRICS")
print(f"   • Accuracy : {acc:.4f}")
print(f"   • F1 score : {f1:.4f}\n")
print("Full classification report:")
print(classification_report(y_test, y_pred))

# Save the model
torch.save(full_model.state_dict(), "models/mlp_optuna.pth")
print("Model saved to 'models/mlp_optuna.pth'")
