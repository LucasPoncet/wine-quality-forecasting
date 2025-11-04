# ------------------------------------------------------------
#  COMPLETE TRAINING SCRIPT  –  binary wine-quality classifier
# ------------------------------------------------------------
from collections import Counter

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import torch
from ClassesData.WineDataModule import DatasetLoader
from ClassesML.Scope import ScopeClassifier
from ClassesML.TabularMLP import TabularMLP
from ClassesML.TrainerTabular import TrainerClassifier
from Embedding import build_cat_mapping
from feature_Engineering import add_engineered_features
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. Load parquet splits via DatasetLoader -------------------
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
cat_cols = ["station", "region", "cepages"]

loader = DatasetLoader(
    train_path=train_path,
    test_path=test_path,
    target_col="label",
    num_cols=num_cols,
    cat_cols=cat_cols,
    valid_frac=0.2,
    dtype=torch.float32,
)

train_ds, valid_ds, test_ds, onehot_mapping, _ = loader.load_tabular_data()

# Add engineered features to the datasets
feature_ids = ["B", "A", "D"]  # diurnal-range, heat/rain, etc.

(train_ds, valid_ds, test_ds), num_cols, cat_cols = add_engineered_features(
    datasets=(train_ds, valid_ds, test_ds),
    num_cols=num_cols,
    cat_cols=cat_cols,
    feature_ids=feature_ids,
)
if cat_cols:
    mapping, x_cat_train, vocab_sizes = build_cat_mapping(
        {col: train_ds.tensors[1][:, i].cpu().numpy() for i, col in enumerate(cat_cols)}, cat_cols
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

    embedding_sizes = {
        col: (len(mapping[col]), int(max(4, np.sqrt(vocab) // 2)))
        for col, vocab in zip(cat_cols, vocab_sizes, strict=False)
    }
else:
    embedding_sizes = {}


def ensure_cat_tensor(ds: TensorDataset):
    x_num, x_cat, y = ds.tensors
    if x_cat.numel() == 0:  # (N,0) ou (0,)
        x_cat = torch.empty((len(x_num), 0), dtype=torch.long)
    ds.tensors = (x_num, x_cat, y)


for ds in (train_ds, valid_ds, test_ds):
    ensure_cat_tensor(ds)

# ---------- 2. Clean numerical data (nan / inf) ------------------------
for ds in (train_ds, valid_ds):
    x_num = torch.nan_to_num(ds.tensors[0], nan=0.0, posinf=0.0, neginf=0.0)
    ds.tensors = (x_num, *ds.tensors[1:])

# ---------- 3. Prepare data splits -------------------------------------
x_num_train, x_cat_train, y_train = train_ds.tensors
x_num_valid, x_cat_valid, y_valid = valid_ds.tensors

print("Label counts:", np.bincount(y_train.numpy()))


# ---------- 4. Optional shape check ------------------------------------
train_loader = DataLoader(
    TensorDataset(x_num_train, x_cat_train, y_train), batch_size=512, shuffle=True
)
batch = next(iter(train_loader))
print("Minibatch shapes →  x_num:", batch[0].shape, "x_cat:", batch[1].shape, "y:", batch[2].shape)

# ---------- 5. Model / hyper-parameters -------------------------------
n_classes = 2
print("Detected n_classes =", n_classes)
hyperparameters = {
    "hidden_layers_size": [128, 64],
    "activation": "relu",
    "batch_normalization": False,
    "dropout_rate": 0.1,
    "output_dim": n_classes,
    "num_numeric_features": len(num_cols),
    "learning_rate": 0.0001,
    "max_epoch": 1500,
}


model = TabularMLP(hyperparameters, embedding_sizes).to(device)
scope = ScopeClassifier(model, hyperparameters, steps_per_epoch=len(train_loader))

# ---------- 6. Balanced loss -------------------------------------------
cnt = Counter(y_train.cpu().numpy())
total = len(y_train)
weights = [total / cnt[cls] for cls in [0, 1]]
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
scope.criterion = criterion

# ---------- 7. Inference on validation & test ---------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
x_num_test, x_cat_test, y_test = test_ds.tensors
probs_test_folds = []
best_thrs = []
global_best_acc = -1.0
global_best_w = None
max_epochs = hyperparameters["max_epoch"]
train_hist_all = np.zeros(max_epochs)
valid_hist_all = np.zeros(max_epochs)
fold_counts = np.zeros(max_epochs)
y_np = y_train.cpu().numpy()
for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y_np), y_np)):
    print(f"\n── Fold {fold + 1}/5 ──")

    x_num_tr, x_cat_tr, y_tr = x_num_train[tr_idx], x_cat_train[tr_idx], y_train[tr_idx]
    x_num_va, x_cat_va, y_va = x_num_train[va_idx], x_cat_train[va_idx], y_train[va_idx]

    trainer = TrainerClassifier(hyperparameter=hyperparameters)
    trainer.set_model(model=TabularMLP(hyperparameters, embedding_sizes).to(device), device=device)
    scope = ScopeClassifier(trainer.model, hyperparameters, steps_per_epoch=len(tr_idx) // 512 + 1)
    scope.criterion = criterion
    trainer.set_scope(scope)

    trainer.set_data(
        x_train=(x_num_tr, x_cat_tr), y_train=y_tr, x_valid=(x_num_va, x_cat_va), y_valid=y_va
    )
    train_acc_hist, valid_acc_hist = trainer.run()
    fold_path = f"models/mlp_fold{fold + 1}.pth"
    # torch.save(trainer.model.state_dict(), fold_path)
    # print("   modèle sauvegardé ", fold_path)
    L = len(train_acc_hist)
    train_hist_all[:L] += np.array(train_acc_hist)
    valid_hist_all[:L] += np.array(valid_acc_hist)
    fold_counts[:L] += 1

    with torch.no_grad():
        p_va = (
            torch.softmax(trainer.model(x_num_va.to(device), x_cat_va.to(device)), 1)[:, 1]
            .cpu()
            .numpy()
        )
    pr, rc, th = precision_recall_curve(y_va.cpu().numpy(), p_va)
    rc_, pr_ = rc[:-1], pr[:-1]

    mask = rc_ >= 0.75
    if mask.any():
        idx_best = np.argmax(pr_[mask])
        best_thr = th[mask][idx_best]
    else:
        best_thr = 0.5

    best_thrs.append(best_thr)

    with torch.no_grad():
        p_te = (
            torch.softmax(trainer.model(x_num_test.to(device), x_cat_test.to(device)), 1)[:, 1]
            .cpu()
            .numpy()
        )
    probs_test_folds.append(p_te)

mask = fold_counts > 0
train_mean = train_hist_all[mask] / fold_counts[mask]
valid_mean = valid_hist_all[mask] / fold_counts[mask]

plt.figure(figsize=(6, 3))
plt.plot(train_mean, label="train (mean)")
plt.plot(valid_mean, label="valid (mean)")
plt.xlabel("epoch")
plt.ylabel("accuracy (%)")
plt.title("Accuracy – moyenne 5 folds")
plt.legend()
plt.tight_layout()
plt.show()


# ---------- 8. Agrégation des folds ------------------------------------

best_thr_global = float(np.median(best_thrs))
probs_test_mean = np.mean(probs_test_folds, axis=0)
y_pred = (probs_test_mean >= best_thr_global).astype(int)

print(f"\nSeuil médian CV : {best_thr_global:.3f}")
print(classification_report(y_test.cpu().numpy(), y_pred))

# torch.save(global_best_w, "models/mlp_best_valid.pth")
# print(f" Modèle meilleur fold sauvegardé  → models/mlp_best_valid.pth "
#       f"(valid {global_best_acc:.2f}%)")
# ---------- 10. LightGBM baseline --------------------------------------

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

X_train = torch.cat([x_num_train, x_1hot_train], dim=1).cpu().numpy()
X_valid = torch.cat([x_num_valid, x_1hot_valid], dim=1).cpu().numpy()
y_train_np, y_valid_np = y_train.cpu().numpy(), y_valid.cpu().numpy()

lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05)
lgbm.fit(X_train, y_train_np)


importances = lgbm.booster_.feature_importance(importance_type="gain")
feat_names = num_cols + cat_cols
top = sorted(
    zip(feat_names, importances[: len(feat_names)], strict=False), key=lambda x: x[1], reverse=True
)[:15]
print("Top gains:")
for name, gain in top:
    print(f"{name:25s} {gain:,.0f}")

lgbm_pred = lgbm.predict(X_valid)
lgbm_pred = np.asarray(lgbm_pred)
lgbm_acc = accuracy_score(y_valid_np, lgbm_pred)
print("LGBM valid acc:", lgbm_acc)


if cat_cols:
    onehots_te = []
    for idx, col in enumerate(cat_cols):
        n_cls = embedding_sizes[col][0]
        onehots_te.append(torch.nn.functional.one_hot(x_cat_test[:, idx], n_cls).float())
    x_1hot_test = torch.cat(onehots_te, 1)
else:
    x_1hot_test = torch.empty((len(x_num_test), 0))

x_lgbm_test = torch.cat([x_num_test, x_1hot_test], 1).cpu().numpy()
lgbm_test_preds = np.asarray(lgbm.predict(x_lgbm_test))
y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

lgbm_test_acc = accuracy_score(y_test_np, lgbm_test_preds)
lgbm_test_f1 = f1_score(y_test_np, lgbm_test_preds)

print(f"\n LGBM Test Accuracy: {lgbm_test_acc:.4f}")
print(f" LGBM Test F1 Score:  {lgbm_test_f1:.4f}\n")
print(classification_report(y_test_np, lgbm_test_preds))
