import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import torch
from ClassesData.WineDataModule import DatasetLoader
from ClassesML.Losses import FocalLoss
from ClassesML.Scope import ScopeClassifier
from ClassesML.TabNetEncoder import TabNetClassifier
from ClassesML.TrainerTabular import TrainerClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 1. Load parquet splits via DatasetLoader --------------------
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
    valid_frac=0.20,
    dtype=torch.float32,
)

train_ds, valid_ds, test_ds, onehot, _ = loader.load_tabular_data()

# ---------- 2. Clean numerical NaN/Inf ---------------------------------
for ds in (train_ds, valid_ds):
    x_num = torch.nan_to_num(ds.tensors[0], nan=0.0, posinf=0.0, neginf=0.0)
    ds.tensors = (x_num, *ds.tensors[1:])

x_num_train, x_cat_train, y_train = train_ds.tensors
x_num_valid, x_cat_valid, y_valid = valid_ds.tensors

print("Label counts train:", np.bincount(y_train.numpy()))

# ---------- 3. Embedding sizes -----------------------------------------
num_regions = len(onehot["region"])
num_stations = len(onehot["station"])
num_cepages = len(onehot["cepages"])

embedding_sizes = {
    "region": (num_regions, 8),
    "station": (num_stations, 16),
    "cepages": (num_cepages, 8),
}

# ---------- 4. DataLoader (for steps_per_epoch) ------------------------
train_loader = DataLoader(
    TensorDataset(x_num_train, x_cat_train, y_train),
    batch_size=512,
    shuffle=True,
)
print("Minibatch sample:", next(iter(train_loader))[0].shape)

# ---------- 5. Hyperparameter TabNet ---------------------------------
n_classes = 2
hyper = {
    "learning_rate": 1.5e-3,
    "max_epoch": 90,  # early-stop on precision
    # network
    "n_steps": 8,
    "n_d": 64,
    "n_a": 64,
    "shared_layers": 1,
    "step_layers": 2,
    "gamma": 1.8,
    "lambda_sparse": 1e-4,
    "virtual_batch": 32,
    "emb_dropout": 0.2,
}

model = TabNetClassifier(
    embedding_sizes=embedding_sizes,
    num_numeric_features=len(num_cols),
    output_dim=n_classes,
    n_steps=hyper["n_steps"],
    shared_layers=hyper["shared_layers"],
    step_layers=hyper["step_layers"],
    emb_dropout=hyper["emb_dropout"],
    virtual_batch_size=hyper["virtual_batch"],
).to(device)

# ---------- 6. Scope + scheduler + loss --------------------------------
scope = ScopeClassifier(model, hyper, steps_per_epoch=len(train_loader))

criterion = FocalLoss(alpha=0.8, gamma=2.0).to(device)
scope.criterion = criterion

# ---------- 7. Trainer --------------------------------------------------
trainer = TrainerClassifier(hyperparameter=hyper)
trainer.set_model(model, device)
trainer.set_scope(scope)
trainer.set_data(
    x_train=(x_num_train, x_cat_train),
    y_train=y_train,
    x_valid=(x_num_valid, x_cat_valid),
    y_valid=y_valid,
)

train_acc, valid_acc = trainer.run()

# ---------- 8. Accuracy curves --------------------------------------
plt.plot(train_acc, label="Train")
plt.plot(valid_acc, label="Valid")
plt.title("TabNet – accuracy vs epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# ---------- 9. Validation quick check ----------------------------------
THRESH = 0.5
model.eval()
with torch.no_grad():
    logits = model(x_num_valid.to(device), x_cat_valid.to(device))
    probs = torch.softmax(logits, dim=1)[:, 1]  # proba classe 1
    pred_val = (probs > THRESH).long().cpu().numpy()


def pick_threshold(y_true, prob_cls1, target_prec=0.85):
    p, r, t = precision_recall_curve(y_true, prob_cls1)
    p, r = p[1:], r[1:]  # align with t
    idx = np.where(p >= target_prec)[0]
    return t[idx[0]] if idx.size else t[np.argmax(p)]


best_thresh = pick_threshold(y_valid.numpy(), probs.cpu().numpy(), target_prec=0.85)
print(f"Best threshold for precision {best_thresh:.2f} (target 0.85)")
print(f"Seuil={THRESH:.2f}")
print(classification_report(y_valid.numpy(), pred_val))


# ---------- 10. LightGBM baseline ----------------------------
def onehot_batch(x_cat, dims):
    return torch.cat(
        [
            torch.nn.functional.one_hot(x_cat[:, i], num_classes=d).float()
            for i, d in enumerate(dims)
        ],
        dim=1,
    )


x1h_train = onehot_batch(x_cat_train, [num_regions, num_stations, num_cepages])
x1h_valid = onehot_batch(x_cat_valid, [num_regions, num_stations, num_cepages])

X_train = torch.cat([x_num_train, x1h_train], 1).cpu().numpy()
X_valid = torch.cat([x_num_valid, x1h_valid], 1).cpu().numpy()

lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05)
lgbm.fit(X_train, y_train.numpy())
lgbm_pred = lgbm.predict(X_valid)
lgbm_pred = np.asarray(lgbm_pred)
print("LGBM valid acc:", accuracy_score(y_valid.numpy(), lgbm_pred))

# ---------- 11. Eval on test set ------------------------------
x_num_test, x_cat_test, y_test = test_ds.tensors
with torch.no_grad():
    test_logits = model(x_num_test.to(device), x_cat_test.to(device))
    probs_test = torch.softmax(test_logits, 1)[:, 1]
    test_pred = (probs_test > THRESH).long().cpu().numpy()

print("\nTabNet Test Metrics")
print(classification_report(y_test.numpy(), test_pred))

# Save the model
torch.save(model.state_dict(), "models/tabnet_model.pth")
print("Model saved as 'tabnet_model.pth'")


# ---------- 12. Analyse of Masks------------------------------------
def collect_global_masks(model, loader, device):
    model.eval()
    all_masks = []
    with torch.no_grad():
        for xb_num, xb_cat, _ in loader:
            _, masks = model(xb_num.to(device), xb_cat.to(device), return_masks=True)
            batch_mask = torch.stack(masks).mean(0)
            all_masks.append(batch_mask.cpu())
    return torch.cat(all_masks).mean(0)


global_mask = collect_global_masks(model, train_loader, device)

feature_names = (
    num_cols
    + [f"region_emb_{i}" for i in range(embedding_sizes["region"][1])]
    + [f"station_emb_{i}" for i in range(embedding_sizes["station"][1])]
    + [f"cepage_emb_{i}" for i in range(embedding_sizes["cepages"][1])]
)

# top-20
topk = torch.topk(global_mask, k=20)
print("\nTop-20 features selon TabNet (mask moyen) :")
for idx, val in zip(topk.indices, topk.values, strict=False):
    print(f"{feature_names[idx]:25s}  {val:.3f}")
