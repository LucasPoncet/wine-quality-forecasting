from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from ClassesData.WineDataModule import DatasetLoader
from ClassesML.FTT import FTTransformer
from ClassesML.Scope import ScopeClassifier
from ClassesML.TrainerTabular import TrainerClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    valid_frac=0.2,
    dtype=torch.float32,
)

train_ds, valid_ds, test_ds, onehot_mapping, _ = loader.load_tabular_data()


for ds in (train_ds, valid_ds):
    x_num = torch.nan_to_num(ds.tensors[0], nan=0.0, posinf=0.0, neginf=0.0)
    ds.tensors = (x_num, *ds.tensors[1:])


x_num_train, x_cat_train, y_train = train_ds.tensors
x_num_valid, x_cat_valid, y_valid = valid_ds.tensors

print("Label counts:", np.bincount(y_train.numpy()))


num_regions = len(onehot_mapping["region"])
num_stations = len(onehot_mapping["station"])
num_cepages = len(onehot_mapping["cepages"])


train_loader = DataLoader(
    TensorDataset(x_num_train, x_cat_train, y_train), batch_size=512, shuffle=True
)
batch = next(iter(train_loader))
print("Minibatch shapes →  x_num:", batch[0].shape, "x_cat:", batch[1].shape, "y:", batch[2].shape)


n_classes = 2

hyperparameters = {
    "d_model": 128,
    "n_layers": 6,
    "n_heads": 8,
    "activation": "gelu",
    "dropout_rate": 0.3,
    "output_dim": n_classes,
    "learning_rate": 3e-4,
    "max_epoch": 300,
    "weight_decay": 5e-4,
}


embedding_sizes = {
    "region": (num_regions, None),
    "station": (num_stations, None),
    "cepages": (num_cepages, None),
}
model = FTTransformer(
    hyper=hyperparameters, embedding_sizes=embedding_sizes, num_numeric_features=len(num_cols)
).to(device)

scope = ScopeClassifier(model, hyperparameters, steps_per_epoch=len(train_loader))


cnt = Counter(y_train.cpu().numpy())
total = len(y_train)
weights = [total / cnt[cls] for cls in [0, 1]]
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
scope.criterion = criterion


trainer = TrainerClassifier(hyperparameter=hyperparameters)
trainer.set_model(model=model, device=device)
trainer.set_scope(scope=scope)
trainer.set_data(
    x_train=(x_num_train, x_cat_train),
    y_train=y_train,
    x_valid=(x_num_valid, x_cat_valid),
    y_valid=y_valid,
)

train_acc_hist, valid_acc_hist = trainer.run()


plt.figure()
plt.plot(train_acc_hist, label="Train accuracy")
plt.plot(valid_acc_hist, label="Valid accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()


with torch.no_grad():
    y_hat = model(x_num_valid.to(device), x_cat_valid.to(device))
    pred = y_hat.argmax(dim=1).cpu().numpy()

print("Validation predictions shape:", pred.shape)


x_num_test, x_cat_test, y_test = test_ds.tensors
with torch.no_grad():
    y_test_hat = model(x_num_test.to(device), x_cat_test.to(device))
    test_pred = y_test_hat.argmax(dim=1).cpu().numpy()
if test_pred.ndim > 1:
    test_pred = (test_pred > 0.5).astype(int)

if isinstance(test_pred, torch.Tensor):
    test_pred = test_pred.cpu().numpy()
if isinstance(y_test, torch.Tensor):
    y_test = y_test.cpu().numpy()


test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)

print(f"\n Test Accuracy: {test_acc:.4f}")
print(f" Test F1 Score:  {test_f1:.4f}\n")


print(classification_report(y_test, test_pred))

# Save the model
# torch.save(model.state_dict(), "models/mlp_wine_quality.pth")
# print("Model saved to 'models/mlp_wine_quality.pth'")
