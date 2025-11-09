import numpy as np
import torch
from src.models.components.scope import ScopeClassifier
from ClassesML.TrainerTabular import TrainerClassifier
from sklearn.model_selection import StratifiedKFold


def cross_val_run(x_num, x_cat, y, build_model_fn, hyper, n_splits=5, seed=100):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_acc = []
    y_np = y.cpu().numpy()
    X_dummy = np.zeros((len(y_np), 1))
    for fold, (idx_tr, idx_va) in enumerate(skf.split(X_dummy, y_np)):
        print(f"\n── Fold {fold + 1}/{n_splits} ──")

        model = build_model_fn()
        bs = hyper["batch_size"]
        scope = ScopeClassifier(model, hyper, steps_per_epoch=len(idx_tr) // bs)

        trainer = TrainerClassifier(hyper)
        trainer.set_model(model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        trainer.set_scope(scope)

        trainer.set_data(
            (x_num[idx_tr], x_cat[idx_tr]), y[idx_tr], (x_num[idx_va], x_cat[idx_va]), y[idx_va]
        )

        _, val_hist = trainer.run()
        fold_acc.append(max(val_hist))

    print(f"\nMean CV accuracy: {sum(fold_acc) / n_splits:.2f}%")
    return sum(fold_acc) / n_splits
