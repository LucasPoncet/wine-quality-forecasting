import torch
import torch.nn as nn
import torch.optim as optim
from ClassesML.EarlyStopper import EarlyStopper
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


class ScopeClassifier:
    def __init__(self, model, hyperparameters, steps_per_epoch=None):
        self.criterion: torch.nn.Module = hyperparameters.get("criterion", nn.CrossEntropyLoss())
        self.optimizer = optim.Adam(
            model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=1e-5
        )
        if "patience_lr" in hyperparameters:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", patience=hyperparameters["patience_lr"], factor=0.1
            )
        elif "cycle_lr" in hyperparameters:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=hyperparameters["cycle_lr"],
                steps_per_epoch=steps_per_epoch,
                epochs=hyperparameters["max_epoch"],
            )
        else:
            self.scheduler = None

        if "early_stopping" in hyperparameters:
            self.early_stopping = EarlyStopper(hyperparameters=hyperparameters)
        else:
            self.early_stopping = None
