from __future__ import annotations

import torch.nn as nn


class Utilities:
    @staticmethod
    def get_activation(name: str | None) -> nn.Module | None:
        name = (name or "linear").lower()
        match name:
            case "relu":
                return nn.ReLU()
            case "sigmoid":
                return nn.Sigmoid()
            case "tanh":
                return nn.Tanh()
            case "linear" | _:
                return None

    @staticmethod
    def compute_accuracy(y_true, y_logits, count=False):
        preds = y_logits.argmax(dim=1)
        correct = (preds == y_true).sum().item()
        return correct if count else 100.0 * correct / y_true.size(0)
