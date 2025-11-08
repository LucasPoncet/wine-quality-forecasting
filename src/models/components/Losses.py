import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Compute Focal Loss for classification tasks.

    Parameters
    ----------
    alpha : float, default=0.8
        Weighting factor for balancing positive/negative classes.
        (Binary version: [1.0, alpha])
    gamma : float, default=2.0
        Focusing parameter (>0) that reduces loss for well-classified examples.
    reduction : {"mean", "sum", "none"}, default="mean"
        Specifies reduction applied to the final loss.

    Notes
    -----
    For multi-class tasks, alpha can be a scalar or a list of per-class weights.
    """

    def __init__(
        self, alpha: float | list[float] = 0.8, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1.0, float(alpha)])
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, target):
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor
            Raw model outputs (before softmax), shape (N, C).
        target : Tensor
            Ground truth class indices, shape (N,).

        Returns
        -------
        Tensor
            Scalar loss (if reduction ≠ "none") or per-sample loss.
        """
        if target.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"target must be integer tensor, got {target.dtype}")
        if logits.ndim != 2:
            raise ValueError(f"logits must be 2D (N, C), got shape {logits.shape}")
        ce_loss = self.ce(logits, target)
        pt = torch.exp(-ce_loss)
        at = self.alpha.to(logits.device)[target]
        loss = at * (1.0 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")
