"""Loss functions for class-imbalanced defect classification.

Why a custom focal loss? Defect datasets are imbalanced by nature — "good" parts
vastly outnumber any single defect type, and some defects are rare. The commonly
copy-pasted focal-loss snippet uses a *scalar* ``alpha`` (often ``alpha=1``), which
rescales every class by the same constant and therefore does **nothing** for
imbalance — only the ``(1 - pt) ** gamma`` modulating term helps there. The real
lever for imbalance is a *per-class* ``alpha`` weight vector. This implementation
supports both and is numerically stable (works on raw logits via log-softmax).

Reference: Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet), 2017.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

AlphaLike = Union[None, float, int, Sequence[float], torch.Tensor]


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    Args:
        alpha: imbalance weighting.
            * ``None``  -> no class weighting (pure focal).
            * scalar    -> uniform rescale (does NOT address imbalance; kept for parity).
            * sequence/tensor of length ``C`` -> per-class weights (the real lever;
              e.g. inverse-frequency weights from the training-set class counts).
        gamma: focusing parameter. ``0`` recovers (weighted) cross-entropy; higher
            values down-weight easy, well-classified examples more aggressively.
        reduction: ``"mean"`` | ``"sum"`` | ``"none"``.

    Shapes:
        inputs:  ``(N, C)`` raw logits.
        targets: ``(N,)`` int64 class indices in ``[0, C)``.
    """

    def __init__(
        self,
        alpha: AlphaLike = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")
        self.gamma = float(gamma)
        self.reduction = reduction

        self.alpha_scalar: Optional[float] = None
        alpha_vec: Optional[torch.Tensor] = None
        if isinstance(alpha, (int, float)):
            self.alpha_scalar = float(alpha)
        elif isinstance(alpha, (list, tuple)):
            alpha_vec = torch.as_tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            alpha_vec = alpha.detach().clone().float()
        elif alpha is not None:
            raise TypeError(f"unsupported alpha type: {type(alpha)!r}")
        # registered as a buffer so it follows the module across .to(device);
        # registering None is valid and simply means "no per-class weighting".
        self.register_buffer("alpha", alpha_vec)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"expected inputs of shape (N, C), got {tuple(inputs.shape)}")
        targets = targets.long()
        log_probs = torch.log_softmax(inputs, dim=1)
        # log-prob and prob of the *true* class, per sample
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt  # base focal (alpha == 1)

        if self.alpha is not None:
            if self.alpha.numel() != inputs.size(1):
                raise ValueError(
                    f"alpha has {self.alpha.numel()} entries but there are "
                    f"{inputs.size(1)} classes"
                )
            loss = self.alpha.gather(0, targets) * loss
        elif self.alpha_scalar is not None:
            loss = self.alpha_scalar * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
