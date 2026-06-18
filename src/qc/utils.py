"""Small shared helpers: reproducibility, device, class weighting."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and torch (CPU+CUDA) for reproducible runs.

    Note: full determinism on GPU also needs deterministic algorithms and can
    cost speed; we seed the RNGs (enough for reproducible eval numbers) and
    leave cuDNN in its fast, non-deterministic default for training throughput.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights_from_counts(
    counts: Sequence[int], scheme: str = "inverse"
) -> torch.Tensor:
    """Per-class weights for imbalance, suitable as focal-loss ``alpha``.

    ``inverse``      -> w_c = total / (C * count_c)   (full inverse frequency)
    ``inverse_sqrt`` -> sqrt of the above             (gentler; avoids over-weighting
                                                        ultra-rare classes into noise)
    Weights are normalised to mean 1 so the overall loss scale is unchanged.
    """
    c = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1.0)
    total = c.sum()
    n_classes = c.numel()
    w = total / (n_classes * c)
    if scheme == "inverse_sqrt":
        w = w.sqrt()
    elif scheme != "inverse":
        raise ValueError(f"unknown scheme {scheme!r}")
    return w / w.mean()


def bbox_voc_to_yolo(xmin, ymin, xmax, ymax, width, height):
    """Convert a Pascal-VOC box (absolute corners) to YOLO (normalised cx, cy, w, h)."""
    cx = ((xmin + xmax) / 2) / width
    cy = ((ymin + ymax) / 2) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return cx, cy, w, h
