"""Dataset loading, transforms, and a data audit.

Data is the real bottleneck in defect QC — defects are rare and labelling is
expensive, so class imbalance is the norm. ``audit_dataset`` makes that imbalance
visible (it's the first thing to look at, before any modelling).

Augmentation note: for defects, augmentation must NOT destroy the defect signal.
A thin scratch or hairline crack can be cropped or blurred away, so we keep the
train transforms mild (flips, small rotation, modest brightness/contrast) and use
a plain Resize rather than RandomResizedCrop.
"""

from __future__ import annotations

import pathlib
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_loaders(
    data_root: str,
    img_size: int = 96,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], List[str], List[int]]:
    """Return {train,val,test} loaders, class names, and train-set class counts.

    Expects an ImageFolder layout: ``<root>/<split>/<class>/*.png``.
    """
    root = pathlib.Path(data_root)
    datasets = {
        split: ImageFolder(root / split, transform=build_transforms(img_size, split == "train"))
        for split in ("train", "val", "test")
    }
    class_names = datasets["train"].classes
    counts = Counter(label for _, label in datasets["train"].samples)
    train_counts = [counts.get(i, 0) for i in range(len(class_names))]
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, ds in datasets.items()
    }
    return loaders, class_names, train_counts


def audit_dataset(data_root: str) -> pd.DataFrame:
    """Count images per class per split; record the imbalance ratio in ``df.attrs``."""
    root = pathlib.Path(data_root)
    rows: Dict[str, Dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n = sum(1 for _ in cls_dir.glob("*.png"))
            rows.setdefault(cls_dir.name, {})[split] = n
    df = pd.DataFrame(rows).T.fillna(0).astype(int).reindex(columns=["train", "val", "test"], fill_value=0)
    df = df.reindex(sorted(df.index))
    if "train" in df.columns and df["train"].sum() > 0:
        tr = df["train"]
        df.attrs["imbalance_ratio"] = float(tr.max() / max(1, tr[tr > 0].min()))
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Audit a defect dataset (class balance).")
    ap.add_argument("--data", default="data/sample")
    args = ap.parse_args()
    df = audit_dataset(args.data)
    print(df.to_string())
    if "imbalance_ratio" in df.attrs:
        print(f"\nmajority:minority (train) imbalance ratio = {df.attrs['imbalance_ratio']:.1f} : 1")
        print("-> a non-trivial ratio is why we weight the loss (see qc/losses.py, qc/utils.py).")
