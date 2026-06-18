"""Arrange the NEU-CLS steel-surface dataset into the ImageFolder layout the
classification pipeline expects, and (optionally) build a second copy with an
*induced* class imbalance in the training set.

NEU-CLS is naturally balanced (300 images x 6 classes). Real QC data is not — rare
defects are the whole problem — so to test the focal-loss / class-weighting code on
real images we cap a few classes in TRAIN only, while keeping val/test balanced so the
evaluation stays fair. The imbalance is induced (clearly labelled), not a property of NEU.

Usage:
    python scripts/prepare_neu.py --src data/raw/NEU-CLS
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import random
import re
import shutil

IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png"}


def class_of(stem: str) -> str:
    m = re.match(r"^([A-Za-z][A-Za-z_\- ]*?)[_\- ]*\d+$", stem)
    name = m.group(1) if m else stem
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def copy_many(files, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dst_dir / f.name)


def main():
    ap = argparse.ArgumentParser(description="Prepare NEU-CLS into ImageFolder splits")
    ap.add_argument("--src", default="data/raw/NEU-CLS")
    ap.add_argument("--out-balanced", default="data/neu")
    ap.add_argument("--out-imb", default="data/neu_imb")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--rare", default="crazing,pitted_surface", help="classes to make rare in TRAIN")
    ap.add_argument("--rare-cap", type=int, default=30, help="train images kept for each rare class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    imgs = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
    by_class: dict[str, list[pathlib.Path]] = collections.defaultdict(list)
    for p in imgs:
        by_class[class_of(p.stem)].append(p)
    rare = {c.strip() for c in args.rare.split(",") if c.strip()}

    bal = pathlib.Path(args.out_balanced)
    imb = pathlib.Path(args.out_imb)
    for root in (bal, imb):
        if root.exists():
            shutil.rmtree(root)

    summary_bal, summary_imb = {}, {}
    for cls, files in sorted(by_class.items()):
        files = sorted(files)
        random.Random(args.seed).shuffle(files)
        n = len(files)
        n_test = int(round(n * args.test))
        n_val = int(round(n * args.val))
        test_files = files[:n_test]
        val_files = files[n_test : n_test + n_val]
        train_files = files[n_test + n_val :]

        # balanced: full train + val + test
        copy_many(train_files, bal / "train" / cls)
        copy_many(val_files, bal / "val" / cls)
        copy_many(test_files, bal / "test" / cls)
        summary_bal[cls] = (len(train_files), len(val_files), len(test_files))

        # imbalanced: same val/test; train capped for rare classes
        imb_train = train_files[: args.rare_cap] if cls in rare else train_files
        copy_many(imb_train, imb / "train" / cls)
        copy_many(val_files, imb / "val" / cls)
        copy_many(test_files, imb / "test" / cls)
        summary_imb[cls] = (len(imb_train), len(val_files), len(test_files))

    def show(title, summ):
        print(f"\n{title}")
        print(f"  {'class':16s} train  val  test")
        for cls, (tr, va, te) in summ.items():
            print(f"  {cls:16s} {tr:5d} {va:4d} {te:4d}")
        tr_counts = [v[0] for v in summ.values()]
        print(f"  train imbalance ratio = {max(tr_counts) / max(1, min(tr_counts)):.1f} : 1")

    show(f"BALANCED  -> {bal}", summary_bal)
    show(f"INDUCED-IMBALANCE (train only; rare={sorted(rare)} capped to {args.rare_cap})  -> {imb}", summary_imb)


if __name__ == "__main__":
    main()
