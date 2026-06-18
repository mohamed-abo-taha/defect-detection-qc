"""Generate a small SYNTHETIC defect dataset so the whole pipeline runs end-to-end
with zero downloads.

This exists ONLY to validate the pipeline and to demonstrate the imbalance story.
Synthetic textures are NOT a substitute for real industrial images — swap in a real
dataset for meaningful metrics (see README: MVTec AD, NEU-DET, GC10-DET, Severstal,
or a Roboflow export). Classes are intentionally imbalanced, with 'crack' rare.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["good", "scratch", "dent", "contamination", "crack"]
TRAIN_COUNTS = {"good": 240, "scratch": 80, "dent": 80, "contamination": 60, "crack": 24}
EVAL_FRACTION = 0.25


def _rng(seed):
    return np.random.default_rng(seed)


def _base_texture(rng, size):
    arr = rng.normal(135, 16, (size, size)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").filter(ImageFilter.GaussianBlur(0.8)).convert("RGB")


def _add_scratch(img, rng, size):
    d = ImageDraw.Draw(img)
    x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
    ang = rng.uniform(0, np.pi)
    length = int(rng.integers(size // 2, size))
    x1, y1 = int(x0 + length * np.cos(ang)), int(y0 + length * np.sin(ang))
    shade = int(rng.integers(200, 255)) if rng.random() < 0.5 else int(rng.integers(0, 55))
    d.line([(x0, y0), (x1, y1)], fill=(shade, shade, shade), width=int(rng.integers(1, 3)))


def _add_dent(img, rng, size):
    d = ImageDraw.Draw(img)
    r = int(rng.integers(size // 8, size // 4))
    cx, cy = int(rng.integers(r, size - r)), int(rng.integers(r, size - r))
    for rr, shade in [(r, 95), (int(r * 0.6), 65), (int(r * 0.3), 40)]:
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(shade, shade, shade))


def _add_contamination(img, rng, size):
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(6, 16))):
        r = int(rng.integers(1, 4))
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        shade = int(rng.integers(0, 60))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(shade, shade, shade))


def _add_crack(img, rng, size):
    d = ImageDraw.Draw(img)
    x, y = int(rng.integers(0, size)), 0
    pts = [(x, y)]
    while y < size:
        x = int(np.clip(x + rng.integers(-6, 7), 0, size - 1))
        y += int(rng.integers(4, 10))
        pts.append((x, y))
    d.line(pts, fill=(20, 20, 20), width=1)


_DRAW = {
    "scratch": _add_scratch,
    "dent": _add_dent,
    "contamination": _add_contamination,
    "crack": _add_crack,
}


def _make_image(cls, rng, size):
    img = _base_texture(rng, size)
    if cls in _DRAW:
        _DRAW[cls](img, rng, size)
    return img


def make_dataset(root, size=96, seed=42):
    root = pathlib.Path(root)
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            (root / split / cls).mkdir(parents=True, exist_ok=True)
    counter = 0
    for cls in CLASSES:
        n_train = TRAIN_COUNTS[cls]
        n_eval = max(4, int(n_train * EVAL_FRACTION))
        for split, n in (("train", n_train), ("val", n_eval), ("test", n_eval)):
            for i in range(n):
                counter += 1
                _make_image(cls, _rng(seed + counter), size).save(root / split / cls / f"{cls}_{i:04d}.png")
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a synthetic defect dataset")
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = make_dataset(args.out, args.size, args.seed)
    print(f"wrote synthetic dataset -> {root.resolve()}")
    for split in ("train", "val", "test"):
        total = sum(1 for _ in (root / split).rglob("*.png"))
        print(f"  {split}: {total} images")
    print("NOTE: synthetic data — for pipeline validation only, not real metrics.")
