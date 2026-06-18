"""PaDiM-style anomaly detection: train on GOOD parts only, flag deviations.

This is the approach real QC often prefers over supervised detection: defects are rare,
varied, and open-set, so instead of enumerating every defect you characterise what
"normal" looks like and measure how far a part deviates from it. We take features from a
pretrained CNN, fit a per-position multivariate Gaussian over good images, and score test
images by Mahalanobis distance (per-position -> anomaly map; image score = max).

Reference: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
Detection and Localization", 2020.

Demonstrated here on the synthetic 'good' vs defect split (fast, offline). For the real
benchmark, point --data at an MVTec AD category laid out as <root>/train/good and
<root>/test/<good|defect>/.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from qc.data import IMAGENET_MEAN, IMAGENET_STD
from qc.utils import get_device, seed_everything


class PaDiM:
    def __init__(self, backbone="resnet18", n_features=100, img_size=256, device=None, seed=42):
        self.device = device or get_device()
        self.img_size = img_size
        self.n_features = n_features
        self.seed = seed
        self.backbone = timm.create_model(
            backbone, pretrained=True, features_only=True, out_indices=(1, 2, 3)
        ).eval().to(self.device)
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self._idx = None
        self.mean = None
        self.cov_inv = None
        self.H = self.W = None

    def _load(self, paths):
        return torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths])

    @torch.no_grad()
    def _embed(self, x):
        feats = self.backbone(x.to(self.device))
        base = feats[-1].shape[-2:]  # smallest map -> fewer positions, lighter covariance
        feats = [F.interpolate(f, size=base, mode="bilinear", align_corners=False) for f in feats]
        emb = torch.cat(feats, dim=1)  # (N, C, H, W)
        if self._idx is None:
            g = torch.Generator().manual_seed(self.seed)
            self._idx = torch.randperm(emb.shape[1], generator=g)[: self.n_features].to(self.device)
        return emb[:, self._idx]  # (N, d, H, W)

    @torch.no_grad()
    def fit(self, good_paths, batch=16):
        chunks = [self._embed(self._load(good_paths[i:i + batch])).cpu()
                  for i in range(0, len(good_paths), batch)]
        emb = torch.cat(chunks)                       # (N, d, H, W)
        N, d, H, W = emb.shape
        self.H, self.W = H, W
        emb = emb.permute(2, 3, 1, 0).reshape(H * W, d, N)  # (HW, d, N)
        self.mean = emb.mean(-1)                      # (HW, d)
        identity = torch.eye(d)
        cov = torch.empty(H * W, d, d)
        for p in range(H * W):
            centred = emb[p] - self.mean[p:p + 1].T   # (d, N)
            cov[p] = (centred @ centred.T) / (N - 1) + 0.01 * identity  # shrinkage -> invertible
        self.cov_inv = torch.linalg.inv(cov)
        return self

    @torch.no_grad()
    def anomaly_map(self, path):
        emb = self._embed(self._load([path])).cpu()[0]   # (d, H, W)
        d, H, W = emb.shape
        e = emb.reshape(d, H * W).T                       # (HW, d)
        diff = e - self.mean
        dist = torch.einsum("pi,pij,pj->p", diff, self.cov_inv, diff).clamp(min=0).sqrt()
        amap = dist.reshape(1, 1, H, W)
        amap = F.interpolate(amap, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return amap[0, 0].numpy()

    def score(self, path):
        return float(self.anomaly_map(path).max())


def main():
    ap = argparse.ArgumentParser(description="PaDiM anomaly detection (train on good, flag deviations)")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--good-class", default="good")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-features", type=int, default=100)
    ap.add_argument("--out", default="outputs/anomaly")
    args = ap.parse_args()

    seed_everything(42)
    root = pathlib.Path(args.data)
    good_train = [str(p) for p in sorted((root / "train" / args.good_class).glob("*")) if p.is_file()]
    print(f"fitting PaDiM on {len(good_train)} good images ...")
    padim = PaDiM(n_features=args.n_features, img_size=args.img_size).fit(good_train)

    paths, labels = [], []
    for cls_dir in sorted((root / "test").iterdir()):
        if cls_dir.is_dir():
            for p in cls_dir.glob("*"):
                paths.append(str(p))
                labels.append(0 if cls_dir.name == args.good_class else 1)
    scores = [padim.score(p) for p in paths]
    auroc = roc_auc_score(labels, scores)
    print(f"image-level AUROC (good vs defect): {auroc:.4f}  on {len(paths)} test images "
          f"({sum(labels)} defect / {len(labels) - sum(labels)} good)")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    worst = max((i for i, l in enumerate(labels) if l == 1), key=lambda i: scores[i])
    amap = padim.anomaly_map(paths[worst])
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    from qc.explain import overlay

    overlay(Image.open(paths[worst]).convert("RGB"), amap).save(out / "anomaly_overlay.png")
    print(f"saved anomaly heatmap for {pathlib.Path(paths[worst]).name} -> {out / 'anomaly_overlay.png'}")


if __name__ == "__main__":
    main()
