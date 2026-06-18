"""Confidence calibration: is a "92% confident" prediction right 92% of the time?

In QC, confidence drives the route-to-human decision, so a *miscalibrated* model is unsafe
even at high accuracy. We measure Expected Calibration Error (ECE), draw a reliability
diagram, and apply temperature scaling (a single scalar T fit on a held-out split) to
recalibrate without changing accuracy.

Reference: Guo et al., "On Calibration of Modern Neural Networks", 2017.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from qc.data import make_loaders
from qc.model import build_model
from qc.utils import get_device


@torch.no_grad()
def collect_logits(model, loader, device):
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += (m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()).item()
    return ece


def fit_temperature(logits, labels):
    # Optimise in log-space so T stays positive, with a strong-Wolfe line search — a raw-T
    # LBFGS is unstable for this 1-D fit and can diverge to ~0 (degenerate sharpening).
    log_t = torch.nn.Parameter(torch.zeros(1))  # T = exp(0) = 1 at start
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def reliability_diagram(probs, labels, path, title, n_bins=15):
    conf, pred = probs.max(1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            xs.append(conf[m].mean().item())
            ys.append(correct[m].mean().item())
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    plt.plot(xs, ys, "o-", label="model")
    plt.xlabel("confidence")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Confidence calibration (ECE + temperature scaling)")
    ap.add_argument("--ckpt", default="outputs/clf_focal/best.pt")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--out", default=None, help="defaults to <ckpt dir>/calib")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ck["num_classes"], ck["arch"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    out = pathlib.Path(args.out or (pathlib.Path(args.ckpt).parent / "calib"))
    out.mkdir(parents=True, exist_ok=True)

    loaders, _, _ = make_loaders(args.data, ck.get("img_size", 96), 64, 0)
    val_logits, val_y = collect_logits(model, loaders["val"], device)
    test_logits, test_y = collect_logits(model, loaders["test"], device)

    p_before = test_logits.softmax(1)
    ece_before = expected_calibration_error(p_before, test_y)
    T = fit_temperature(val_logits, val_y)
    p_after = (test_logits / T).softmax(1)
    ece_after = expected_calibration_error(p_after, test_y)

    print(f"temperature T = {T:.3f}")
    print(f"ECE before = {ece_before:.4f}   after temperature scaling = {ece_after:.4f}")
    reliability_diagram(p_before, test_y, out / "reliability_before.png", f"before (ECE={ece_before:.3f})")
    reliability_diagram(p_after, test_y, out / "reliability_after.png", f"after T={T:.2f} (ECE={ece_after:.3f})")
    print(f"saved reliability diagrams -> {out}")


if __name__ == "__main__":
    main()
