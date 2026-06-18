"""Evaluation: per-class precision/recall/F1, confusion matrix, and the
QC-specific framing that a single accuracy number hides.

In quality control a *missed defect* (false negative) and a *false alarm* (false
positive) have very different costs — a missed defect ships; a false alarm wastes an
inspector's time. So we collapse to good-vs-defect and report the miss rate (FN) and
false-alarm rate (FP) separately, in addition to the per-class report.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

from qc.data import make_loaders
from qc.model import build_model
from qc.utils import get_device


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["num_classes"], ck["arch"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, ck


@torch.no_grad()
def gather_preds(model, loader, device):
    ys, ps = [], []
    for x, y in loader:
        probs = model(x.to(device)).softmax(1).cpu()
        ps.append(probs.argmax(1))
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def qc_good_vs_defect(y, p, class_names):
    if "good" not in class_names:
        return None
    gi = class_names.index("good")
    true_defect = y != gi
    pred_defect = p != gi
    tp = int((true_defect & pred_defect).sum())
    fn = int((true_defect & ~pred_defect).sum())
    fp = int((~true_defect & pred_defect).sum())
    tn = int((~true_defect & ~pred_defect).sum())
    return {
        "defect_recall": tp / max(1, tp + fn),
        "miss_rate_FN": fn / max(1, tp + fn),
        "false_alarm_rate_FP": fp / max(1, fp + tn),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate a defect classifier checkpoint")
    ap.add_argument("--ckpt", default="outputs/clf/best.pt")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default=None, help="defaults to <ckpt dir>/eval")
    args = ap.parse_args()

    device = get_device()
    model, ck = load_model(args.ckpt, device)
    class_names = ck["class_names"]
    img_size = ck.get("img_size", 96)
    out = pathlib.Path(args.out or (pathlib.Path(args.ckpt).parent / "eval"))
    out.mkdir(parents=True, exist_ok=True)

    loaders, _, _ = make_loaders(args.data, img_size, 64, 0)
    y, p = gather_preds(model, loaders[args.split], device)

    report = classification_report(y, p, target_names=class_names, output_dict=True, zero_division=0)
    report_txt = classification_report(y, p, target_names=class_names, zero_division=0)
    bal_acc = balanced_accuracy_score(y, p)
    qc = qc_good_vs_defect(y, p, class_names)

    print(f"== {args.ckpt} on {args.split} ==")
    print(report_txt)
    print(f"macro_f1={report['macro avg']['f1-score']:.4f}  balanced_accuracy={bal_acc:.4f}")
    if qc:
        print(
            f"QC good-vs-defect: defect_recall={qc['defect_recall']:.3f}  "
            f"miss_rate(FN)={qc['miss_rate_FN']:.3f}  false_alarm(FP)={qc['false_alarm_rate_FP']:.3f}"
        )

    cm = confusion_matrix(y, p)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"Confusion matrix ({args.split})")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close()

    (out / "report.json").write_text(
        json.dumps(
            {
                "ckpt": str(args.ckpt),
                "split": args.split,
                "macro_f1": report["macro avg"]["f1-score"],
                "balanced_accuracy": bal_acc,
                "qc_good_vs_defect": qc,
                "per_class": report,
            },
            indent=2,
        )
    )
    (out / "classification_report.txt").write_text(report_txt + f"\nbalanced_accuracy={bal_acc:.4f}\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
