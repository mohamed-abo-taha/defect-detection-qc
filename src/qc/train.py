"""Train the defect classifier — the focal-loss / class-imbalance track.

Run two ways to *earn* the imbalance story with real numbers (not a fabricated
85%->96%):

    python src/qc/train.py --data data/sample --loss ce    --out outputs/clf_ce
    python src/qc/train.py --data data/sample --loss focal  --out outputs/clf_focal

then compare with src/qc/evaluate.py. For real datasets add ``--pretrained`` (the default).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from qc.data import make_loaders
from qc.losses import FocalLoss
from qc.model import build_model
from qc.utils import class_weights_from_counts, get_device, seed_everything


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu()
        ps.append(preds)
        ys.append(y)
    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()
    return f1_score(y, p, average="macro"), accuracy_score(y, p)


def main():
    ap = argparse.ArgumentParser(description="Train defect classifier")
    ap.add_argument("--data", default="data/sample")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["focal", "ce"], default="focal")
    ap.add_argument("--alpha", choices=["none", "inverse", "inverse_sqrt"], default="inverse_sqrt")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/clf")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    loaders, class_names, counts = make_loaders(
        args.data, args.img_size, args.batch_size, args.num_workers
    )
    print(f"device={device} classes={class_names}")
    print(f"train class counts={dict(zip(class_names, counts))}")

    model = build_model(len(class_names), args.arch, args.pretrained).to(device)

    if args.loss == "focal":
        alpha = None if args.alpha == "none" else class_weights_from_counts(counts, args.alpha).to(device)
        criterion = FocalLoss(alpha=alpha, gamma=args.gamma)
        crit_desc = f"focal(gamma={args.gamma}, alpha={args.alpha})"
    else:
        criterion = torch.nn.CrossEntropyLoss()
        crit_desc = "cross_entropy"
    print(f"loss={crit_desc}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for x, y in tqdm(loaders["train"], desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        val_f1, val_acc = run_eval(model, loaders["val"], device)
        train_loss = running / max(1, n)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_macro_f1": val_f1, "val_acc": val_acc}
        )
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}  val_acc={val_acc:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "loss": crit_desc,
                },
                out / "best.pt",
            )

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nbest val_macro_f1={best_f1:.4f}  ->  {out / 'best.pt'}")


if __name__ == "__main__":
    main()
