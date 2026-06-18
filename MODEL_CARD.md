# Model card — cv-quality-control

Honest summary of the models in this repo, what they're for, and where they fail.

## Intended use
- **Assistive** visual inspection for manufacturing QC: flag likely defects and show *where* the
  model looked, so a human can review faster.
- **Not** a pass/fail certificate, a safety device, or a replacement for human inspection.

## Models
| Model | Task | Data | Headline metric |
|---|---|---|---|
| `models/clf_neu.pt` | 6-class steel-defect **classification** (ResNet-18, ImageNet transfer) | NEU-CLS (1,800 imgs) | macro-F1 **0.996** on balanced test |
| `models/yolo_neu.pt` | steel-defect **detection** (YOLOv8n) | NEU-DET (1,800 imgs, boxes) | mAP@50 **0.767**, mAP@50-95 **0.452** |

Imbalance study (classification, induced 7:1 train imbalance): cross-entropy macro-F1 0.9926 →
focal+inverse-√ weights **0.9963**; rare `crazing` recall 0.96 → 1.00. On the *synthetic* hard
benchmark (from scratch) the same change moved macro-F1 0.82 → 0.88. Numbers are reproducible from
the README commands (fixed seed, fixed split).

## Per-class detection (NEU-DET, mAP@50)
patches 0.93 · scratches 0.91 · pitted_surface 0.89 · inclusion 0.87 · rolled-in_scale 0.63 ·
**crazing 0.36** (faint, diffuse — the hardest class; consistent with published baselines).

## Known limitations / failure modes
- **Dataset-bound.** Trained on NEU steel surfaces; it will *not* generalise to other products,
  lighting, or cameras without retraining. Re-measure per deployment.
- **NEU is an easy academic benchmark.** ~99% classification / 0.77 mAP is not what a noisy
  production line looks like.
- **Out-of-distribution inputs** (a non-steel image) get a confident-but-meaningless label — use the
  confidence gate in the demo and route low-confidence cases to a human.
- **`crazing`** is unreliable in both tracks.
- Calibration: see `qc/calibrate.py` (ECE + temperature scaling) — confidence should be calibrated
  before it's used to make the route-to-human decision.

## Ethical / honest-use notes
No "100% accurate", "zero-defect", or "fully autonomous" claims. The ROI model (`qc/roi.py`) is an
assumptions-in / range-out estimate, not a guarantee.
