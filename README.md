# cv-quality-control

A computer-vision pipeline for **manufacturing defect detection**, built end-to-end and
with an emphasis on the parts that actually decide whether such a system is useful:
honest evaluation, class imbalance, explainability, calibration, anomaly detection, and
edge-ready inference.

It is deliberately **honest about what automated visual inspection can and cannot do**.
It assists human inspectors; it does not replace them, and it makes no "100% accurate /
zero-defect / fully autonomous" claims. Every metric below is measured and reproducible.

---

## Three tracks

| Track | What it does | When to use it | Entry point |
|---|---|---|---|
| **Classification** | Is this part good, and which defect type? | You have per-image labels; want the imbalance/Grad-CAM/ONNX story | `src/qc/train.py` |
| **Detection (YOLOv8)** | *Where* are the defects (boxes)? | You have box annotations (NEU-DET, GC10-DET, Roboflow) | `scripts/yolo_train.py` |
| **Anomaly (PaDiM)** | Flag deviations from "normal", no defect labels | Defects are rare/varied/open-set — the realistic QC case | `src/qc/anomaly.py` |

The classification track is the runnable spine (trains in seconds on the included synthetic
data, no downloads). Detection is the article's headline approach; anomaly detection (train on
good parts only) is what real open-set QC often uses.

---

## Honest results

**The size of the gain from imbalance handling depends entirely on how hard the problem is** —
the nuance the source article skips.

### 1. Synthetic data, ResNet-18 from scratch (hard problem)

> ⚠️ A **synthetic** dataset (`scripts/make_sample.py`) to validate the pipeline and demonstrate
> the imbalance technique. Not real-world performance.

Same data, same seed, same 20 epochs, from scratch. The only change is the loss:

| metric (held-out test) | Cross-entropy | Focal + inverse-√ weights |
|---|---|---|
| macro-F1 | 0.820 | **0.883** |
| balanced accuracy | 0.800 | **0.863** |
| `crack` recall (rarest, 24 train imgs) | 0.67 | **0.83** |
| `scratch` F1 (hardest class) | 0.52 | **0.71** |
| defect **miss rate** (FN) | 16.4% | **13.1%** |
| **false-alarm rate** (FP) | 0.0% | 0.0% |

Miss rate and false-alarm rate are reported separately because in QC their costs differ — a
missed defect ships; a false alarm wastes an inspector's time. `evaluate.py` always breaks it out.

### 2. Real data — NEU-CLS steel surface, ResNet-18 + ImageNet transfer (easy problem)

1,800 real steel images, 6 defect classes (`scripts/prepare_neu.py` pulls it from figshare). NEU-CLS
is naturally balanced, so to exercise the imbalance handling I **induced a 7:1 train imbalance**
(`crazing` & `pitted_surface` capped to 30) while keeping val/test balanced for a fair comparison.

| metric (balanced test, 270 imgs) | CE | Focal + weights | balanced ref |
|---|---|---|---|
| macro-F1 | 0.9926 | **0.9963** | 0.9963 |
| `crazing` recall (rare) | 0.96 | **1.00** | 1.00 |

Honest read: a pretrained backbone is strong enough that even at 7:1 the CE baseline holds ~99%.
Focal+weights recovers the missed rare-class samples — a real but small (1–2 image) gain. **Big
deltas appear on hard problems (experiment 1), not easy ones.**

### 3. Detection — YOLOv8n on NEU-DET (real bounding boxes)

`scripts/prepare_neu_det.py` pulls NEU-DET (boxes) from GitHub, converts VOC→YOLO, stratified
80/20 split (1,440 train / 360 val). YOLOv8n, 50 epochs, 256 px, ~5 min on GPU:

| (360 val imgs, 805 boxes) | mAP@50 | mAP@50-95 |
|---|---|---|
| **all (6 classes)** | **0.767** | **0.452** |
| best — patches / scratches / pitted_surface | 0.93 / 0.91 / 0.89 | 0.64 / 0.58 / 0.55 |
| worst — `crazing` | 0.36 | 0.13 |

`crazing` (faint, diffuse) drags the mean down — consistent with published NEU-DET baselines.
Boxes visualised + ONNX-exported via `scripts/yolo_infer.py`.

### 4. Anomaly detection — PaDiM (train on *good* parts only)

`src/qc/anomaly.py` fits a per-position Gaussian over pretrained-CNN features of good images and
scores test images by Mahalanobis distance — **no defect labels needed**. On the synthetic
good-vs-defect split it reaches **image-level AUROC 1.0** (synthetic defects are visually obvious;
real MVTec AD categories land ~0.9–0.98) and produces a pixel-level anomaly heatmap. This is the
open-set route real QC prefers when you can't enumerate every defect in advance.

### 5. Confidence calibration (does "92%" mean 92%?)

`src/qc/calibrate.py` measures Expected Calibration Error, draws a reliability diagram, and fits
temperature scaling — because in QC confidence drives the route-to-human decision. Temperature
scaling roughly **halves ECE** (CE: 0.11→0.06 at T=0.68; focal: 0.24→0.06 at T=0.34). Both T<1:
trained from scratch with focal loss the model is *under*-confident, so scaling sharpens it — the
opposite of the usual overconfident-net story, reported honestly.

**Edge inference (ONNX, CPU, batch=1):** synthetic 96px model PyTorch 8.31 ms → **ONNXRuntime 1.84
ms/img (543 FPS)**; NEU 128px model 9.85 → **2.32 ms/img (432 FPS)**, outputs identical to ≤7.6e-6.
TensorRT is faster on NVIDIA HW but needs the TensorRT package; for YOLO use `yolo export format=engine`.

---

## A note on focal loss (correcting a common snippet)

The widely copy-pasted focal-loss code uses a **scalar** `alpha` (often `alpha=1`), which rescales
every class equally and does **nothing** for imbalance — only the `(1 - pt)^γ` term helps there. The
real lever is a **per-class** `alpha` weight vector. `src/qc/losses.py` supports both; a unit test
pins the scalar pitfall. (This CE-based focal loss is for the classification track — YOLO uses its
own BCE+DFL+CIoU loss, so it doesn't plug in there.)

---

## Quickstart

```bash
pip install -e .                          # installs the `qc` package (or: pip install -r requirements.txt)
# GPU training: install a CUDA-matched torch first (see requirements.txt)

python scripts/make_sample.py             # synthetic data/  (no downloads)
python src/qc/data.py --data data/sample  # audit class balance

python src/qc/train.py --data data/sample --loss ce    --no-pretrained --out outputs/clf_ce
python src/qc/train.py --data data/sample --loss focal --no-pretrained --out outputs/clf_focal
python src/qc/evaluate.py --ckpt outputs/clf_focal/best.pt   # per-class report + confusion matrix
python src/qc/explain.py  --ckpt outputs/clf_focal/best.pt --image data/sample/test/crack/crack_0000.png
python src/qc/export.py   --ckpt outputs/clf_focal/best.pt   # ONNX + parity + latency
python src/qc/anomaly.py  --data data/sample                 # PaDiM anomaly detection (AUROC + heatmap)
python src/qc/calibrate.py --ckpt outputs/clf_focal/best.pt  # ECE + reliability diagram + temp scaling
python src/qc/roi.py                                         # illustrative ROI model

streamlit run app.py                       # classify + detect + ROI demo
pytest -q                                  # 17 unit tests
```

### Detection track — YOLOv8 on real boxes
```bash
python scripts/prepare_neu_det.py            # NEU-DET -> YOLO format, stratified 80/20 split
python scripts/yolo_train.py --data data/neu_det/data.yaml --epochs 50 --imgsz 256
python scripts/yolo_infer.py --weights runs/detect/outputs/yolo/neu/weights/best.pt \
    --source data/neu_det/images/val --limit 6 --device cpu --onnx
```

### Use a real dataset
- **MVTec AD** — the standard anomaly benchmark (train on good only; pixel masks) → anomaly track.
- **NEU-DET / GC10-DET** — metal-surface defects with boxes → detection track.
- **Severstal** (Kaggle) — steel segmentation, naturally imbalanced.
- Any **Roboflow** YOLO export → point `configs/defect_detection.example.yaml` at it.

For classification, arrange images as `data/<your>/<split>/<class>/*.png`, pass `--data data/<your>`,
and drop `--no-pretrained` for ImageNet transfer learning.

---

## Business case (honest)
`src/qc/roi.py` is an **illustrative** labor-offset model: your assumptions in, a range out. It nets
the cost of *missed defects* (escapes) and *false alarms* against labor saved — so it can (and does, at
low recall / high escape cost) go **negative**. That's the point: automation isn't automatically
profitable. Defaults show ~$247k/yr net at 95% recall; raise the escape cost and watch it flip.

## Deploying
- A trained classifier (`models/clf_neu.pt`) and 6 sample images (`samples/`) are **committed**, so a
  fresh clone / cloud deploy works out of the box (`*.pt` and `data/` are otherwise git-ignored).
- **Streamlit Community Cloud** (CPU, ~1 GB RAM): use `requirements-deploy.txt` (CPU torch wheels,
  classification-only — `ultralytics` is heavy and imported lazily, so detection just won't load there).
- **Docker**: `docker build -t cv-qc . && docker run -p 8501:8501 cv-qc` (CPU image; `.dockerignore`
  keeps `data/`, `outputs/`, `runs/` out). *Image not built on this machine — verify locally.*

## Developing
- `pip install -e ".[dev]"`, then `pytest -q` — **17 unit tests** (focal-loss behaviour incl. the
  scalar-alpha pitfall, class weighting, VOC→YOLO conversion, the QC miss/false-alarm metric).
- GitHub Actions (`.github/workflows/ci.yml`) runs the fast tests on CPU for every push.
- `MODEL_CARD.md` documents data, metrics, intended use, and limitations.

## Honest limitations
- A model trained on one product line / camera / lighting will **not** generalise without retraining.
- NEU is an *easy* academic benchmark; ~99% / 0.77 mAP is not a noisy production line.
- Synthetic data proves the *pipeline*, not field accuracy.
- Out-of-distribution inputs get a confident-but-meaningless label — the demo's confidence gate routes
  those to a human. Grad-CAM shows *where* the model looked, not *why* a part is truly defective.

## Layout
```
src/qc/   losses.py data.py model.py train.py evaluate.py explain.py export.py utils.py
          anomaly.py (PaDiM)  calibrate.py (ECE+temp)  convert.py (VOC→YOLO)  roi.py (business case)
scripts/  make_sample.py  prepare_neu.py (NEU-CLS)  prepare_neu_det.py (NEU-DET→YOLO)
          yolo_train.py · yolo_infer.py (detection: train / predict+export)
models/   clf_neu.pt · yolo_neu.pt (committed)        samples/  6 demo images
tests/    test_losses/utils/metrics/convert.py (17 tests)        conftest.py
app.py    Streamlit demo (classify + detect + ROI)    configs/  YOLO data template
pyproject.toml · Dockerfile · MODEL_CARD.md · .github/workflows/ci.yml
```

## Stack
Python 3.12 · PyTorch 2.6 (CUDA) · timm · torchvision · Ultralytics YOLOv8 · scikit-learn ·
ONNX / ONNXRuntime · Streamlit · pytest + GitHub Actions CI · Docker. Verified on an RTX 4070 SUPER.
