"""Generate the figures shown in the README into assets/.

Detection and Grad-CAM panels are built here from the committed models and data. The confusion
matrix, anomaly map, and calibration plots are copied from outputs/ if you've run evaluate.py /
anomaly.py / calibrate.py first.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image

from qc.explain import explain_path
from qc.model import build_model
from qc.utils import get_device

ASSETS = pathlib.Path("assets")
ASSETS.mkdir(exist_ok=True)


def hcat(items, height=256, pad=10, bg=(255, 255, 255)):
    imgs = [(im if isinstance(im, Image.Image) else Image.open(im)).convert("RGB") for im in items]
    imgs = [im.resize((max(1, round(im.width * height / im.height)), height)) for im in imgs]
    width = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    canvas = Image.new("RGB", (width, height), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + pad
    return canvas


def first(pattern):
    hits = sorted(pathlib.Path().glob(pattern))
    return hits[0] if hits else None


made = []

# Detection panel (input | predicted boxes), via the ONNX runner
try:
    from qc.yolo_onnx import YoloOnnx

    det = YoloOnnx("models/yolo_neu.onnx")
    src = Image.open(first("data/neu_det/images/val/scratches_*.jpg") or first("data/neu_det/images/val/*.jpg"))
    annotated, _ = det(src, conf=0.25)
    hcat([src, annotated]).save(ASSETS / "detection.png")
    made.append("detection.png")
except Exception as e:
    print("skip detection:", e)

# Grad-CAM panel (input | heatmap)
try:
    device = get_device()
    ck = torch.load("models/clf_neu.pt", map_location=device, weights_only=False)
    model = build_model(ck["num_classes"], ck["arch"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    path = first("data/neu/test/scratches/*.jpg") or first("data/neu/test/*/*.jpg")
    overlay_img, label, conf = explain_path(model, str(path), ck.get("img_size", 128), device, ck["class_names"])
    hcat([Image.open(path), overlay_img]).save(ASSETS / "gradcam.png")
    made.append(f"gradcam.png ({label} {conf:.2f})")
except Exception as e:
    print("skip gradcam:", e)

# Copy the plots produced by evaluate.py / anomaly.py / calibrate.py, if present
if pathlib.Path("outputs/neu_imb_ce/eval/confusion_matrix.png").exists():
    shutil.copy("outputs/neu_imb_ce/eval/confusion_matrix.png", ASSETS / "confusion_matrix.png")
    made.append("confusion_matrix.png")
if pathlib.Path("outputs/anomaly/anomaly_overlay.png").exists():
    hcat(["outputs/anomaly/anomaly_overlay.png"], height=256).save(ASSETS / "anomaly.png")
    made.append("anomaly.png")
rb, ra = "outputs/clf_ce/calib/reliability_before.png", "outputs/clf_ce/calib/reliability_after.png"
if pathlib.Path(rb).exists() and pathlib.Path(ra).exists():
    hcat([rb, ra], height=340).save(ASSETS / "calibration.png")
    made.append("calibration.png")

print("wrote:", made)
