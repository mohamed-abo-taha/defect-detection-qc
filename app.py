"""Streamlit demo — manufacturing defect QC.

Two modes (classification + Grad-CAM, and YOLO detection), a confidence gate that routes
uncertain / out-of-distribution inputs to a human, and an illustrative ROI panel.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import streamlit as st
import torch
from PIL import Image

from qc.data import build_transforms
from qc.explain import GradCAM, overlay
from qc.model import build_model, last_conv_layer
from qc.roi import ROIInputs, estimate
from qc.utils import get_device

st.set_page_config(page_title="Defect QC demo", layout="centered")
st.title("Manufacturing defect QC — demo")
st.caption(
    "Assistive visual inspection, not a pass/fail certificate. Trained on a specific dataset (NEU "
    "steel surfaces); it will not generalise to unseen products, lighting or cameras without retraining."
)

mode = st.sidebar.radio("Mode", ["Classify (+ Grad-CAM)", "Detect (boxes)"])
gate = st.sidebar.slider(
    "Confidence gate", 0.0, 1.0, 0.60, 0.05,
    help="Below this confidence, route the part to a human inspector instead of trusting the label. "
    "Out-of-distribution images (not this product) usually fall below it.",
)


@st.cache_resource
def load_classifier(ckpt):
    device = get_device()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(ck["num_classes"], ck["arch"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    return model, ck, device


@st.cache_resource
def load_detector(weights):
    from ultralytics import YOLO  # lazy: classification-only deploys don't need ultralytics

    return YOLO(weights)


def run_classify(ckpt):
    up = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp"])
    if not up:
        st.write("Upload an image to classify.")
        return
    if not pathlib.Path(ckpt).exists():
        st.warning(f"No checkpoint at `{ckpt}`. Train one (`python src/qc/train.py`) or set the path.")
        return
    model, ck, device = load_classifier(ckpt)
    pil = Image.open(io.BytesIO(up.read())).convert("RGB")
    img_size, names = ck.get("img_size", 96), ck["class_names"]
    x = build_transforms(img_size, train=False)(pil).unsqueeze(0).to(device)
    gradcam = GradCAM(model, last_conv_layer(model))
    cam, idx, conf = gradcam(x)
    gradcam.remove()
    ov = overlay(pil.resize((img_size, img_size)), cam)

    c1, c2 = st.columns(2)
    c1.image(pil, caption="input", use_container_width=True)
    c2.image(ov, caption="Grad-CAM — where the model looked", use_container_width=True)
    if conf < gate:
        st.warning(
            f"⚠️ Uncertain — best guess **{names[idx]}** at {conf * 100:.1f}% "
            f"(below the {gate * 100:.0f}% gate). **Route to a human inspector.**"
        )
    else:
        st.metric("Prediction", names[idx], f"{conf * 100:.1f}% confidence")
    st.progress(min(1.0, float(conf)))
    st.caption(
        "The gate is the out-of-distribution guard: an unfamiliar image (wrong product/lighting, or "
        "not steel at all) usually lands below it — which is exactly when a human should look."
    )


def run_detect(weights):
    up = st.file_uploader("Upload a product image", type=["png", "jpg", "jpeg", "bmp"])
    if not up:
        st.write("Upload an image to detect defects.")
        return
    if not pathlib.Path(weights).exists():
        st.warning(f"No weights at `{weights}`. Train the detector (`scripts/yolo_train.py`) or set the path.")
        return
    try:
        model = load_detector(weights)
    except Exception:
        st.error(
            "Detection couldn't load here. It needs OpenCV and ultralytics, which don't always import "
            "on a hosted free-tier image. Use Classify mode, or run detection locally with "
            "`scripts/yolo_infer.py`."
        )
        return
    pil = Image.open(io.BytesIO(up.read())).convert("RGB")
    res = model.predict(pil, conf=max(0.05, gate * 0.4), device="cpu", verbose=False)[0]
    annotated = res.plot()[:, :, ::-1]  # ultralytics returns BGR; flip to RGB
    c1, c2 = st.columns(2)
    c1.image(pil, caption="input", use_container_width=True)
    c2.image(annotated, caption="detections", use_container_width=True)
    dets = [(res.names[int(b.cls)], round(float(b.conf), 2)) for b in res.boxes]
    st.write(f"**{len(dets)} detection(s):** {dets}" if dets else "No defects above the threshold.")


if mode.startswith("Classify"):
    run_classify(st.sidebar.text_input("Classifier checkpoint", "models/clf_neu.pt"))
else:
    run_detect(st.sidebar.text_input("YOLO weights", "models/yolo_neu.pt"))

with st.sidebar.expander("💰 ROI model (illustrative)"):
    st.caption("A transparent what-if, not a guarantee — your assumptions in, a range out.")
    ppd = st.number_input("Parts / day", 1_000, 1_000_000, 20_000, step=1_000)
    recall = st.slider("Model defect recall", 0.50, 1.00, 0.90, 0.01)
    fa = st.slider("False-alarm rate", 0.0, 0.20, 0.02, 0.01)
    r = estimate(ROIInputs(parts_per_day=int(ppd), model_defect_recall=recall, model_false_alarm_rate=fa))
    st.metric("Net saving / year", f"${r['net_saving_per_year']:,.0f}")
    st.caption(
        f"labor saved/day ${r['labor_saved_per_day']:,.0f} − escape cost/day "
        f"${r['escape_cost_per_day']:,.0f} = net/day ${r['net_saving_per_day']:,.0f}"
    )

st.caption(
    "Automated visual inspection assists people; it does not replace them. Automated tools catch only "
    "a fraction of real-world defects — measure on your own line before trusting any number."
)
