"""Run the exported YOLOv8 detector with onnxruntime, no ultralytics or OpenCV at inference.

The Streamlit demo uses this for detection so the hosted app doesn't depend on cv2, which needs
system libraries that aren't always available on a free-tier image. Training and ONNX export still
use ultralytics (scripts/yolo_train.py, scripts/yolo_infer.py).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw
from torchvision.ops import nms

NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


class YoloOnnx:
    """Minimal YOLOv8 ONNX runner: preprocess, run, decode, NMS, draw boxes."""

    def __init__(self, path, imgsz=256, names=NEU_CLASSES):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.names = names

    def __call__(self, pil_img, conf=0.25, iou=0.45):
        pil_img = pil_img.convert("RGB")
        w0, h0 = pil_img.size
        resized = pil_img.resize((self.imgsz, self.imgsz))
        x = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

        out = self.session.run(None, {self.input_name: x})[0]  # (1, 4 + num_classes, N)
        preds = out[0].T  # (N, 4 + num_classes)
        boxes, scores = preds[:, :4], preds[:, 4:]
        cls = scores.argmax(1)
        confs = scores.max(1)

        keep = confs >= conf
        boxes, cls, confs = boxes[keep], cls[keep], confs[keep]
        annotated = pil_img.copy()
        if len(boxes) == 0:
            return annotated, []

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        idx = nms(torch.from_numpy(xyxy).float(), torch.from_numpy(confs).float(), iou).numpy()
        xyxy, cls, confs = xyxy[idx], cls[idx], confs[idx]

        xyxy[:, [0, 2]] *= w0 / self.imgsz  # scale boxes back to the original image
        xyxy[:, [1, 3]] *= h0 / self.imgsz

        draw = ImageDraw.Draw(annotated)
        dets = []
        for box, c, cf in zip(np.atleast_2d(xyxy), np.atleast_1d(cls), np.atleast_1d(confs)):
            draw.rectangle([float(v) for v in box], outline=(0, 200, 255), width=2)
            draw.text((float(box[0]), max(0.0, float(box[1]) - 10)), f"{self.names[int(c)]} {cf:.2f}", fill=(0, 200, 255))
            dets.append((self.names[int(c)], round(float(cf), 2)))
        return annotated, dets


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 ONNX inference, no ultralytics or cv2")
    ap.add_argument("--onnx", default="models/yolo_neu.onnx")
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="outputs/onnx_detect.jpg")
    args = ap.parse_args()

    annotated, dets = YoloOnnx(args.onnx)(Image.open(args.image), conf=args.conf)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.out)
    print(f"{len(dets)} detections: {dets}  ->  {args.out}")


if __name__ == "__main__":
    main()
