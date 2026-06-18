"""YOLOv8 detection track — localises & classifies defects with bounding boxes.

This is the article's headline approach and the right tool when you have a *detection*
dataset (boxes), e.g. NEU-DET, GC10-DET, or a Roboflow export in YOLO format.

A note on focal loss: Ultralytics YOLO uses its own composite loss (BCE for class +
Distribution Focal Loss + CIoU for boxes), so the cross-entropy FocalLoss in
qc/losses.py does NOT plug into it. For class imbalance in detection, prefer
class-balanced sampling, the ``cls`` loss-weight, and augmentation instead. Use
qc/losses.py for the classification track.

Usage:
    python scripts/yolo_train.py --data configs/defect_detection.example.yaml --epochs 100
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser(description="Train YOLOv8 on a defect-detection dataset")
    ap.add_argument("--data", required=True, help="path to an Ultralytics YOLO data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights (auto-downloaded on first use)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", default="outputs/yolo")
    ap.add_argument("--name", default="exp")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    metrics = model.val()
    print(f"mAP50-95={getattr(metrics.box, 'map', None)}  mAP50={getattr(metrics.box, 'map50', None)}")
    print("Per-class precision/recall are in the run dir; export with: yolo export model=<best.pt> format=onnx")


if __name__ == "__main__":
    main()
