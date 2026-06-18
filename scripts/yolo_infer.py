"""Run a trained YOLO defect detector: draw predicted boxes on image(s) and
optionally export the model to ONNX.

    # a directory (takes the first --limit images) or one/more explicit files
    python scripts/yolo_infer.py --weights runs/detect/outputs/yolo/neu/weights/best.pt \
        --source data/neu_det/images/val --limit 6 --device cpu --onnx
"""

from __future__ import annotations

import argparse
import pathlib

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description="YOLO defect inference + optional ONNX export")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", nargs="+", required=True, help="image file(s) or a directory")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=8, help="max images to take from a directory source")
    ap.add_argument("--out", default="outputs/yolo_pred")
    ap.add_argument("--device", default=None, help="e.g. 0 (GPU) or cpu")
    ap.add_argument("--onnx", action="store_true")
    args = ap.parse_args()

    # Resolve sources: expand a directory to its first --limit images; verify files exist.
    # (NEU-DET is split randomly, so don't assume a specific filename is in val.)
    sources = []
    for s in args.source:
        p = pathlib.Path(s)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)[: args.limit]
            if not files:
                ap.error(f"no images found in directory {s}")
            sources += [str(f) for f in files]
        elif p.is_file():
            sources.append(str(p))
        else:
            ap.error(f"source not found: {s}  (it may have landed in the train split, not val)")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    if args.onnx:
        print("exported ONNX ->", model.export(format="onnx", imgsz=args.imgsz))

    results = model.predict(
        source=sources, imgsz=args.imgsz, conf=args.conf, device=args.device,
        save=True, project=args.out, name="pred", exist_ok=True,
    )
    print("annotated images saved to:", model.predictor.save_dir)
    for r in results:
        dets = [(r.names[int(b.cls)], round(float(b.conf), 3)) for b in r.boxes]
        print(f"  {pathlib.Path(r.path).name}: {len(dets)} boxes {dets[:8]}")


if __name__ == "__main__":
    main()
