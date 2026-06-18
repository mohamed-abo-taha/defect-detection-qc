"""Export the trained classifier to ONNX, verify numerical parity, and benchmark
CPU latency (PyTorch vs ONNXRuntime).

ONNX is the portable, dependency-light path to "edge" deployment that runs anywhere
ONNXRuntime does. TensorRT (the article's other suggestion) gives the best latency on
NVIDIA hardware but needs the TensorRT package + a GPU; for the YOLO detection track
you can also do ``yolo export format=engine``. We report measured latency honestly
rather than claiming a speedup.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from qc.model import build_model


def export_onnx(model, img_size, path):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return path


def _bench(fn, runs):
    for _ in range(5):  # warm-up
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) / runs * 1000.0


@torch.no_grad()
def _torch_infer(model, x):
    return model(x)


def main():
    ap = argparse.ArgumentParser(description="Export classifier to ONNX + benchmark")
    ap.add_argument("--ckpt", default="outputs/clf/best.pt")
    ap.add_argument("--out", default=None, help="defaults to <ckpt dir>/model.onnx")
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    img_size = ck.get("img_size", 96)
    out_path = str(args.out or (pathlib.Path(args.ckpt).parent / "model.onnx"))

    model = build_model(ck["num_classes"], ck["arch"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.eval()

    export_onnx(model, img_size, out_path)
    print(f"exported ONNX -> {out_path}")

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(out_path))
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        torch_out = model(x).numpy()
    onnx_out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"torch vs onnxruntime  max|delta| = {max_diff:.2e}  ({'OK' if max_diff < 1e-3 else 'MISMATCH'})")

    torch_ms = _bench(lambda: _torch_infer(model, x), args.runs)
    onnx_ms = _bench(lambda: sess.run(None, {"input": x.numpy()}), args.runs)
    print(f"\nLatency (CPU, batch=1, {args.runs} runs):")
    print(f"  PyTorch      : {torch_ms:6.2f} ms/img   ({1000 / torch_ms:6.1f} FPS)")
    print(f"  ONNXRuntime  : {onnx_ms:6.2f} ms/img   ({1000 / onnx_ms:6.1f} FPS)")


if __name__ == "__main__":
    main()
