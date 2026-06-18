"""Grad-CAM explainability — implemented from scratch (no extra dependency).

Why it matters for QC: a prediction is only trustworthy if the model looked at the
actual defect. Grad-CAM highlights the pixels that most influenced the predicted
class. If the heatmap ignores the visible defect (or fires on the background), that's
a signal to route the part to a human inspector rather than trust the label.

Reference: Selvaraju et al., "Grad-CAM", ICCV 2017.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qc.data import build_transforms
from qc.model import build_model, last_conv_layer
from qc.utils import get_device


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pool the gradients
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        conf = logits.softmax(1)[0, class_idx].item()
        return cam.cpu().numpy(), class_idx, conf


def overlay(pil_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = cm.jet(cam)[..., :3]
    base = np.asarray(pil_img.convert("RGB").resize((cam.shape[1], cam.shape[0]))) / 255.0
    blended = (1 - alpha) * base + alpha * heat
    return Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8))


def explain_path(model, img_path, img_size, device, class_names=None, out_path=None):
    pil = Image.open(img_path).convert("RGB")
    x = build_transforms(img_size, train=False)(pil).unsqueeze(0).to(device)
    gradcam = GradCAM(model, last_conv_layer(model))
    cam, idx, conf = gradcam(x)
    gradcam.remove()
    ov = overlay(pil.resize((img_size, img_size)), cam)
    if out_path:
        ov.save(out_path)
    label = class_names[idx] if class_names else str(idx)
    return ov, label, conf


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM for one image")
    ap.add_argument("--ckpt", default="outputs/clf/best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="outputs/clf/gradcam.png")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ck["num_classes"], ck["arch"], pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()

    _, label, conf = explain_path(
        model, args.image, ck.get("img_size", 96), device, ck["class_names"], args.out
    )
    print(f"prediction={label}  confidence={conf:.3f}  ->  {args.out}")


if __name__ == "__main__":
    main()
